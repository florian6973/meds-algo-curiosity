"""Curiosity event tokenization (paper Sec. 5.2, Table 2), fitted on the training split only.

A subject becomes ``[sex] [race] [age] [BOS]`` followed by its events in time order. Before the first event of
each new timestamp come ETHOS-style time-gap tokens (none below one minute; one or more "6 month" tokens,
``round(gap / 180 d)``, from six months on). An event is its code token(s) (an ICD code as 3/2/2-character
sub-tokens) followed, for numeric categories, by one of ``num_quantiles`` decile tokens shared across codes.
Generated time tokens advance the clock by the geometric mean of their bucket (paper Sec. 5.5).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from .config import CuriosityConfig

PAD, SEP, BOS, UNK = "[PAD]", "[SEP]", "[BOS]", "[UNK]"
DEATH_CODE = "MEDS_DEATH"
_MINUTE, _DAY = 60.0, 86400.0
# ETHOS buckets (ethos-paper special_tokens.py, 30-day months) with the paper's extra "1-5 minutes" bucket.
TIME_BUCKETS = (
    ("1m-5m", 1 * _MINUTE, 5 * _MINUTE),
    ("5m-15m", 5 * _MINUTE, 15 * _MINUTE),
    ("15m-1h", 15 * _MINUTE, 60 * _MINUTE),
    ("1h-2h", 60 * _MINUTE, 120 * _MINUTE),
    ("2h-6h", 2 * 3600.0, 6 * 3600.0),
    ("6h-12h", 6 * 3600.0, 12 * 3600.0),
    ("12h-1d", 12 * 3600.0, _DAY),
    ("1d-3d", _DAY, 3 * _DAY),
    ("3d-1w", 3 * _DAY, 7 * _DAY),
    ("1w-2w", 7 * _DAY, 14 * _DAY),
    ("2w-1mt", 14 * _DAY, 30 * _DAY),
    ("1mt-3mt", 30 * _DAY, 90 * _DAY),
    ("3mt-6mt", 90 * _DAY, 180 * _DAY),
)
SIX_MONTHS = 180 * _DAY
_EVENT_COLUMNS = ["subject_id", "time", "code", "numeric_value"]


def _time_token(name: str) -> str:
    return f"TIME//{name}"


def _split_icd(code: str) -> tuple[str, str, str]:
    """``DIAGNOSIS//ICD//10//E1165`` -> head ``DIAGNOSIS//ICD//10//E11`` and sub-tokens ``65`` and ``""``."""
    prefix, _, value = code.rpartition("//")
    return f"{prefix}//{value[:3]}", value[3:5], value[5:7]


def gap_tokens(seconds: float) -> list[str]:
    """Time tokens for a gap between consecutive timestamps."""
    if seconds >= SIX_MONTHS:
        return [_time_token("6mt")] * max(1, round(seconds / SIX_MONTHS))
    for name, low, high in TIME_BUCKETS:
        if low <= seconds < high:
            return [_time_token(name)]
    return []


def _token_seconds(name: str) -> float:
    if name == _time_token("6mt"):
        return SIX_MONTHS
    low, high = next((low, high) for bucket, low, high in TIME_BUCKETS if _time_token(bucket) == name)
    return math.sqrt(low * high)


def _categorize(config: CuriosityConfig) -> pl.Expr:
    expression = pl.lit(None, dtype=pl.String)
    for category in reversed(config.categories):
        expression = pl.when(pl.col("code").str.contains(category.pattern)).then(pl.lit(category.name)).otherwise(expression)
    return expression.alias("category")


def _kept_events(frame: pl.LazyFrame, config: CuriosityConfig) -> pl.LazyFrame:
    numeric = [category.name for category in config.categories if category.numeric]
    return (
        frame.with_columns(_categorize(config))
        .filter(pl.col("category").is_not_null())
        .filter(~pl.col("category").is_in(numeric) | (pl.col("numeric_value").is_not_null() & pl.col("numeric_value").is_finite()))
    )


@dataclass(frozen=True)
class _Vocabulary:
    """Fitted tokenizer state: token strings (index = id), per-code decile edges and training codes."""

    tokens: tuple[str, ...]
    quantiles: dict[str, tuple[float, ...]]  # code -> (min, cut_1..cut_{q-1}, max) on training values
    split_codes: tuple[str, ...]  # training ICD codes represented by sub-tokens
    num_quantiles: int
    age_bucket_years: int

    @classmethod
    def fit(cls, train: pl.LazyFrame, config: CuriosityConfig, race: pl.LazyFrame | None) -> _Vocabulary:
        kept = _kept_events(train, config)
        counts = kept.group_by("category", "code").agg(pl.len().alias("count")).collect()
        codes: list[str] = []
        split_codes: list[str] = []
        for category in config.categories:
            ranked = counts.filter(pl.col("category") == category.name).sort(["count", "code"], descending=[True, False])
            selected = ranked.head(category.top_k) if category.top_k else ranked
            (split_codes if category.split_icd else codes).extend(sorted(selected["code"].to_list()))
        numeric_codes = set(
            counts.filter(pl.col("category").is_in([c.name for c in config.categories if c.numeric]))["code"]
        ) & set(codes)
        q = config.num_quantiles
        quantiles = {}
        if numeric_codes:
            stats = (
                kept.filter(pl.col("code").is_in(sorted(numeric_codes)))
                .group_by("code")
                .agg(
                    pl.col("numeric_value").min().alias("min"),
                    *(pl.col("numeric_value").quantile(i / q, "linear").alias(f"c{i}") for i in range(1, q)),
                    pl.col("numeric_value").max().alias("max"),
                )
                .collect()
            )
            columns = ["min", *(f"c{i}" for i in range(1, q)), "max"]
            quantiles = {row["code"]: tuple(float(row[c]) for c in columns) for row in stats.iter_rows(named=True)}
        heads, subs = set(), set()
        for code in split_codes:
            head, middle, tail = _split_icd(code)
            heads.add(head)
            subs.update(f"ICD//{position}//{part}" for position, part in (("4-5", middle), ("6-7", tail)) if part)
        races = []
        if race is not None:
            races = sorted(race.select(pl.col("race").drop_nulls().unique()).collect()["race"].to_list())
        tokens = (
            PAD, SEP, BOS, UNK,
            *(_time_token(name) for name, _, _ in TIME_BUCKETS), _time_token("6mt"),
            *(f"Q//{i}" for i in range(1, q + 1)),
            *(f"AGE//{low}" for low in range(0, 125, config.age_bucket_years)),
            *(f"RACE//{value}" for value in races),
            *(f"CODE//{code}" for code in codes),
            *(f"CODE//{head}" for head in sorted(heads)),
            *sorted(subs),
        )
        return cls(tuple(tokens), quantiles, tuple(split_codes), q, config.age_bucket_years)

    # ---- persistence -------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "tokens": list(self.tokens),
            "quantiles": {code: list(edges) for code, edges in self.quantiles.items()},
            "split_codes": list(self.split_codes),
            "num_quantiles": self.num_quantiles,
            "age_bucket_years": self.age_bucket_years,
        }, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> _Vocabulary:
        value = json.loads(path.read_text())
        return cls(
            tuple(value["tokens"]),
            {code: tuple(edges) for code, edges in value["quantiles"].items()},
            tuple(value["split_codes"]),
            value["num_quantiles"],
            value["age_bucket_years"],
        )

    # ---- lookups -----------------------------------------------------------------------------------------
    @property
    def ids(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def codes(self) -> set[str]:
        """Every MEDS code with a fitted token or decile statistic."""
        return {token[6:] for token in self.tokens if token.startswith("CODE//")} - {
            _split_icd(code)[0] for code in self.split_codes
        } | set(self.split_codes) | set(self.quantiles)

    # ---- transform ---------------------------------------------------------------------------------------
    def transform(self, events: pl.DataFrame, config: CuriosityConfig, race: pl.DataFrame | None) -> pl.DataFrame:
        """One row per subject: ``tokens`` (ids) and ``times`` (event time per token; null for the prefix)."""
        ids = self.ids
        births = (
            events.filter(pl.col("code") == "MEDS_BIRTH").group_by("subject_id").agg(pl.col("time").min().alias("birth"))
        )
        kept = _kept_events(events.lazy(), config).collect()
        timed = kept.filter(pl.col("time").is_not_null()).with_row_index("_order").sort(
            "subject_id", "time", "_order", maintain_order=True
        )
        code_map = pl.DataFrame(
            {"code": list(self._code_tokens()), "code_tokens": list(self._code_tokens().values())},
            schema={"code": pl.String, "code_tokens": pl.List(pl.Int32)},
        )
        q = self.num_quantiles
        cuts = pl.DataFrame(
            {"code": list(self.quantiles), **{f"c{i}": [edges[i] for edges in self.quantiles.values()] for i in range(1, q)}},
            schema={"code": pl.String, **{f"c{i}": pl.Float64 for i in range(1, q)}},
        )
        gap = (pl.col("time") - pl.col("time").shift(1).over("subject_id")).dt.total_microseconds() / 1e6
        new_time = pl.col("time") != pl.col("time").shift(1).over("subject_id")
        bucket = pl.sum_horizontal([(pl.col("_gap") >= low).cast(pl.Int32) for _, low, _ in TIME_BUCKETS])
        bucket_ids = [ids[_time_token(name)] for name, _, _ in TIME_BUCKETS]
        timed = (
            timed.with_columns(pl.when(new_time).then(gap).otherwise(None).alias("_gap"))
            .join(code_map, on="code", how="left")
            .join(cuts, on="code", how="left")
            .with_columns(
                pl.when(pl.col("_gap") >= SIX_MONTHS)
                .then(pl.lit(ids[_time_token("6mt")], dtype=pl.Int32).repeat_by(
                    (pl.col("_gap") / SIX_MONTHS).round().cast(pl.Int64).clip(lower_bound=1)))
                .when(pl.col("_gap").is_not_null() & (bucket > 0))
                .then(pl.concat_list(bucket.replace_strict(range(1, len(bucket_ids) + 1), bucket_ids, default=None, return_dtype=pl.Int32)))
                .otherwise(pl.lit([], dtype=pl.List(pl.Int32)))
                .alias("_gap_tokens"),
                pl.col("code_tokens").fill_null(pl.lit([ids[UNK]], dtype=pl.List(pl.Int32))),
                pl.when(pl.col("c1").is_not_null())
                .then(pl.concat_list(
                    (1 + pl.sum_horizontal([(pl.col("numeric_value") > pl.col(f"c{i}")).cast(pl.Int32) for i in range(1, q)]))
                    .replace_strict(range(1, q + 1), [ids[f"Q//{i}"] for i in range(1, q + 1)], return_dtype=pl.Int32)
                ))
                .otherwise(pl.lit([], dtype=pl.List(pl.Int32)))
                .alias("_value_tokens"),
            )
            .with_columns(pl.concat_list("_gap_tokens", "code_tokens", "_value_tokens").alias("_tokens"))
            .with_columns(pl.col("time").repeat_by(pl.col("_tokens").list.len()).alias("_times"))
            .group_by("subject_id", maintain_order=True)
            .agg(pl.col("_tokens").list.explode(keep_nulls=False, empty_as_null=False), pl.col("_times").list.explode(keep_nulls=False, empty_as_null=False), pl.col("time").min().alias("_first"))
        )
        subjects = events.select(pl.col("subject_id").unique().sort())
        static = (
            kept.filter(pl.col("time").is_null()).join(code_map, on="code", how="inner")
            .group_by("subject_id", maintain_order=True).agg(pl.col("code_tokens").list.explode(keep_nulls=False, empty_as_null=False).alias("_static"))
        )
        prefix = subjects.join(static, on="subject_id", how="left").join(timed.select("subject_id", "_first"), on="subject_id", how="left")
        prefix = prefix.join(births, on="subject_id", how="left")
        age_years = (pl.col("_first") - pl.col("birth")).dt.total_days() / 365.25
        age_index = (age_years // self.age_bucket_years * self.age_bucket_years).clip(0, 120).cast(pl.Int64)
        age_names = list(range(0, 125, self.age_bucket_years))
        prefix = prefix.with_columns(
            pl.when(age_years.is_not_null())
            .then(pl.concat_list(age_index.replace_strict(age_names, [ids[f"AGE//{a}"] for a in age_names], default=None, return_dtype=pl.Int32)))
            .otherwise(pl.lit([], dtype=pl.List(pl.Int32))).alias("_age")
        )
        race_tokens = pl.lit([], dtype=pl.List(pl.Int32))
        if race is not None:
            race_ids = race.with_columns(
                pl.col("race").replace_strict(
                    [t[6:] for t in self.tokens if t.startswith("RACE//")],
                    [ids[t] for t in self.tokens if t.startswith("RACE//")],
                    default=None, return_dtype=pl.Int32,
                ).alias("_race_id")
            ).select("subject_id", "_race_id")
            prefix = prefix.join(race_ids, on="subject_id", how="left")
            race_tokens = pl.when(pl.col("_race_id").is_not_null()).then(pl.concat_list("_race_id")).otherwise(race_tokens)
        prefix = prefix.with_columns(
            pl.concat_list(
                pl.col("_static").fill_null(pl.lit([], dtype=pl.List(pl.Int32))), race_tokens, "_age",
                pl.lit([ids[BOS]], dtype=pl.List(pl.Int32)),
            ).alias("_prefix")
        )
        out = prefix.select("subject_id", "_prefix").join(timed.select("subject_id", "_tokens", "_times"), on="subject_id", how="left")
        return out.select(
            "subject_id",
            pl.concat_list("_prefix", pl.col("_tokens").fill_null(pl.lit([], dtype=pl.List(pl.Int32)))).alias("tokens"),
            pl.concat_list(
                pl.lit(None, dtype=pl.Datetime("us")).repeat_by(pl.col("_prefix").list.len()),
                pl.col("_times").fill_null(pl.lit([], dtype=pl.List(pl.Datetime("us")))).cast(pl.List(pl.Datetime("us"))),
            ).alias("times"),
        ).sort("subject_id")

    def _code_tokens(self) -> dict[str, list[int]]:
        ids = self.ids
        mapping = {token[6:]: [index] for token, index in ids.items() if token.startswith("CODE//")}
        for head, _, _ in map(_split_icd, self.split_codes):
            mapping.pop(head, None)
        for code in self.split_codes:
            head, middle, tail = _split_icd(code)
            mapping[code] = [ids[f"CODE//{head}"]] + [
                ids[f"ICD//{position}//{part}"] for position, part in (("4-5", middle), ("6-7", tail)) if part
            ]
        return mapping

    # ---- generation --------------------------------------------------------------------------------------
    def prompt_gap(self, seconds: float) -> list[int]:
        ids = self.ids
        return [ids[name] for name in gap_tokens(seconds)]

    def decode(self, tokens: list[int], prediction_time: datetime, uniforms: np.ndarray) -> tuple[list[tuple], datetime, int]:
        """Generated ids -> (``(time, code, numeric_value)`` events, final clock, invalid-token count).

        The clock starts at the prediction time (the prompt already ends with the real elapsed gap); generated
        events sharing it are placed one microsecond later, strictly after the prediction time. A decile token
        draws its value uniformly within the bucket's training range (paper App. G).
        """
        heads = {_split_icd(code)[0] for code in self.split_codes}
        next_part = {"ICD//4-5//": 3, "ICD//6-7//": 5}  # sub-token -> ICD characters it must follow
        clock, events, invalid = prediction_time, [], 0
        future = prediction_time + timedelta(microseconds=1)
        code, value, icd_length = None, None, None

        def flush():
            nonlocal code, value, icd_length, invalid
            if code is not None:
                invalid += code in self.quantiles and value is None  # a lab without its decile
                events.append((max(clock, future), code, value))
            code, value, icd_length = None, None, None

        for position, index in enumerate(tokens):
            token = self.tokens[index]
            if token.startswith("TIME//"):
                flush()
                clock += timedelta(seconds=_token_seconds(token))
            elif token.startswith("CODE//"):
                flush()
                code = token[6:]
                icd_length = 3 if code in heads else None
            elif token.startswith("ICD//"):
                position_prefix = token[:10]
                if icd_length is not None and next_part[position_prefix] == icd_length:
                    part = token[10:]
                    code, icd_length = code + part, icd_length + len(part)
                else:
                    invalid += 1
            elif token.startswith("Q//"):
                if code in self.quantiles and value is None:
                    edges, bucket = self.quantiles[code], int(token[3:])
                    value = float(edges[bucket - 1] + uniforms[position] * (edges[bucket] - edges[bucket - 1]))
                else:
                    invalid += 1
            else:  # special, age or race token: no event
                flush()
        flush()
        return events, clock, invalid
