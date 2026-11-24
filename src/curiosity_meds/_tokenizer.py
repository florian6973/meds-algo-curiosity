"""Curiosity event tokenization, dataset-agnostic (arXiv 2611.15710 Sec. 2 and 2508.12104v3 Sec. 5.2, adapted: see
IMPLEMENTATION_REPORT.md), fitted on the training split only.

Every MEDS code is one whole-code token: no event categories, no per-dataset mapping and no code-hierarchy split. A
code with enough numeric values is followed by one of 30 shared percentile-bucket tokens (the paper's lab buckets); a
value-less row of a code with text values by a result token; a row carrying the optional ``route``, ``dose`` or
``frequency`` columns by attribute tokens. Codes below one frequency threshold become ``[UNK]``, as in the paper;
rarer result and attribute values become ``RESULT//[OTHER]`` or are dropped. Only MEDS standard codes are special:
``MEDS_BIRTH`` dates the age token and ``MEDS_DEATH`` is placed at the end of its date and closes the record with
``[EOS]``.

Preprocessing stores *events*, not sequences: one row per event with its token ids, plus per-subject demographics.
:func:`_Vocabulary.sequences` turns them into token sequences at a cutoff. Without a cutoff it gives the training
sequence; with the prediction time ``T`` it gives the prompt, which depends only on what is recorded at or before T:

``[BOS] [static codes] [race] [age at start] [year at start]`` then the visible events in time order, with ETHOS time
tokens before each new timestamp, then (training only) ``[EOS]`` after a death or six-month padding tokens up to the
right-censoring time; a prompt instead ends with the time tokens of the elapsed gap to ``T``.

Everything that could depend on the future is evaluated at the cutoff: the start of the timeline (age and year
tokens), race (the first race recorded by then), and the downsampling of numeric codes (a value is dropped only once
the later value of its window is itself visible). Padding never enters a prompt.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl

from .config import CuriosityConfig

PAD, BOS, EOS, UNK = "[PAD]", "[BOS]", "[EOS]", "[UNK]"
BIRTH_CODE, DEATH_CODE = "MEDS_BIRTH", "MEDS_DEATH"
_MINUTE, _DAY = 60.0, 86400.0
# ETHOS buckets (ethos-paper special_tokens.py, 30-day months) with the 0226 paper's extra "1-5 minutes" bucket;
# unchanged in 0826.
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

# Value buckets (0826 Sec. 2.3.4): two 0.1% tails and 28 interior buckets whose edges are equal-width in z between the
# 0.1st and 99.9th percentiles, converted to percentile levels shared by every code.
_Z = NormalDist().inv_cdf(0.001)
LAB_LEVELS = tuple(NormalDist().cdf(_Z + i * (-2 * _Z) / 28) for i in range(29))
LAB_BUCKETS = len(LAB_LEVELS) + 1  # 30
ATTRIBUTES = ("route", "dose", "frequency")  # optional MEDS columns -> attribute tokens (0826 Sec. 2.6)
_EVENT_COLUMNS = ["subject_id", "time", "code", "numeric_value", "text_value", *ATTRIBUTES]


def _time_token(name: str) -> str:
    return f"TIME//{name}"


def gap_tokens(seconds: float) -> list[str]:
    """Time tokens for a gap between consecutive timestamps (none below one minute)."""
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


def age_token(days: float) -> str:
    """Age at the start of the timeline (0826 Sec. 2.1): days in week 1, weeks to 4 weeks, 4-week steps to 1 year,
    6-month steps to 5 years, then years (up to 120)."""
    days = max(days, 0.0)
    if days < 7:
        return f"AGE//{int(days)}d"
    if days < 28:
        return f"AGE//{int(days // 7)}w"
    if days < 365.25:
        return f"AGE//{4 * int(days // 28)}w"
    years = days / 365.25
    if years < 5:
        return f"AGE//{6 * int(years * 2)}mo"
    return f"AGE//{min(int(years), 120)}y"


AGE_TOKENS = tuple(dict.fromkeys(age_token(d) for d in (*range(0, 365), *np.arange(365.25, 121 * 365.25, 30.0))))


def year_token(year: int) -> str:
    low = year // 5 * 5
    return f"YEAR//{low}-{low + 4}"


def _with_columns(frame: pl.LazyFrame) -> pl.LazyFrame:
    schema = frame.collect_schema()
    missing = [pl.lit(None, dtype=pl.String).alias(c) for c in _EVENT_COLUMNS if c not in schema]
    return frame.with_columns(missing).select(
        pl.col("subject_id").cast(pl.Int64), pl.col("time").cast(pl.Datetime("us")), pl.col("code").cast(pl.String),
        pl.col("numeric_value").cast(pl.Float64), pl.col("text_value").cast(pl.String),
        *(pl.col(c).cast(pl.String) for c in ATTRIBUTES),
    ).with_row_index("_order")


def _numeric() -> pl.Expr:
    return pl.col("numeric_value").is_not_null() & pl.col("numeric_value").is_finite()


def _supersede(frame: pl.DataFrame, key: str, window_us: int | None) -> pl.DataFrame:
    """``_superseded``: the time of the next value of the same ``key`` in the same clock-aligned window, if any.

    A value is dropped from a sequence only once that later value is visible (at or before the cutoff), so the
    latest value per window is kept without ever looking past the cutoff. ``None`` disables downsampling.
    """
    if not window_us:
        return frame.with_columns(pl.lit(None, dtype=pl.Datetime("us")).alias("_superseded"))
    return (
        frame.with_columns((pl.col("time").dt.epoch("us") // window_us).alias("_window"))
        .sort("subject_id", key, "time", "_order")
        .with_columns(
            pl.when(pl.col("_window").shift(-1).over("subject_id", key) == pl.col("_window"))
            .then(pl.col("time").shift(-1).over("subject_id", key)).alias("_superseded")
        )
    )


def _races(column: pl.DataFrame | None) -> pl.DataFrame:
    """First recorded race per subject and its time (null for a static record), from an optional race column
    (MIMIC-IV-MEDS 0.1.0 admission rows). Race recorded as codes (``RACE//<race>``) is an ordinary event."""
    schema = {"subject_id": pl.Int64, "race": pl.String, "time": pl.Datetime("us")}
    if column is None or not column.height:
        return pl.DataFrame(schema=schema)
    frame = column.select(pl.col("subject_id").cast(pl.Int64), pl.col("race").cast(pl.String),
                          pl.col("time").cast(pl.Datetime("us"))).filter(pl.col("race").is_not_null())
    return frame.sort("subject_id", "time", "race", nulls_last=False).group_by("subject_id", maintain_order=True).first()


@dataclass(frozen=True)
class _Vocabulary:
    """Fitted tokenizer state (training split only)."""

    tokens: tuple[str, ...]
    quantiles: dict[str, tuple[float, ...]]  # code -> (min, q(levels)..., max): value buckets
    text_codes: tuple[str, ...]  # codes whose value-less rows get result tokens
    cutoff: str  # latest training time (ISO): right-censoring bound

    # ---- fitting -----------------------------------------------------------------------------------------------
    @classmethod
    def fit(cls, train: pl.LazyFrame, config: CuriosityConfig, race: pl.LazyFrame | None) -> _Vocabulary:
        rows = _with_columns(train).filter(pl.col("code") != BIRTH_CODE).collect()
        timed = rows.filter(pl.col("time").is_not_null())
        floor = max(config.min_code_count, config.min_code_share * timed.height)

        def kept(values: pl.Series) -> list[str]:
            """Values seen at least ``floor`` times (one threshold for codes, result and attribute values)."""
            counts = values.drop_nulls().value_counts()
            return sorted(counts.filter(pl.col("count") >= floor)[values.name].to_list())

        codes = kept(rows["code"])
        in_vocabulary = rows.filter(pl.col("code").is_in(codes))
        numeric = in_vocabulary.filter(_numeric())
        bucketed = numeric.group_by("code").len().filter(pl.col("len") >= config.min_bucket_observations)["code"]
        text = in_vocabulary.filter(pl.col("numeric_value").is_null() & pl.col("text_value").is_not_null())
        text_codes = kept(text["code"])
        results = kept(text.filter(pl.col("code").is_in(text_codes))["text_value"])
        attributes = [f"ATTR//{c.upper()}//{v}" for c in ATTRIBUTES for v in kept(in_vocabulary[c])]
        quantiles = {}
        if len(bucketed):
            stats = numeric.filter(pl.col("code").is_in(bucketed.implode())).group_by("code").agg(
                pl.col("numeric_value").min().alias("min"),
                *(pl.col("numeric_value").quantile(level, "linear").alias(f"q{i}") for i, level in enumerate(LAB_LEVELS)),
                pl.col("numeric_value").max().alias("max"),
            )
            columns = ["min", *(f"q{i}" for i in range(len(LAB_LEVELS))), "max"]
            quantiles = {row["code"]: tuple(float(row[c]) for c in columns) for row in stats.iter_rows(named=True)}
        races = sorted(_races(race.collect() if race is not None else None)["race"].unique().to_list())
        starts = timed.group_by("subject_id").agg(pl.col("time").min())["time"].dt.year()
        years = sorted({year_token(y) for y in range(int(starts.min()), int(starts.max()) + 1)}) if len(starts) else []
        tokens = (
            PAD, BOS, EOS, UNK,
            *(_time_token(name) for name, _, _ in TIME_BUCKETS), _time_token("6mt"),
            *AGE_TOKENS, *years, *(f"RACE//{value}" for value in races),
            *(f"LABQ//{i}" for i in range(1, LAB_BUCKETS + 1)),
            *(f"RESULT//{v}" for v in results), "RESULT//[OTHER]", "RESULT//[MISSING]",
            *attributes,
            *(f"CODE//{code}" for code in codes),
        )
        cutoff = timed["time"].max()
        return cls(tuple(dict.fromkeys(tokens)), quantiles, tuple(text_codes),
                   (cutoff or datetime(1970, 1, 1)).isoformat())

    # ---- persistence ---------------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "tokens": list(self.tokens),
            "quantiles": {code: list(edges) for code, edges in self.quantiles.items()},
            "text_codes": list(self.text_codes),
            "cutoff": self.cutoff,
        }, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> _Vocabulary:
        v = json.loads(path.read_text())
        return cls(tuple(v["tokens"]), {code: tuple(edges) for code, edges in v["quantiles"].items()},
                   tuple(v["text_codes"]), v["cutoff"])

    # ---- lookups -------------------------------------------------------------------------------------------------
    @property
    def ids(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def codes(self) -> set[str]:
        """Every MEDS code with a token in the fitted vocabulary."""
        return {token[6:] for token in self.tokens if token.startswith("CODE//")}

    # ---- events --------------------------------------------------------------------------------------------------
    def events(self, frame: pl.DataFrame, race: pl.DataFrame | None, config: CuriosityConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
        """``(events, subjects)`` for one shard.

        ``events``: ``subject_id, time, order, tokens, superseded`` (one row per event; ``superseded`` is the time of
        the later value of the same numeric code in the same window, which hides this one once visible).
        ``subjects``: ``subject_id, static, race, race_time, birth, censor`` (``censor`` is null after a death).
        """
        ids = self.ids
        rows = _with_columns(frame.lazy()).collect()
        code_ids = pl.DataFrame({"code": [t[6:] for t in self.tokens if t.startswith("CODE//")],
                                 "_code_id": [ids[t] for t in self.tokens if t.startswith("CODE//")]},
                                schema={"code": pl.String, "_code_id": pl.Int32})
        rows = rows.join(code_ids, on="code", how="left").with_columns(pl.col("_code_id").fill_null(ids[UNK]))
        code = pl.concat_list("_code_id")
        timed = rows.filter(pl.col("time").is_not_null() & ~pl.col("code").is_in([BIRTH_CODE, DEATH_CODE]))

        # Value token: a percentile bucket, a result token (value-less rows of text codes), or nothing.
        q = len(LAB_LEVELS)
        cuts = pl.DataFrame(
            {"code": list(self.quantiles), **{f"q{i}": [edges[i + 1] for edges in self.quantiles.values()] for i in range(q)}},
            schema={"code": pl.String, **{f"q{i}": pl.Float64 for i in range(q)}},
        )
        results = {t[8:]: i for t, i in ids.items() if t.startswith("RESULT//")}
        bucket = 1 + pl.sum_horizontal([(pl.col("numeric_value") > pl.col(f"q{i}")).cast(pl.Int32) for i in range(q)])
        value = (
            pl.when(pl.col("q0").is_not_null() & _numeric())
            .then(pl.concat_list(bucket.replace_strict(range(1, LAB_BUCKETS + 1),
                                                       [ids[f"LABQ//{i}"] for i in range(1, LAB_BUCKETS + 1)],
                                                       return_dtype=pl.Int32)))
            .when(pl.col("code").is_in(list(self.text_codes)) & pl.col("numeric_value").is_null())
            .then(pl.concat_list(
                pl.when(pl.col("text_value").is_null()).then(pl.lit(results["[MISSING]"], dtype=pl.Int32))
                .otherwise(pl.col("text_value").replace_strict(
                    list(results), list(results.values()), default=results["[OTHER]"], return_dtype=pl.Int32))
            ))
            .otherwise(pl.lit([], dtype=pl.List(pl.Int32)))
        )
        attributes = [
            pl.col(column).replace_strict(
                [t.split("//", 2)[2] for t in self.tokens if t.startswith(f"ATTR//{column.upper()}//")],
                [ids[t] for t in self.tokens if t.startswith(f"ATTR//{column.upper()}//")],
                default=None, return_dtype=pl.Int32)
            for column in ATTRIBUTES
        ]
        window = int(config.downsample_hours * 3600 * 1_000_000) if config.downsample_hours else None
        timed = timed.join(cuts, on="code", how="left")
        # Downsampling (0826 Sec. 2.4, applied to every numeric code): latest value per code in each window.
        numeric = _supersede(timed.filter(_numeric()), "code", window)
        others = timed.filter(~_numeric()).with_columns(pl.lit(None, dtype=pl.Datetime("us")).alias("_superseded"))
        columns = ["subject_id", "time", pl.col("_order").cast(pl.Float64).alias("order"),
                   pl.concat_list(code, value, *attributes).list.drop_nulls().alias("tokens"),
                   pl.col("_superseded").alias("superseded")]
        parts = [numeric.select(columns), others.select(columns)]

        # Death (0826 Sec. 2.2.2): MEDS_DEATH at the last instant of its date, then EOS; later events are dropped and a
        # record without death is right-censored.
        deaths = rows.filter((pl.col("code") == DEATH_CODE) & pl.col("time").is_not_null()).group_by("subject_id").agg(
            pl.col("time").min().dt.date().cast(pl.Datetime("us")).add(pl.duration(days=1, microseconds=-1)).alias("death"),
            pl.col("_code_id").first(),
        )
        parts.append(deaths.select(
            "subject_id", pl.col("death").alias("time"), pl.lit(float("inf")).alias("order"),
            pl.concat_list("_code_id", pl.lit(ids[EOS], dtype=pl.Int32)).alias("tokens"),
            pl.lit(None, dtype=pl.Datetime("us")).alias("superseded"),
        ))
        events = (
            pl.concat(parts)
            .join(deaths.select("subject_id", "death"), on="subject_id", how="left")
            .filter(pl.col("death").is_null() | (pl.col("time") <= pl.col("death")))
            .drop("death")
            .sort("subject_id", "time", "order")
        )

        # Subjects: static codes, race and its first recorded time, birth, right-censoring time.
        subjects = rows.select("subject_id").unique().sort("subject_id")
        static = rows.filter(pl.col("time").is_null() & (pl.col("code") != BIRTH_CODE)).group_by(
            "subject_id", maintain_order=True).agg(code.explode(keep_nulls=False, empty_as_null=False).alias("static"))
        births = rows.filter((pl.col("code") == BIRTH_CODE) & pl.col("time").is_not_null()).group_by("subject_id").agg(
            pl.col("time").min().alias("birth"))
        last = events.group_by("subject_id").agg(pl.col("time").max().alias("last"))
        cutoff = datetime.fromisoformat(self.cutoff)
        subjects = (
            subjects.join(static, on="subject_id", how="left").join(births, on="subject_id", how="left")
            .join(last, on="subject_id", how="left").join(deaths.select("subject_id", "death"), on="subject_id", how="left")
            .with_columns(
                pl.when(pl.col("death").is_null() & pl.col("last").is_not_null())
                .then(pl.min_horizontal(pl.col("last") + pl.duration(days=config.follow_up_days),
                                        pl.max_horizontal(pl.lit(cutoff), pl.col("last"))))
                .alias("censor"),
                pl.col("static").fill_null(pl.lit([], dtype=pl.List(pl.Int32))),
            )
        )
        race_ids = {t[6:]: i for t, i in ids.items() if t.startswith("RACE//")}
        subjects = subjects.join(_races(race).select(
            "subject_id",
            pl.col("race").replace_strict(race_ids, default=None, return_dtype=pl.Int32),
            pl.col("time").alias("race_time"),
        ), on="subject_id", how="left")
        return events, subjects.select("subject_id", "static", "race", "race_time", "birth", "censor")

    # ---- sequences -----------------------------------------------------------------------------------------------
    def sequences(self, events: pl.DataFrame, subjects: pl.DataFrame, cutoffs: pl.DataFrame | None = None) -> pl.DataFrame:
        """Token sequences. Without ``cutoffs``: one training sequence per subject (all events, superseded values
        dropped, then EOS or padding). With ``cutoffs`` (``subject_id, prediction_time``): one prompt per row, built
        only from what is recorded at or before ``prediction_time`` and ending with the elapsed-gap time tokens."""
        ids = self.ids
        training = cutoffs is None
        if training:
            cutoffs = subjects.select("subject_id", pl.lit(None, dtype=pl.Datetime("us")).alias("prediction_time"))
        keys = cutoffs.select(pl.col("subject_id").cast(pl.Int64), pl.col("prediction_time").cast(pl.Datetime("us")))
        keys = keys.with_row_index("_key")
        cut = pl.col("prediction_time")
        visible = (
            keys.join(events, on="subject_id", how="inner")
            .filter(
                pl.when(cut.is_null()).then(pl.col("superseded").is_null())
                .otherwise((pl.col("time") <= cut) & (pl.col("superseded").is_null() | (pl.col("superseded") > cut)))
            )
            .sort("_key", "time", "order")
        )
        bucket_ids = [ids[_time_token(name)] for name, _, _ in TIME_BUCKETS]
        six = ids[_time_token("6mt")]

        def gap_expr(seconds: pl.Expr) -> pl.Expr:
            bucket = pl.sum_horizontal([(seconds >= low).cast(pl.Int32) for _, low, _ in TIME_BUCKETS])
            return (
                pl.when(seconds >= SIX_MONTHS)
                .then(pl.lit(six, dtype=pl.Int32).repeat_by((seconds / SIX_MONTHS).round().cast(pl.Int64).clip(lower_bound=1)))
                .when(seconds.is_not_null() & (bucket > 0))
                .then(pl.concat_list(bucket.replace_strict(range(1, len(bucket_ids) + 1), bucket_ids, default=None,
                                                           return_dtype=pl.Int32)))
                .otherwise(pl.lit([], dtype=pl.List(pl.Int32)))
            )

        previous = pl.col("time").shift(1).over("_key")
        gap = pl.when(pl.col("time") != previous).then((pl.col("time") - previous).dt.total_microseconds() / 1e6)
        body = (
            visible.with_columns(pl.concat_list(gap_expr(gap), "tokens").alias("_tokens"))
            .group_by("_key", maintain_order=True)
            .agg(pl.col("_tokens").explode(keep_nulls=False, empty_as_null=False).alias("_body"),
                 pl.col("time").min().alias("_start"), pl.col("time").max().alias("_last"))
        )
        out = keys.join(subjects, on="subject_id", how="left").join(body, on="_key", how="left")
        start = pl.coalesce("_start", cut)
        age_days = (start - pl.col("birth")).dt.total_seconds() / 86400.0
        age_ids = {t: i for t, i in ids.items() if t.startswith("AGE//")}
        year_ids = {t: i for t, i in ids.items() if t.startswith("YEAR//")}
        years = sorted(year_ids)

        def year_id(year):
            if year is None or not years:
                return None
            token = year_token(int(year))
            return year_ids.get(token, year_ids[years[0]] if token < years[0] else year_ids[years[-1]])

        empty = pl.lit([], dtype=pl.List(pl.Int32))
        out = out.with_columns(
            pl.when(age_days.is_not_null()).then(pl.concat_list(age_days.map_elements(
                lambda d: age_ids[age_token(d)], return_dtype=pl.Int32))).otherwise(empty).alias("_age"),
            pl.when(start.is_not_null()).then(pl.concat_list(start.dt.year().map_elements(year_id, return_dtype=pl.Int32)))
            .otherwise(empty).list.drop_nulls().alias("_year"),
            pl.when(pl.col("race").is_not_null() & (pl.col("race_time").is_null() | cut.is_null() | (pl.col("race_time") <= cut)))
            .then(pl.concat_list("race")).otherwise(empty).alias("_race"),
            pl.col("_body").fill_null(empty),
        )
        if training:
            padding = (pl.col("censor") - pl.col("_last")).dt.total_seconds() / SIX_MONTHS
            tail = pl.when(pl.col("censor").is_not_null()).then(
                pl.lit(six, dtype=pl.Int32).repeat_by(padding.round().cast(pl.Int64).clip(lower_bound=0))).otherwise(empty)
        else:
            elapsed = (cut - pl.col("_last")).dt.total_microseconds() / 1e6
            tail = pl.when(pl.col("_last").is_not_null()).then(gap_expr(elapsed)).otherwise(empty)
        return out.select(
            "subject_id", "prediction_time",
            pl.concat_list(pl.lit([ids[BOS]], dtype=pl.List(pl.Int32)), "static", "_race", "_age", "_year", "_body",
                           tail.alias("_tail")).alias("tokens"),
        ).sort("subject_id", "prediction_time")

    # ---- generation ----------------------------------------------------------------------------------------------
    def decode(self, tokens: list[int], prediction_time: datetime, uniforms: np.ndarray) -> tuple[list[tuple], datetime, int]:
        """Generated ids -> (``(time, code, numeric_value, text_value)`` events, final clock, invalid-token count).

        The clock starts at the prediction time (the prompt ends with the real elapsed gap); generated events sharing
        it are placed one microsecond later. A bucket token draws its value uniformly within the bucket (0226 App. G).
        Attribute tokens carry no event; BOS or EOS ends the record.
        """
        clock, events, invalid = prediction_time, [], 0
        future = prediction_time + timedelta(microseconds=1)
        code = value = text = None

        def flush():
            nonlocal code, value, text
            if code is not None:
                events.append((max(clock, future), code, value, text))
            code = value = text = None

        for position, index in enumerate(tokens):
            token = self.tokens[index]
            kind, _, rest = token.partition("//")
            if token in (BOS, EOS):
                flush()
                break
            if kind == "TIME":
                flush()
                clock += timedelta(seconds=_token_seconds(token))
            elif kind == "CODE":
                flush()
                code = rest
            elif kind == "LABQ":
                if code in self.quantiles and value is None:
                    edges, bucket = self.quantiles[code], int(rest)
                    value = float(edges[bucket - 1] + uniforms[position] * (edges[bucket] - edges[bucket - 1]))
                else:
                    invalid += 1
            elif kind == "RESULT":
                if code is not None and code in self.text_codes and text is None:
                    text = None if rest in ("[OTHER]", "[MISSING]") else rest
                else:
                    invalid += 1
            elif kind == "ATTR":
                continue  # attributes qualify the preceding code
            else:  # PAD, UNK, demographic tokens: no event
                flush()
                invalid += token != UNK
        flush()
        return events, clock, invalid
