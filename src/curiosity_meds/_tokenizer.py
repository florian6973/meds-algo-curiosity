"""Curiosity-0826 event tokenization (arXiv 2611.15710 Sec. 2; unchanged parts from 2508.12104v3 Sec. 5.2), fitted on
the training split only.

Preprocessing stores *events*, not sequences: one row per event with its token ids, plus per-subject demographics.
:func:`sequences` turns them into token sequences at a cutoff. Without a cutoff it gives the training sequence; with
the prediction time ``T`` it gives the prompt, which depends only on what is recorded at or before ``T``:

``[BOS] [sex/static] [race] [age at start] [year at start]`` then the visible events in time order, with ETHOS time
tokens before each new timestamp, then (training only) ``[EOS]`` after a death or six-month padding tokens up to the
right-censoring time; a prompt instead ends with the time tokens of the elapsed gap to ``T``.

Everything that could depend on the future is evaluated at the cutoff: the start of the timeline (age and year
tokens), race (the first race recorded by then), and vital downsampling (a vital is dropped only once the later value
of its 6-hour window is itself visible). Padding never enters a prompt.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import polars as pl
from aces.config import PlainPredicateConfig

from .config import CATEGORIES, CuriosityConfig

PAD, BOS, EOS, UNK = "[PAD]", "[BOS]", "[EOS]", "[UNK]"
END_CAUSE_OF_DEATH = "DEATH//END_PRELIMINARY_CAUSE"
REPORTED_FLAG = "MED//PATIENT_REPORTED"
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
NEVER = datetime(9999, 12, 31)  # time of padding tokens: after every cutoff, so no prompt contains them

# Lab value buckets (Sec. 2.3.4): two 0.1% tails and 28 interior buckets whose edges are equal-width in z between the
# 0.1st and 99.9th percentiles, converted to percentile levels shared by every lab.
_Z = NormalDist().inv_cdf(0.001)
LAB_LEVELS = tuple(NormalDist().cdf(_Z + i * (-2 * _Z) / 28) for i in range(29))
LAB_BUCKETS = len(LAB_LEVELS) + 1  # 30

# Vital bins (Table 2): thresholds, left-edge exclusive, with a catch-all below the first and above the last.
# Plausibility bounds (Sec. 2.4) remove values outside (0, upper].
VITALS = {
    "SBP": ((0, *range(80, 141, 10), 160, 180, 200, 370), 370.0),
    "DBP": ((0, *range(50, 121, 10), 360), 360.0),
    "HEIGHT": ((0, 10, *range(15, 101, 5), 108), 108.0),
    "WEIGHT": ((*range(0, 51), *range(52, 101, 2), *range(103, 161, 3), *range(165, 261, 5), *range(270, 301, 10),
                *range(320, 481, 20)), 1400.0),
    "BMI": ((0, 18, *range(19, 41), 204), 204.0),
}
# category -> (vital, factor to the paper's unit)
VITAL_CATEGORIES = {
    "vital_sbp_mmhg": ("SBP", 1.0), "vital_dbp_mmhg": ("DBP", 1.0), "vital_height_in": ("HEIGHT", 1.0),
    "vital_height_cm": ("HEIGHT", 1 / 2.54), "vital_weight_lb": ("WEIGHT", 1.0),
    "vital_weight_kg": ("WEIGHT", 2.20462262185), "vital_bmi": ("BMI", 1.0),
}
VITAL_WINDOW_US = 6 * 3600 * 1_000_000  # latest value per vital type in each 6-hour window
MEDICATION_ATTRIBUTES = ("route", "dose", "frequency")  # optional MEDS columns -> Route_Form, Dose, Frequency tokens
_ATC = re.compile(r"^[A-Z]\d\d[A-Z]{2}\d{2}$")
_RANK = {"birth": 0, "gestational_age": 1, "birth_weight": 2}  # birth tokens lead their instant (Sec. 2.1)
_EVENT_COLUMNS = ["subject_id", "time", "code", "numeric_value", "text_value", *MEDICATION_ATTRIBUTES]


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
    """Age at the start of the timeline (Sec. 2.1): days in week 1, weeks to 4 weeks, 4-week steps to 1 year,
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


def gestational_age_token(weeks: float) -> str:
    """Completed weeks with edge buckets (Sec. 2.1)."""
    if weeks < 22:
        return "BIRTH//GA//lt22w"
    if weeks > 44:
        return "BIRTH//GA//gt44w"
    return f"BIRTH//GA//{int(weeks)}w"


GESTATIONAL_TOKENS = ("BIRTH//GA//lt22w", *(f"BIRTH//GA//{w}w" for w in range(22, 45)), "BIRTH//GA//gt44w")


def _split(code: str, atc: bool) -> tuple[str, str, str]:
    """``<prefix>//E1165`` -> ``<prefix>//E11``, ``65``, ``""`` (ICD 3/2/2; ATC 3/2/rest)."""
    prefix, _, value = code.rpartition("//")
    return f"{prefix}//{value[:3]}", value[3:5], value[5:] if atc else value[5:7]


def _sub_token(kind: str, position: int, part: str) -> str:
    return f"{kind}//{'4-5' if position == 1 else '6-'}//{part}"


def category_predicates(predicates: Mapping[str, Any]) -> tuple[tuple[str, dict], ...]:
    """``(category, ACES plain-predicate spec)`` in matching order from a dataset category file. A category the dataset
    does not record is omitted (core rejects empty predicates); an unknown ``curiosity_*`` name is an error, so a typo
    cannot silently drop a category."""
    known = {f"curiosity_{c}" for c in CATEGORIES}
    unknown = sorted(name for name in predicates if name.startswith("curiosity_") and name not in known)
    if unknown:
        raise ValueError(f"Unknown Curiosity categories {unknown}; expected names among {sorted(known)}")
    if not known & predicates.keys():
        raise ValueError("The dataset category file defines no curiosity_* category")
    selected = []
    for category in CATEGORIES:
        spec = predicates.get(f"curiosity_{category}")
        if spec is None:
            continue
        spec = dict(spec)
        if "code" not in spec or "expr" in spec or spec.get("other_cols"):
            raise ValueError(f"curiosity_{category} must be an ACES plain predicate on code/numeric_value")
        PlainPredicateConfig(**spec).MEDS_eval_expr()  # rejects malformed code fields early
        selected.append((category, spec))
    return tuple(selected)


def _with_columns(frame: pl.LazyFrame) -> pl.LazyFrame:
    schema = frame.collect_schema()
    missing = [pl.lit(None, dtype=pl.String).alias(c) for c in _EVENT_COLUMNS if c not in schema]
    return frame.with_columns(missing).select(
        pl.col("subject_id").cast(pl.Int64), pl.col("time").cast(pl.Datetime("us")), pl.col("code").cast(pl.String),
        pl.col("numeric_value").cast(pl.Float64), pl.col("text_value").cast(pl.String),
        *(pl.col(c).cast(pl.String) for c in MEDICATION_ATTRIBUTES),
    )


def _categorized(frame: pl.LazyFrame, categories: tuple[tuple[str, dict], ...]) -> pl.LazyFrame:
    category = pl.lit(None, dtype=pl.String)
    for name, spec in reversed(categories):
        category = pl.when(PlainPredicateConfig(**spec).MEDS_eval_expr()).then(pl.lit(name)).otherwise(category)
    return _with_columns(frame).with_row_index("_order").with_columns(category.alias("category")).filter(
        pl.col("category").is_not_null()
    )


def _vital_values(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Vital rows in the paper's units: numeric_value, else a number in text_value; a ``S/D`` text blood pressure
    (MIMIC-IV OMR before MIMIC-IV-MEDS 0.2) is not split here and is dropped. Implausible values are removed."""
    factor = pl.col("category").replace_strict({c: f for c, (_, f) in VITAL_CATEGORIES.items()}, default=None,
                                               return_dtype=pl.Float64)
    vital = pl.col("category").replace_strict({c: v for c, (v, _) in VITAL_CATEGORIES.items()}, default=None,
                                              return_dtype=pl.String)
    upper = vital.replace_strict({v: u for v, (_, u) in VITALS.items()}, default=None, return_dtype=pl.Float64)
    value = pl.coalesce(pl.col("numeric_value"), pl.col("text_value").str.strip_chars().cast(pl.Float64, strict=False))
    return (
        frame.filter(pl.col("category").is_in(list(VITAL_CATEGORIES)))
        .with_columns(vital.alias("vital"), (value * factor).alias("value"))
        .filter((pl.col("value") > 0) & (pl.col("value") <= upper))
    )


def _races(kept: pl.DataFrame, column: pl.DataFrame | None) -> pl.DataFrame:
    """First recorded race per subject and its time (null for a static record): from ``race`` category events
    (``RACE//<value>`` codes, MIMIC-IV-MEDS >= 0.2) and from a race column (MIMIC-IV-MEDS 0.1.0 admission rows)."""
    schema = {"subject_id": pl.Int64, "race": pl.String, "time": pl.Datetime("us")}
    rows = [kept.filter(pl.col("category") == "race").select(
        "subject_id", pl.col("code").str.strip_prefix("RACE//").alias("race"), "time")]
    if column is not None and column.height:
        rows.append(column.select(pl.col("subject_id").cast(pl.Int64), pl.col("race").cast(pl.String),
                                  pl.col("time").cast(pl.Datetime("us"))))
    frame = pl.concat([r.cast(schema) for r in rows]).filter(pl.col("race").is_not_null())
    return frame.sort("subject_id", "time", "race", nulls_last=False).group_by("subject_id", maintain_order=True).first()


@dataclass(frozen=True)
class _Vocabulary:
    """Fitted tokenizer state (training split only)."""

    categories: tuple[tuple[str, dict], ...]
    tokens: tuple[str, ...]
    quantiles: dict[str, tuple[float, ...]]  # code -> (min, q(levels)..., max): lab and birth-weight buckets
    text_labs: tuple[str, ...]  # lab codes whose text results get result tokens
    split_codes: dict[str, str]  # training code -> "ICD" | "ATC" split into sub-tokens
    vital_codes: dict[str, tuple[str, float]]  # vital -> (MEDS code generated for it, factor to the paper unit)
    gestational_code: str | None
    cutoff: str  # latest training time (ISO): right-censoring bound

    # ---- fitting -----------------------------------------------------------------------------------------------
    @classmethod
    def fit(cls, train: pl.LazyFrame, config: CuriosityConfig, categories: tuple[tuple[str, dict], ...],
            race: pl.LazyFrame | None) -> _Vocabulary:
        kept = _categorized(train, categories).collect()
        timed = kept.filter(pl.col("time").is_not_null())

        def counts(frame: pl.DataFrame) -> list[tuple[str, int]]:
            ranked = frame.group_by("code").agg(pl.len().alias("n")).sort(["n", "code"], descending=[True, False])
            return list(ranked.iter_rows())

        def share_kept(frame: pl.DataFrame, share: float, minimum: int = 1) -> list[str]:
            total = frame.height
            return sorted(code for code, n in counts(frame) if n >= minimum and n >= share * total)

        by = {name: kept.filter(pl.col("category") == name) for name, _ in categories}
        empty = kept.clear()
        get = lambda name: by.get(name, empty)  # noqa: E731
        whole, split_codes = [], {}
        for name in ("sex", "birth", "death", "cause_of_death", "encounter"):
            whole += sorted(get(name)["code"].unique().to_list())
        for code in sorted(get("diagnosis")["code"].unique().to_list()):
            split_codes[code] = "ICD"
        for name in ("medication", "reported_medication"):
            for code in sorted(get(name)["code"].unique().to_list()):
                if _ATC.match(code.rpartition("//")[2]):
                    split_codes[code] = "ATC"
                else:
                    whole.append(code)
        whole += share_kept(get("procedure"), config.procedure_min_share)

        # Labs (Sec. 2.3): numeric labs bucketed per code; non-numeric labs with result tokens; value-less rows
        # (collection markers emitted by the ETL, e.g. MIMIC-IV-MEDS LAB//SPECIMEN_COLLECTED) as plain components.
        labs = get("lab")
        numeric = labs.filter(pl.col("numeric_value").is_not_null() & pl.col("numeric_value").is_finite())
        text = labs.filter(pl.col("numeric_value").is_null() & pl.col("text_value").is_not_null())
        bare = labs.filter(pl.col("numeric_value").is_null() & pl.col("text_value").is_null())
        numeric_labs = share_kept(numeric, config.lab_numeric_min_share, config.lab_min_bucket_observations)
        text_labs = share_kept(text, config.lab_text_min_share)
        bare_labs = share_kept(bare, config.lab_numeric_min_share)
        text_values = share_kept(text.filter(pl.col("code").is_in(text_labs)).select(pl.col("text_value").alias("code")),
                                 config.lab_text_value_min_share)
        birth_weight = get("birth_weight").filter(pl.col("numeric_value").is_not_null())
        weight_codes = share_kept(birth_weight, 0.0, config.lab_min_bucket_observations)
        whole += sorted(set(numeric_labs) | set(text_labs) | set(bare_labs) | set(weight_codes))
        quantiles = {}
        bucketed = pl.concat([numeric.filter(pl.col("code").is_in(numeric_labs)),
                              birth_weight.filter(pl.col("code").is_in(weight_codes))])
        if bucketed.height:
            stats = bucketed.group_by("code").agg(
                pl.col("numeric_value").min().alias("min"),
                *(pl.col("numeric_value").quantile(level, "linear").alias(f"q{i}") for i, level in enumerate(LAB_LEVELS)),
                pl.col("numeric_value").max().alias("max"),
            )
            columns = ["min", *(f"q{i}" for i in range(len(LAB_LEVELS))), "max"]
            quantiles = {row["code"]: tuple(float(row[c]) for c in columns) for row in stats.iter_rows(named=True)}

        # Medication attributes (Sec. 2.6): values below the floor are null (no token).
        meds = pl.concat([get("medication"), get("reported_medication")])
        attributes = []
        for column in MEDICATION_ATTRIBUTES:
            values = meds.filter(pl.col(column).is_not_null()).select(pl.col(column).alias("code"))
            attributes += [f"MED//{column.upper()}//{v}" for v in share_kept(values, config.medication_attribute_min_share)]

        # Vitals: the most frequent training code of each vital is the one generated futures use.
        vital_codes = {}
        vitals = _vital_values(kept.lazy()).collect()
        for vital in VITALS:
            ranked = counts(vitals.filter(pl.col("vital") == vital))
            if ranked:
                category = vitals.filter(pl.col("code") == ranked[0][0])["category"][0]
                vital_codes[vital] = (ranked[0][0], VITAL_CATEGORIES[category][1])
        gestational = counts(get("gestational_age"))

        heads, subs = set(), set()
        for code, kind in split_codes.items():
            head, *parts = _split(code, kind == "ATC")
            heads.add(head)
            subs.update(_sub_token(kind, position, part) for position, part in enumerate(parts, 1) if part)
        races = sorted(_races(kept, race.collect() if race is not None else None)["race"].unique().to_list())
        starts = timed.group_by("subject_id").agg(pl.col("time").min())["time"].dt.year()
        years = sorted({year_token(y) for y in range(int(starts.min()), int(starts.max()) + 1)}) if len(starts) else []
        tokens = (
            PAD, BOS, EOS, UNK, END_CAUSE_OF_DEATH, REPORTED_FLAG,
            *(_time_token(name) for name, _, _ in TIME_BUCKETS), _time_token("6mt"),
            *AGE_TOKENS, *years, *(f"RACE//{value}" for value in races),
            *(f"LABQ//{i}" for i in range(1, LAB_BUCKETS + 1)),
            *(f"RESULT//{v}" for v in text_values), "RESULT//[OTHER]", "RESULT//[MISSING]",
            *(f"VITAL//{vital}//{i}" for vital, (edges, _) in VITALS.items() if vital in vital_codes
              for i in range(len(edges) + 1)),
            *(GESTATIONAL_TOKENS if gestational else ()),
            *attributes,
            *(f"CODE//{code}" for code in sorted(set(whole))),
            *(f"CODE//{head}" for head in sorted(heads - {f"{c}" for c in whole})),
            *sorted(subs),
        )
        cutoff = timed["time"].max()
        return cls(categories, tuple(dict.fromkeys(tokens)), quantiles, tuple(text_labs), split_codes, vital_codes,
                   gestational[0][0] if gestational else None, (cutoff or datetime(1970, 1, 1)).isoformat())

    # ---- persistence ---------------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "categories": [[name, spec] for name, spec in self.categories],
            "tokens": list(self.tokens),
            "quantiles": {code: list(edges) for code, edges in self.quantiles.items()},
            "text_labs": list(self.text_labs),
            "split_codes": self.split_codes,
            "vital_codes": {k: list(v) for k, v in self.vital_codes.items()},
            "gestational_code": self.gestational_code,
            "cutoff": self.cutoff,
        }, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> _Vocabulary:
        v = json.loads(path.read_text())
        return cls(tuple((name, spec) for name, spec in v["categories"]), tuple(v["tokens"]),
                   {code: tuple(edges) for code, edges in v["quantiles"].items()}, tuple(v["text_labs"]),
                   v["split_codes"], {k: tuple(x) for k, x in v["vital_codes"].items()}, v["gestational_code"],
                   v["cutoff"])

    # ---- lookups -------------------------------------------------------------------------------------------------
    @property
    def ids(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def codes(self) -> set[str]:
        """Every MEDS code the tokenizer fitted (a token, a split, bucket statistics or a generated vital code)."""
        heads = {_split(code, kind == "ATC")[0] for code, kind in self.split_codes.items()}
        whole = {t[6:] for t in self.tokens if t.startswith("CODE//")} - (heads - set(self.split_codes))
        return whole | set(self.split_codes) | set(self.quantiles) | {c for c, _ in self.vital_codes.values()} | (
            {self.gestational_code} if self.gestational_code else set())

    def _code_tokens(self) -> dict[str, list[int]]:
        ids = self.ids
        mapping = {token[6:]: [index] for token, index in ids.items() if token.startswith("CODE//")}
        for code, kind in self.split_codes.items():
            head, *parts = _split(code, kind == "ATC")
            mapping[code] = [ids[f"CODE//{head}"]] + [
                ids[_sub_token(kind, position, part)] for position, part in enumerate(parts, 1) if part
            ]
        return mapping

    # ---- events --------------------------------------------------------------------------------------------------
    def events(self, frame: pl.DataFrame, race: pl.DataFrame | None, config: CuriosityConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
        """``(events, subjects)`` for one shard.

        ``events``: ``subject_id, time, rank, order, tokens, superseded`` (one row per event; ``superseded`` is the
        time of the later vital of the same 6-hour window, which hides this one once visible).
        ``subjects``: ``subject_id, static, race, race_time, birth, censor`` (``censor`` is null after a death).
        """
        ids = self.ids
        code_tokens = self._code_tokens()
        unk = ids[UNK]
        kept = _categorized(frame.lazy(), self.categories).collect()
        code_map = pl.DataFrame({"code": list(code_tokens), "_code": list(code_tokens.values())},
                                schema={"code": pl.String, "_code": pl.List(pl.Int32)})
        kept = kept.join(code_map, on="code", how="left")
        code = pl.col("_code").fill_null(pl.lit([unk], dtype=pl.List(pl.Int32)))
        timed = kept.filter(pl.col("time").is_not_null())
        parts = []

        def add(rows: pl.DataFrame, tokens: pl.Expr) -> None:
            parts.append(rows.select(
                "subject_id", "time", pl.col("category").replace_strict(_RANK, default=3, return_dtype=pl.Int32).alias("rank"),
                pl.col("_order").cast(pl.Float64).alias("order"), tokens.alias("tokens"),
                pl.lit(None, dtype=pl.Datetime("us")).alias("superseded"),
            ))

        add(timed.filter(pl.col("category").is_in(["birth", "encounter", "diagnosis", "procedure"])), code)

        # Labs and birth weight: component, then a bucket, a result token, or nothing (collection marker).
        q = len(LAB_LEVELS)
        cuts = pl.DataFrame(
            {"code": list(self.quantiles), **{f"q{i}": [edges[i + 1] for edges in self.quantiles.values()] for i in range(q)}},
            schema={"code": pl.String, **{f"q{i}": pl.Float64 for i in range(q)}},
        )
        results = {t[8:]: i for t, i in ids.items() if t.startswith("RESULT//")}
        labs = timed.filter(pl.col("category").is_in(["lab", "birth_weight"])).join(cuts, on="code", how="left")
        bucket = 1 + pl.sum_horizontal([(pl.col("numeric_value") > pl.col(f"q{i}")).cast(pl.Int32) for i in range(q)])
        value = (
            pl.when(pl.col("q0").is_not_null() & pl.col("numeric_value").is_not_null())
            .then(pl.concat_list(bucket.replace_strict(range(1, LAB_BUCKETS + 1),
                                                       [ids[f"LABQ//{i}"] for i in range(1, LAB_BUCKETS + 1)],
                                                       return_dtype=pl.Int32)))
            .when(pl.col("code").is_in(list(self.text_labs)) & pl.col("numeric_value").is_null())
            .then(pl.concat_list(
                pl.when(pl.col("text_value").is_null()).then(pl.lit(results["[MISSING]"], dtype=pl.Int32))
                .otherwise(pl.col("text_value").replace_strict(
                    list(results), list(results.values()), default=results["[OTHER]"], return_dtype=pl.Int32))
            ))
            .otherwise(pl.lit([], dtype=pl.List(pl.Int32)))
        )
        add(labs, pl.concat_list(code, value))

        # Gestational age: one token at its time.
        if "BIRTH//GA//22w" in ids:
            weeks = timed.filter((pl.col("category") == "gestational_age") & pl.col("numeric_value").is_not_null())
            add(weeks, pl.concat_list(pl.col("numeric_value").map_elements(
                lambda w: ids[gestational_age_token(w)], return_dtype=pl.Int32)))

        # Medications: code (ATC 3/2/rest), attribute tokens; a flag after each block of reported medications.
        attributes = [
            pl.col(column).map_elements(lambda v, c=column: ids.get(f"MED//{c.upper()}//{v}"), return_dtype=pl.Int32)
            for column in MEDICATION_ATTRIBUTES
        ]
        meds = timed.filter(pl.col("category").is_in(["medication", "reported_medication"]))
        add(meds, pl.concat_list(code, *attributes).list.drop_nulls())
        reported = meds.filter(pl.col("category") == "reported_medication")
        if reported.height:
            flags = reported.group_by("subject_id", "time").agg(pl.col("_order").max())
            parts.append(flags.select(
                "subject_id", "time", pl.lit(3, dtype=pl.Int32).alias("rank"),
                (pl.col("_order").cast(pl.Float64) + 0.5).alias("order"),
                pl.lit([ids[REPORTED_FLAG]], dtype=pl.List(pl.Int32)).alias("tokens"),
                pl.lit(None, dtype=pl.Datetime("us")).alias("superseded"),
            ))

        # Vitals: one binned token; latest value per type in each 6-hour window (Sec. 2.4).
        vitals = _vital_values(timed.lazy()).filter(pl.col("vital").is_in(list(self.vital_codes))).collect()
        if vitals.height:
            vitals = vitals.with_columns(
                pl.struct("vital", "value").map_elements(
                    lambda r: ids[f"VITAL//{r['vital']}//{sum(t < r['value'] for t in VITALS[r['vital']][0])}"],
                    return_dtype=pl.Int32).alias("_vital_id"),
                (pl.col("time").dt.epoch("us") // VITAL_WINDOW_US).alias("_window"),
            ).sort("subject_id", "vital", "time", "_order").with_columns(
                pl.when(pl.col("_window").shift(-1).over("subject_id", "vital") == pl.col("_window"))
                .then(pl.col("time").shift(-1).over("subject_id", "vital")).alias("_superseded")
            )
            parts.append(vitals.select(
                "subject_id", "time", pl.lit(3, dtype=pl.Int32).alias("rank"),
                pl.col("_order").cast(pl.Float64).alias("order"), pl.concat_list("_vital_id").alias("tokens"),
                pl.col("_superseded").alias("superseded"),
            ))

        # Death (Sec. 2.2.2): at the last instant of the recorded date, followed by the preliminary causes of death,
        # their closing token and EOS. Later events are dropped; a record without death is right-censored.
        deaths = timed.filter(pl.col("category") == "death").group_by("subject_id").agg(
            pl.col("time").min().dt.date().cast(pl.Datetime("us")).add(pl.duration(days=1, microseconds=-1)).alias("death"),
            pl.col("_code").sort_by("time").first().fill_null(pl.lit([unk], dtype=pl.List(pl.Int32))).alias("_death_code"),
        )
        causes = kept.filter(pl.col("category") == "cause_of_death").group_by("subject_id", maintain_order=True).agg(
            code.explode(keep_nulls=False, empty_as_null=False).alias("_causes"))
        death_rows = deaths.join(causes, on="subject_id", how="left").with_columns(
            pl.concat_list(
                "_death_code",
                pl.when(pl.col("_causes").is_not_null())
                .then(pl.concat_list("_causes", pl.lit([ids[END_CAUSE_OF_DEATH]], dtype=pl.List(pl.Int32))))
                .otherwise(pl.lit([], dtype=pl.List(pl.Int32))),
                pl.lit([ids[EOS]], dtype=pl.List(pl.Int32)),
            ).alias("tokens")
        )
        parts.append(death_rows.select(
            "subject_id", pl.col("death").alias("time"), pl.lit(9, dtype=pl.Int32).alias("rank"),
            pl.lit(0.0).alias("order"), "tokens", pl.lit(None, dtype=pl.Datetime("us")).alias("superseded"),
        ))
        events = (
            pl.concat(parts)
            .join(deaths.select("subject_id", "death"), on="subject_id", how="left")
            .filter(pl.col("death").is_null() | (pl.col("time") <= pl.col("death")))
            .drop("death")
            .sort("subject_id", "time", "rank", "order")
        )

        # Subjects: static tokens, race and its first recorded time, birth, right-censoring time.
        subjects = frame.select(pl.col("subject_id").cast(pl.Int64).unique().sort())
        static = kept.filter(pl.col("time").is_null() & (pl.col("category") != "race")).group_by("subject_id", maintain_order=True).agg(
            code.explode(keep_nulls=False, empty_as_null=False).alias("static"))
        births = frame.lazy().filter(pl.col("code") == "MEDS_BIRTH").group_by("subject_id").agg(
            pl.col("time").min().cast(pl.Datetime("us")).alias("birth")).with_columns(pl.col("subject_id").cast(pl.Int64)).collect()
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
        subjects = subjects.join(_races(kept, race).select(
            "subject_id",
            pl.col("race").replace_strict(race_ids, default=None, return_dtype=pl.Int32),
            pl.col("time").alias("race_time"),
        ), on="subject_id", how="left")
        return events, subjects.select("subject_id", "static", "race", "race_time", "birth", "censor")

    # ---- sequences -----------------------------------------------------------------------------------------------
    def sequences(self, events: pl.DataFrame, subjects: pl.DataFrame, cutoffs: pl.DataFrame | None = None) -> pl.DataFrame:
        """Token sequences. Without ``cutoffs``: one training sequence per subject (all events, superseded vitals
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
            .sort("_key", "time", "rank", "order")
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
            .agg(pl.col("_tokens").explode(keep_nulls=False, empty_as_null=False).alias("_body"), pl.col("time").min().alias("_start"),
                 pl.col("time").max().alias("_last"))
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
        Attribute, flag and closing tokens carry no event; BOS or EOS ends the record.
        """
        splits = {}
        for code, kind in self.split_codes.items():
            head, *_ = _split(code, kind == "ATC")
            splits[head] = kind
        clock, events, invalid = prediction_time, [], 0
        future = prediction_time + timedelta(microseconds=1)
        code = value = text = None
        split_kind, split_length = None, None

        def flush():
            nonlocal code, value, text, split_kind, split_length
            if code is not None:
                events.append((max(clock, future), code, value, text))
            code = value = text = split_kind = split_length = None

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
                split_kind = splits.get(code)
                split_length = 3 if split_kind else None
            elif kind in ("ICD", "ATC"):
                where, _, part = rest.partition("//")
                expected = 3 if where == "4-5" else 5
                if split_kind == kind and split_length == expected:
                    code, split_length = code + part, split_length + len(part)
                else:
                    invalid += 1
            elif kind == "LABQ":
                if code in self.quantiles and value is None:
                    edges, bucket = self.quantiles[code], int(rest)
                    value = float(edges[bucket - 1] + uniforms[position] * (edges[bucket] - edges[bucket - 1]))
                else:
                    invalid += 1
            elif kind == "RESULT":
                if code is not None and code in self.text_labs and text is None:
                    text = None if rest in ("[OTHER]", "[MISSING]") else rest
                else:
                    invalid += 1
            elif kind == "VITAL":
                flush()
                vital, _, bin_ = rest.partition("//")
                edges, upper = VITALS[vital]
                b = int(bin_)
                low = edges[b - 1] if b > 0 else 0.0
                high = edges[b] if b < len(edges) else upper
                vital_code, factor = self.vital_codes[vital]
                events.append((max(clock, future), vital_code, float(low + uniforms[position] * (high - low)) / factor, None))
            elif kind == "BIRTH" and token in GESTATIONAL_TOKENS:
                flush()
                weeks = {"lt22w": 21.5, "gt44w": 44.5}.get(rest[4:], None)
                weeks = weeks if weeks is not None else int(rest[4:-1]) + 0.5
                events.append((max(clock, future), self.gestational_code, weeks, None))
            elif kind == "MED" or token == END_CAUSE_OF_DEATH:
                continue  # attributes and flags qualify the preceding medication; the closing token has no event
            else:  # PAD, UNK, demographic tokens
                flush()
                invalid += token != UNK
        flush()
        return events, clock, invalid
