"""Curiosity-specific behavior: 0826 tokenization, internal temporal cutoffs, configuration effects, sampling and
Eq. 2 aggregation."""

import dataclasses
import json
import random
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch
import yaml
from aces.config import TaskExtractorConfig
from meds_algorithm.conformance.temporal import perturb_future
from meds_algorithm.requests import PreprocessDataRequest

from curiosity_meds import Curiosity, CuriosityConfig
from curiosity_meds import _model
from curiosity_meds._tokenizer import (
    LAB_BUCKETS, LAB_LEVELS, SIX_MONTHS, VITALS, _token_seconds, _Vocabulary, age_token, category_predicates,
    gap_tokens,
)
from curiosity_meds.config import CATEGORIES
from curiosity_meds.curiosity_meds import DATASETS_DIR, _aggregate, _check_generable

from .e2e import PREDICATES_FILE, TASK, TASK_FILE, TEST_CONFIG, build_fixture

T0 = datetime(2150, 1, 1, 8)
MIMIC = yaml.safe_load((DATASETS_DIR / "MIMIC-IV.yaml").read_text())["predicates"]
# Every category, for tokenizer tests: MIMIC-IV's predicates plus the categories MIMIC-IV does not record.
ALL = {**MIMIC, "curiosity_birth": {"code": "BIRTH//DOCUMENTED"}, "curiosity_gestational_age": {"code": "BIRTH//GA"},
       "curiosity_birth_weight": {"code": "BIRTH//WEIGHT//g"}, "curiosity_cause_of_death": {"code": {"regex": "^COD//"}},
       "curiosity_vital_bmi": {"code": "BMI"}, "curiosity_reported_medication": {"code": {"regex": "^HOME_MED//"}}}
CONFIG = CuriosityConfig(lab_min_bucket_observations=5, follow_up_days=365.0)
SCHEMA = {"subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.String, "numeric_value": pl.Float32,
          "text_value": pl.String, "route": pl.String, "race": pl.String}


def _events(rows):
    rows = [(*row, *([None] * (7 - len(row)))) for row in rows]
    return pl.DataFrame(rows, orient="row", schema=SCHEMA)


def _hours(h):
    return T0 + timedelta(hours=h)


TRAIN = _events([
    (1, None, "GENDER//F"),
    (1, T0 - timedelta(days=365.25 * 62), "MEDS_BIRTH"),
    (1, _hours(0), "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", None, None, None, "WHITE"),
    *((1, _hours(k), "LAB//SPECIMEN_COLLECTED//50912//STAT") for k in range(10)),
    *((1, _hours(k + 1), "LAB//RESULT//50912//mg/dL", float(k)) for k in range(10)),
    *((1, _hours(k + 1), "LAB//RESULT//51463//UNK", None, value) for k, value in enumerate(["Neg", "Pos", "Neg"])),
    (1, _hours(2), "LAB//RESULT//51463//UNK"),  # text lab without a result -> missing-value token
    (1, _hours(3), "LAB//220179//mmHg", 120.0),
    (1, _hours(4), "LAB//226730//cm", 170.0),
    (1, _hours(5), "LAB//226512//kg", 70.0),
    (1, _hours(5), "BMI", 24.2),
    (1, _hours(6), "MEDICATION//START//Heparin", None, None, "IV"),
    (1, _hours(6), "MEDICATION//START//N02BE01"),  # an ATC-coded medication
    (1, _hours(6), "HOME_MED//B01AC06"),
    (1, _hours(70), "DIAGNOSIS//ICD//10//E1165"),
    (1, _hours(70), "PROCEDURE//ICD//10//0BH17EZ"),
    (1, _hours(80), "LAB//220045//bpm", 80.0),  # chartevents other than the vitals: the 'other' category
    (2, None, "GENDER//M"),
    (2, T0 - timedelta(days=2), "MEDS_BIRTH"),
    (2, T0 - timedelta(days=2), "BIRTH//DOCUMENTED"),
    (2, T0 - timedelta(days=2), "BIRTH//WEIGHT//g", 3300.0),
    (2, T0 - timedelta(days=2), "BIRTH//GA", 39.4),
    *((2, T0 - timedelta(days=2, minutes=-k), "BIRTH//WEIGHT//g", 3000.0 + k) for k in range(1, 6)),
    (2, _hours(1), "MEDS_DEATH"),
    (2, _hours(1), "COD//I469"),
    (2, _hours(30), "HOSPITAL_DISCHARGE//DIED"),  # after the death date: dropped
])


@pytest.fixture(scope="module")
def vocabulary():
    return _Vocabulary.fit(TRAIN.lazy(), CONFIG, category_predicates(ALL), TRAIN.lazy().select("subject_id", "race",
                                                                                                "time"))


def _tokens(vocabulary, sequence):
    return [vocabulary.tokens[i] for i in sequence]


def _sequences(vocabulary, frame, keys=None):
    race = frame.filter(pl.col("race").is_not_null()).sort("subject_id", "time").group_by("subject_id").agg(
        pl.col("race").first(), pl.col("time").first())
    events, subjects = vocabulary.events(frame, race, CONFIG)
    return vocabulary.sequences(events, subjects, keys)


# ---- configuration and category files ------------------------------------------------------------------------------
def test_category_file_is_chosen_by_dataset_name_or_explicit_file(tmp_path):
    def request(name):
        (tmp_path / name / "metadata").mkdir(parents=True)
        (tmp_path / name / "metadata" / "dataset.json").write_text(json.dumps({"dataset_name": name}))
        return PreprocessDataRequest.from_arguments({"external_meds_dir": str(tmp_path / name),
                                                     "output_data_dir": str(tmp_path / "out")})

    path, source = Curiosity(CuriosityConfig())._category_file(request("MIMIC-IV"))
    assert path == DATASETS_DIR / "MIMIC-IV.yaml" and "dataset_name" in source
    # With the 'other' category a dataset needs no file; a paper-only vocabulary does.
    assert Curiosity(CuriosityConfig())._category_file(request("eICU"))[0] is None
    with pytest.raises(ValueError, match="No Curiosity category file for dataset_name 'eICU2'"):
        Curiosity(CuriosityConfig(other_category=False))._category_file(request("eICU2"))
    explicit = Curiosity(CuriosityConfig(additional_predicates_file=DATASETS_DIR / "MIMIC-IV.yaml"))
    resolved = tmp_path / "resolved_feature_predicates.yaml"  # what the runner composes into staging
    composed = dataclasses.replace(request("other"), predicates_file=resolved)
    assert explicit._category_file(composed) == (resolved, "additional_predicates_file")


def test_category_file_selects_known_categories_in_paper_order():
    assert [c for c, _ in category_predicates(MIMIC)] == [c for c in CATEGORIES if f"curiosity_{c}" in MIMIC] + ["other"]
    assert [c for c, _ in category_predicates(ALL, other=False)] == list(CATEGORIES)
    # No file: death is the MEDS standard code and every other code is 'other'.
    assert category_predicates({}) == (("death", {"code": "MEDS_DEATH"}), ("other", None))
    with pytest.raises(ValueError, match="curiosity_procedures"):  # a typo must not silently drop a category
        category_predicates({**MIMIC, "curiosity_procedures": {"code": "X"}})
    with pytest.raises(ValueError, match="paper-only vocabulary needs"):
        category_predicates({"icu_admission": {"code": "X"}}, other=False)
    with pytest.raises(ValueError, match="plain predicate"):
        category_predicates({**MIMIC, "curiosity_lab": {"expr": "or(a, b)"}})


# ---- 0826 tokens ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("seconds", "expected"), [
    (59, []), (60, ["TIME//1m-5m"]), (300, ["TIME//5m-15m"]), (86400, ["TIME//1d-3d"]),
    (SIX_MONTHS - 1, ["TIME//3mt-6mt"]), (SIX_MONTHS, ["TIME//6mt"]), (400 * 86400, ["TIME//6mt"] * 2),
])
def test_gap_tokens_follow_ethos_buckets(seconds, expected):
    assert gap_tokens(seconds) == expected


def test_time_tokens_decode_to_geometric_bucket_means():
    assert _token_seconds("TIME//1m-5m") == pytest.approx(np.sqrt(60 * 300))
    assert _token_seconds("TIME//6mt") == SIX_MONTHS


@pytest.mark.parametrize(("days", "token"), [
    (0, "AGE//0d"), (6.9, "AGE//6d"), (7, "AGE//1w"), (27, "AGE//3w"), (28, "AGE//4w"), (364, "AGE//52w"),
    (366, "AGE//12mo"), (2.6 * 365.25, "AGE//30mo"), (4.99 * 365.25, "AGE//54mo"), (5 * 365.25, "AGE//5y"),
    (54.5 * 365.25, "AGE//54y"), (130 * 365.25, "AGE//120y"),
])
def test_age_buckets_are_finer_for_children(days, token):
    assert age_token(days) == token


def test_lab_buckets_are_thirty_shared_percentile_levels(vocabulary):
    assert LAB_BUCKETS == 30 and LAB_LEVELS[0] == pytest.approx(0.001) and LAB_LEVELS[-1] == pytest.approx(0.999)
    from statistics import NormalDist

    widths = np.diff([NormalDist().inv_cdf(level) for level in LAB_LEVELS])
    assert np.allclose(widths, widths[0]) and len(widths) == 28
    edges = vocabulary.quantiles["LAB//RESULT//50912//mg/dL"]
    assert len(edges) == 31 and edges[0] == 0 and edges[-1] == 9


def test_tokenizer_emits_every_0826_representation(vocabulary):
    sequences = _sequences(vocabulary, TRAIN)
    first, second = (_tokens(vocabulary, s) for s in sequences["tokens"])
    assert first[:5] == ["[BOS]", "CODE//GENDER//F", "RACE//WHITE", "AGE//62y", "YEAR//2150-2154"]
    # Lab collection markers come from the ETL's own rows; results carry a shared bucket or a result token.
    assert first[5:9] == ["CODE//HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM",
                          "CODE//LAB//SPECIMEN_COLLECTED//50912//STAT", "TIME//1h-2h",
                          "CODE//LAB//SPECIMEN_COLLECTED//50912//STAT"]
    assert first[9:14] == ["CODE//LAB//RESULT//50912//mg/dL", "LABQ//1", "CODE//LAB//RESULT//51463//UNK",
                           "RESULT//Neg", "TIME//1h-2h"]
    assert ["CODE//LAB//RESULT//51463//UNK", "RESULT//[MISSING]"] == first[first.index("RESULT//[MISSING]") - 1:][:2]
    # Vitals: one binned token each in the paper's units (170 cm = 66.9 in; 70 kg = 154.3 lb).
    assert {"VITAL//SBP//5", "VITAL//HEIGHT//13", "VITAL//WEIGHT//94", "VITAL//BMI//8"} <= set(first)
    # Medications: route token; ATC split 3/2/rest; patient-reported block closed by its flag.
    i = first.index("CODE//MEDICATION//START//Heparin")
    assert first[i:i + 9] == ["CODE//MEDICATION//START//Heparin", "MED//ROUTE//IV", "CODE//MEDICATION//START//N02",
                              "ATC//4-5//BE", "ATC//6-//01", "CODE//HOME_MED//B01", "ATC//4-5//AC", "ATC//6-//06",
                              "MED//PATIENT_REPORTED"]
    assert ["CODE//DIAGNOSIS//ICD//10//E11", "ICD//4-5//65"] == first[first.index("CODE//DIAGNOSIS//ICD//10//E11"):][:2]
    assert "CODE//LAB//220045//bpm" in first  # 'other' (one value: below the bucketing minimum, so no bucket)
    paper_only = _Vocabulary.fit(TRAIN.lazy(), CONFIG, category_predicates(ALL, other=False), None)
    assert "CODE//LAB//220045//bpm" not in paper_only.tokens
    # Right-censoring: subject 1 holds the latest training time, so its record ends at the data cutoff (no padding);
    # with a later cutoff, padding covers min(cutoff, last + follow_up_days) in six-month tokens.
    assert not first[-1].startswith("TIME//")
    later = dataclasses.replace(vocabulary, cutoff=(T0 + timedelta(days=900)).isoformat())
    padded = _tokens(later, _sequences(later, TRAIN.filter(pl.col("subject_id") == 1))["tokens"][0])
    assert padded[:-2] == first and padded[-2:] == ["TIME//6mt", "TIME//6mt"]  # 365 days -> round(365 / 180) = 2
    # Birth: event at the birth instant, gestational age and birth weight after it; death at the end of its date,
    # causes, closing token and EOS; nothing after the death date.
    assert second[:7] == ["[BOS]", "CODE//GENDER//M", "AGE//0d", "YEAR//2145-2149", "CODE//BIRTH//DOCUMENTED",
                          "BIRTH//GA//39w", "CODE//BIRTH//WEIGHT//g"] and second[7].startswith("LABQ//")
    assert second[-4:] == ["CODE//MEDS_DEATH", "CODE//COD//I469", "DEATH//END_PRELIMINARY_CAUSE", "[EOS]"]
    assert "CODE//HOSPITAL_DISCHARGE//DIED" not in second
    race = TRAIN.filter(pl.col("race").is_not_null()).group_by("subject_id").agg(pl.col("race").first(),
                                                                               pl.col("time").first())
    events, _ = vocabulary.events(TRAIN, race, CONFIG)
    death = events.filter(pl.col("tokens").list.last() == vocabulary.ids["[EOS]"])["time"].to_list()
    assert death == [datetime(2150, 1, 1, 23, 59, 59, 999999)]  # last instant of the recorded death date


def test_vital_bins_are_left_exclusive_and_implausible_values_dropped(vocabulary):
    # Hours 0 and 7 lie in different 6-hour windows, so both are kept.
    frame = _events([(5, _hours(0), "LAB//220179//mmHg", 80.0), (5, _hours(7), "LAB//220179//mmHg", 80.5),
                     (5, _hours(13), "LAB//220179//mmHg", 371.0), (5, _hours(19), "LAB//220180//mmHg", 0.0)])
    tokens = _tokens(vocabulary, _sequences(vocabulary, frame)["tokens"][0])
    assert [t for t in tokens if t.startswith("VITAL//")] == ["VITAL//SBP//1", "VITAL//SBP//2"]
    assert len(VITALS["WEIGHT"][0]) + 1 == 130 and len(VITALS["SBP"][0]) + 1 == 13


def test_decode_roundtrips_codes_values_and_stops_at_eos(vocabulary):
    ids = vocabulary.ids
    names = ["TIME//1h-2h", "CODE//DIAGNOSIS//ICD//10//E11", "ICD//4-5//65", "CODE//LAB//RESULT//50912//mg/dL",
             "LABQ//15", "CODE//LAB//RESULT//51463//UNK", "RESULT//Pos", "VITAL//WEIGHT//94",
             "CODE//MEDICATION//START//N02", "ATC//4-5//BE", "ATC//6-//01", "MED//ROUTE//IV", "BIRTH//GA//39w",
             "CODE//MEDS_DEATH", "[EOS]", "CODE//GENDER//F"]
    events, clock, invalid = vocabulary.decode([ids[n] for n in names], T0, np.full(len(names), 0.5))
    assert [e[1] for e in events] == ["DIAGNOSIS//ICD//10//E1165", "LAB//RESULT//50912//mg/dL", "LAB//RESULT//51463//UNK",
                                      "LAB//226512//kg", "MEDICATION//START//N02BE01", "BIRTH//GA", "MEDS_DEATH"]
    assert 4.0 <= events[1][2] <= 5.0 and events[2][3] == "Pos" and events[5][2] == 39.5
    assert 69.8 < events[3][2] < 70.8  # lb bin decoded back to the generated code's kg
    assert invalid == 0 and all(e[0] > T0 for e in events)
    orphan = [ids["ICD//4-5//65"], ids["LABQ//1"], ids["RESULT//Neg"]]
    assert vocabulary.decode(orphan, T0, np.zeros(3))[2] == 3


# ---- temporal cutoffs (internal invariance: nothing after T reaches the prompt at T) ---------------------------------
def test_prompt_uses_only_history_at_the_prediction_time(vocabulary):
    late_race = _events([
        (7, None, "GENDER//F"), (7, T0 - timedelta(days=365.25 * 40), "MEDS_BIRTH"),
        (7, _hours(0), "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"),
        (7, _hours(1), "LAB//220179//mmHg", 120.0), (7, _hours(2), "LAB//220179//mmHg", 130.0),
        (7, _hours(9), "LAB//220179//mmHg", 140.0),
        (7, _hours(500), "HOSPITAL_ADMISSION//URGENT//TRANSFER FROM HOSPITAL", None, None, None, "WHITE"),
        (7, _hours(600), "MEDS_DEATH"),
    ])
    keys = pl.DataFrame({"subject_id": [7, 7, 7], "prediction_time": [_hours(-5), _hours(1.5), _hours(2.5)]},
                        schema={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")})
    before, middle, after = (_tokens(vocabulary, s) for s in _sequences(vocabulary, late_race, keys)["tokens"])
    # No event yet: the timeline starts at T (5 h short of 40 years).
    assert before == ["[BOS]", "CODE//GENDER//F", "AGE//39y", "YEAR//2150-2154"]
    # 1.5 h: the 1 h blood pressure (120, bin (110, 120]) is visible because its window's later value is not recorded.
    assert middle[-3:] == ["TIME//1h-2h", "VITAL//SBP//5", "TIME//15m-1h"]
    # 2.5 h: the 2 h value (130) supersedes it; race (recorded later), death, EOS and padding never appear.
    assert after.count("VITAL//SBP//5") == 0 and after.count("VITAL//SBP//6") == 1
    assert not any(t.startswith("RACE//") or t in ("CODE//MEDS_DEATH", "[EOS]") for t in before + middle + after)
    training = _tokens(vocabulary, _sequences(vocabulary, late_race)["tokens"][0])
    assert "RACE//WHITE" in training and training[-1] == "[EOS]"


def test_prompts_are_invariant_to_rewritten_futures(tmp_path):
    """Algorithm-level counterpart of the shared future-blind rerun: the shared check compares only probabilities,
    which a few-sample zero-shot estimate can leave unchanged. Here every prompt must be identical."""
    meds, _ = build_fixture(tmp_path / "fixture")
    index = meds / "task_labels" / TASK
    perturbed = perturb_future(meds, index, tmp_path / "perturbed", splits=["held_out"], seed=3)
    algorithm = Curiosity(TEST_CONFIG)
    train = pl.scan_parquet(sorted((meds / "data" / "train").rglob("*.parquet")))
    vocabulary = _Vocabulary.fit(train, TEST_CONFIG, category_predicates(MIMIC), algorithm._race(train))
    keys = pl.read_parquet(index / "held_out.parquet").select("subject_id", "prediction_time")

    def prompts(directory):
        frame = pl.read_parquet(sorted((directory / "data" / "held_out").rglob("*.parquet")))
        events, subjects = vocabulary.events(frame, algorithm._race(frame.lazy()).collect(), TEST_CONFIG)
        return vocabulary.sequences(events, subjects, keys)

    original, rewritten = prompts(meds), prompts(perturbed)
    late = pl.read_parquet(sorted((meds / "data" / "held_out").rglob("*.parquet"))).join(keys, on="subject_id").filter(
        pl.col("race").is_not_null() & (pl.col("time") > pl.col("prediction_time")))
    assert late["subject_id"].n_unique() > 0, "the fixture must exercise race recorded after the prediction time"
    assert original.equals(rewritten)


def test_random_futures_never_change_random_prompts(vocabulary):
    """Property check on every representation: append random events after random cutoffs; prompts stay identical."""
    rng = random.Random(0)
    codes = [(c, v, t) for c, v, t in TRAIN.select("code", "numeric_value", "text_value").unique().rows() if c]
    base = TRAIN.filter(pl.col("subject_id") == 1)
    times = sorted(t for t in base["time"].drop_nulls().to_list())
    for trial in range(25):
        cutoff = rng.choice(times) + timedelta(minutes=rng.choice([0, 1, 30, 300]))
        extra = [(1, cutoff + timedelta(microseconds=int(10 ** rng.uniform(0, 11))), *rng.choice(codes),
                  rng.choice([None, "PO"]), rng.choice([None, "ASIAN"])) for _ in range(12)]
        keys = pl.DataFrame({"subject_id": [1], "prediction_time": [cutoff]},
                            schema={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")})
        history = base.filter(pl.col("time").is_null() | (pl.col("time") <= cutoff))
        a = _sequences(vocabulary, pl.concat([base, _events(extra)]), keys)
        b = _sequences(vocabulary, history, keys)
        assert a["tokens"].to_list() == b["tokens"].to_list(), f"trial {trial} at {cutoff}"


# ---- architecture, training and generation -------------------------------------------------------------------------
def test_capacity_dropout_and_context_control_the_qwen2_architecture():
    reference = _model.build_model(CuriosityConfig(), 7105, 0).config
    assert (reference.hidden_size, reference.num_attention_heads, reference.intermediate_size,
            reference.num_hidden_layers) == (768, 12, 3072, 6)
    assert reference.max_position_embeddings == 16384
    assert (reference.rope_scaling or {}).get("rope_type", "default") == "default"  # unscaled RoPE (0826 Sec. 3.2)
    assert sum(p.numel() for p in _model.build_model(CuriosityConfig(), 7105, 0).parameters()) == pytest.approx(62e6, rel=0.02)
    wide = _model.build_model(CuriosityConfig(capacity=2.0, dropout=0.1), 100, 0).config
    assert (wide.hidden_size, wide.num_attention_heads, wide.attention_dropout) == (1536, 24, 0.1)
    assert CuriosityConfig(capacity=0.01).hidden_size == 64  # one head minimum
    assert CuriosityConfig(extension_steps=0, extension_epochs=0).generation_context == 8192
    with pytest.raises(ValueError):
        CuriosityConfig(sequences_per_step=10, batch_size=4)


def test_optimizer_batching_and_context_extension_reach_training(monkeypatch):
    seen, shapes = {}, []
    real_adamw, real_loss = torch.optim.AdamW, _model._loss

    def adamw(params, **kwargs):
        seen.update(kwargs)
        return real_adamw(params, **kwargs)

    def loss(model, batch, pad_id, device):
        shapes.append(tuple(batch.shape))
        return real_loss(model, batch, pad_id, device)

    monkeypatch.setattr(_model.torch.optim, "AdamW", adamw)
    monkeypatch.setattr(_model, "_loss", loss)
    config = CuriosityConfig(capacity=1 / 12, num_layers=1, learning_rate=1e-3, weight_decay=0.05, batch_size=2,
                             sequences_per_step=4, context_length=16, extended_context_length=32, extension_steps=3,
                             extension_epochs=0.0, max_new_tokens=4, max_epochs=1, device="cpu")
    sequences = [np.arange(1, 31) % 9 + 1 for _ in range(4)]
    _, summary = _model.pretrain(sequences, [], config, 10, 0, seed=0)
    assert seen == {"lr": 1e-3, "weight_decay": 0.05}
    stage1 = [s for s in shapes if s[1] == 16]
    windows = int(np.ceil(4 * 30 / 16))  # 4 subjects of 30 tokens (each starting with BOS), packed in 16-token windows
    assert [s[0] for s in stage1] == [2] * (windows // 2) and summary["optimizer_steps"] == int(np.ceil(windows / 4))
    assert summary["extension"]["optimizer_steps"] == 3 and any(s[1] == 32 for s in shapes)


def test_generation_stops_at_horizon_or_stop_token():
    config = CuriosityConfig(capacity=1 / 12, num_layers=1, context_length=32, extension_steps=0, extension_epochs=0,
                             max_new_tokens=8, n_samples=6, horizon_days=0.5, device="cpu")
    model = _model.build_model(config, 10, 0).eval()
    one_day = np.full(10, 86400.0)
    one_day[3] = 0.0
    results = _model.generate(model, [1, 2, 3], config, seed=7, token_seconds=one_day, stop_ids={3: "death"})
    assert len(results) == 6
    for tokens, reason in results:
        assert len(tokens) == 1 and reason == ("death" if tokens[0] == 3 else "horizon")
    assert results == _model.generate(model, [1, 2, 3], config, seed=7, token_seconds=one_day, stop_ids={3: "death"})


def test_aggregate_extends_equation_2_with_fallback():
    key = {"subject_id": 1, "prediction_time": T0}
    late, early = T0 + timedelta(days=10), T0 + timedelta(days=1)
    rows = [
        {**key, "boolean_value": True, "stop_reason": "token_budget", "label_window_end": late, "end_time": early},
        {**key, "boolean_value": False, "stop_reason": "horizon", "label_window_end": early, "end_time": late},
        {**key, "boolean_value": False, "stop_reason": "token_budget", "label_window_end": late, "end_time": early},
        {**key, "boolean_value": False, "stop_reason": "death", "label_window_end": late, "end_time": early},
        {**key, "boolean_value": None, "stop_reason": "horizon", "label_window_end": None, "end_time": late},
        {"subject_id": 2, "prediction_time": T0, "boolean_value": None, "stop_reason": "horizon",
         "label_window_end": None, "end_time": late},
    ]
    result = _aggregate(pl.DataFrame(rows), fallback=0.5)
    assert result.select("subject_id", "samples", "resolved", "positive", "predicted_boolean_probability").rows() == [
        (1, 5, 3, 1, pytest.approx(1 / 3)), (2, 1, 0, 0, 0.5),
    ]


def test_tasks_needing_ungenerable_codes_are_rejected(vocabulary, tmp_path):
    fitted = _Vocabulary.fit(_events([
        (1, T0, "ICU_ADMISSION//MICU"), (1, T0 + timedelta(days=3), "ICU_DISCHARGE//MICU"), (2, T0, "MEDS_DEATH"),
    ]).lazy(), CuriosityConfig(), category_predicates(MIMIC), None)
    _check_generable(TaskExtractorConfig.load(TASK_FILE, PREDICATES_FILE), fitted)
    task = tmp_path / "task.yaml"
    task.write_text(TASK_FILE.read_text().replace('regex: "MEDS_DEATH.*"', 'regex: "^LAB//220045//"'))
    icu = pl.concat([TRAIN, _events([(3, T0, "ICU_ADMISSION//MICU"), (3, T0 + timedelta(days=3), "ICU_DISCHARGE//MICU")])])
    with_other = _Vocabulary.fit(icu.lazy(), CONFIG, category_predicates(ALL), None)
    _check_generable(TaskExtractorConfig.load(task, PREDICATES_FILE), with_other)  # an 'other' code: generable
    paper_only = _Vocabulary.fit(icu.lazy(), CONFIG, category_predicates(ALL, other=False), None)
    with pytest.raises(ValueError, match="cannot generate"):
        _check_generable(TaskExtractorConfig.load(task, PREDICATES_FILE), paper_only)


def test_trajectories_cover_every_key_after_its_prediction_time(signal_run):
    samples = pl.read_parquet(signal_run.artifacts["infer_trajectories"] / "samples.parquet")
    events = pl.read_parquet(signal_run.artifacts["infer_trajectories"] / "events.parquet")
    keys = pl.read_parquet(signal_run.labels_dir / "held_out.parquet").select("subject_id", "prediction_time")
    per_key = samples.group_by("subject_id", "prediction_time").agg(pl.col("sample").sort())
    assert per_key.height == keys.height
    assert per_key["sample"].to_list() == [list(range(TEST_CONFIG.n_samples))] * keys.height
    assert (events["time"] > events["prediction_time"]).all()
    horizon = samples.filter(pl.col("stop_reason") == "horizon")
    assert (horizon["end_time"] > horizon["prediction_time"] + timedelta(days=TEST_CONFIG.horizon_days)).all()
    assert set(samples["stop_reason"]) <= {"death", "end_of_record", "horizon", "token_budget"}
    training = json.loads((signal_run.artifacts["pretrain"] / "payload.json").read_text())["training"]
    assert training["extension"]["optimizer_steps"] == TEST_CONFIG.extension_steps


def test_saved_request_and_config_run_in_a_fresh_process(signal_run, tmp_path):
    """Reload the published predict step's request/config JSON in a new interpreter and execute it again."""
    published = signal_run.artifacts["predict_zero_shot_materialized"]
    script = tmp_path / "rerun.py"
    script.write_text(
        "import dataclasses, pathlib, sys\n"
        "from meds_algorithm.requests import ArtifactDestination, PredictZeroShotMaterializedRequest\n"
        "from curiosity_meds import Curiosity, CuriosityConfig\n"
        "from curiosity_meds.curiosity_meds import create_repository\n"
        "published, output = map(pathlib.Path, sys.argv[1:3])\n"
        "config = CuriosityConfig.load(published / 'config.json')\n"
        "request = PredictZeroShotMaterializedRequest.load(published / 'request.json')\n"
        "request = dataclasses.replace(request, output=ArtifactDestination(output))\n"
        "create_repository().runner.run(Curiosity(config), 'predict_zero_shot_materialized', request,\n"
        "                               profile='zero_shot_materialized')\n"
    )
    output = tmp_path / "predictions"
    subprocess.run([sys.executable, str(script), str(published), str(output)], check=True)
    original = pl.read_parquet(published / "predictions.parquet").drop("boolean_value", strict=False).sort("subject_id")
    assert pl.read_parquet(output / "predictions.parquet").sort("subject_id").equals(original)
    assert json.loads((output / "request.json").read_text())["values"]["task"] == json.loads(
        (published / "request.json").read_text())["values"]["task"]


# ---- size presets, race events, MIMIC-IV-MEDS code layouts ----------------------------------------------------------
@pytest.mark.parametrize(("size", "shape"), [
    ("small", (6, 12, 768)), ("medium", (12, 12, 768)), ("large", (16, 32, 2048)), ("xlarge", (40, 40, 2560)),
])
def test_size_presets_follow_the_papers(size, shape):
    config = CuriosityConfig(size=size)
    assert (config.layers, config.num_heads, config.hidden_size) == shape
    with torch.device("meta"):
        built = _model.build_model(config, 7105, 0)
    parameters = sum(p.numel() for p in built.parameters())
    # 0226 Table 3: S 62M, M 119M, L 1B; 0826: ~4.3B (width not reported).
    expected = {"small": 62e6, "medium": 119e6, "large": 1.09e9, "xlarge": 4.2e9}[size]
    assert parameters == pytest.approx(expected, rel=0.02)


def test_size_can_be_overridden_and_is_validated():
    assert CuriosityConfig(size="medium", num_layers=3).layers == 3
    assert CuriosityConfig(size="large", capacity=0.5).num_heads == 16
    with pytest.raises(ValueError, match="size"):
        CuriosityConfig(size="huge")
    with pytest.raises(ValueError, match="num_layers"):
        CuriosityConfig(num_layers=0)


def test_race_events_are_shown_only_once_recorded(vocabulary):
    fitted = _Vocabulary.fit(_events([
        (8, _hours(0), "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"), (8, _hours(0), "RACE//ASIAN"),
        (9, None, "RACE//WHITE"), (9, _hours(0), "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"),
    ]).lazy(), CONFIG, category_predicates(MIMIC), None)
    frame = _events([
        (8, _hours(0), "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"), (8, _hours(50), "RACE//ASIAN"),
        (9, None, "RACE//WHITE"), (9, _hours(0), "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"),
    ])
    keys = pl.DataFrame({"subject_id": [8, 8, 9], "prediction_time": [_hours(1), _hours(60), _hours(1)]},
                        schema={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")})
    events, subjects = fitted.events(frame, None, CONFIG)
    before, after, static = (_tokens(fitted, s) for s in fitted.sequences(events, subjects, keys)["tokens"])
    assert "RACE//ASIAN" not in before and "RACE//ASIAN" in after and static[1] == "RACE//WHITE"
    assert not any(t.startswith("CODE//RACE//") for t in before + after + static)  # header token, not an event


@pytest.mark.parametrize("version", ["0.1.0", "0.4.0"])
def test_mimic_category_file_matches_both_mimic_iv_meds_layouts(version):
    codes = {
        "0.1.0": {"diagnosis": "DIAGNOSIS//ICD//10//E1165", "vital_sbp_mmhg": "LAB//220179//mmHg",
                  "vital_weight_kg": "LAB//226512//kg", "medication": "MEDICATION//START//Heparin",
                  "lab": "LAB//RESULT//50912//mg/dL"},
        "0.4.0": {"diagnosis": "DIAGNOSIS//ICD//10CM//E1165", "vital_sbp_mmhg": "ICU_CHART//220179//mmHg",
                  "vital_weight_kg": "ICU_CHART//226512//kg", "medication": "MEDICATION//START//Heparin//63323026201",
                  "lab": "LAB//SPECIMEN_COLLECTED//50912//STAT", "race": "RACE//WHITE",
                  "procedure": "PROCEDURE//ICD//10PCS//0BH17EZ"},
    }[version]
    frame = _events([(1, T0, code) for code in codes.values()])
    categorized = dict(_categorized_rows(frame))
    assert categorized == {code: category for category, code in codes.items()}
    assert dict(_categorized_rows(_events([(1, T0, "ICU_CHART//220045//bpm"), (1, T0, "MEDS_BIRTH")]))) == {
        "ICU_CHART//220045//bpm": "other"}  # MEDS_BIRTH only dates the age token


def _categorized_rows(frame):
    from curiosity_meds._tokenizer import _categorized

    return _categorized(frame.lazy(), category_predicates(MIMIC)).select("code", "category").collect().rows()


# ---- the 'other' category ------------------------------------------------------------------------------------------
def test_other_codes_get_value_buckets_and_are_downsampled_causally():
    """A mean arterial pressure (not a paper vital) becomes a code and value-bucket token, latest per 6-hour window,
    and a value is hidden only once the later value of its window is visible."""
    map_code = "ICU_CHART//220052//mmHg"
    train = _events([(20, _hours(k), map_code, 60.0 + k) for k in range(0, 60, 7)])
    fitted = _Vocabulary.fit(train.lazy(), CONFIG, category_predicates(MIMIC), None)
    assert map_code in fitted.quantiles and f"CODE//{map_code}" in fitted.tokens
    frame = _events([(21, _hours(1), map_code, 70.0), (21, _hours(3), map_code, 61.0), (21, _hours(30), map_code, 80.0)])
    keys = pl.DataFrame({"subject_id": [21, 21], "prediction_time": [_hours(2), _hours(4)]},
                        schema={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")})
    events, subjects = fitted.events(frame, None, CONFIG)
    early, late = (_tokens(fitted, s) for s in fitted.sequences(events, subjects, keys)["tokens"])
    assert early.count(f"CODE//{map_code}") == 1 and late.count(f"CODE//{map_code}") == 1  # 1 h, then 3 h replaces it
    assert early[early.index(f"CODE//{map_code}") + 1] != late[late.index(f"CODE//{map_code}") + 1]
    training = _tokens(fitted, fitted.sequences(events, subjects)["tokens"][0])
    assert training.count(f"CODE//{map_code}") == 2  # 1 h dropped (same window as 3 h); 30 h kept
    every = fitted.events(frame, None, dataclasses.replace(CONFIG, other_window_hours=None))
    assert _tokens(fitted, fitted.sequences(*every)["tokens"][0]).count(f"CODE//{map_code}") == 3


def test_a_dataset_without_category_file_runs_with_other_only():
    train = _events([
        (30, None, "SEX//F"), (30, T0 - timedelta(days=365.25 * 50), "MEDS_BIRTH"), (30, _hours(0), "ADMIT"),
        *((30, _hours(k), "HR", 80.0 + k) for k in range(0, 70, 7)), (30, _hours(80), "MEDS_DEATH"),
    ])
    fitted = _Vocabulary.fit(train.lazy(), CONFIG, category_predicates({}), None)
    tokens = _tokens(fitted, _sequences(fitted, train)["tokens"][0])
    assert tokens[:4] == ["[BOS]", "CODE//SEX//F", "AGE//50y", "YEAR//2150-2154"] and "CODE//ADMIT" in tokens
    assert tokens[-2:] == ["CODE//MEDS_DEATH", "[EOS]"] and "CODE//MEDS_BIRTH" not in tokens
