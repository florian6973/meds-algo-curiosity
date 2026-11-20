"""Curiosity-specific behavior: tokenization, configuration effects, cutoffs, sampling and Eq. 2 aggregation."""

import dataclasses
import json
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch
import yaml
from aces.config import TaskExtractorConfig
from meds_algorithm.requests import PreprocessDataRequest

from curiosity_meds import Curiosity, CuriosityConfig
from curiosity_meds import _model
from curiosity_meds._tokenizer import SIX_MONTHS, _token_seconds, _Vocabulary, category_predicates, gap_tokens
from curiosity_meds.config import CodeCategory
from curiosity_meds.curiosity_meds import DATASETS_DIR, _aggregate, _check_generable, _prompt

from .e2e import PREDICATES_FILE, TASK_FILE, TEST_CONFIG

T0 = datetime(2150, 1, 1)
MIMIC_CATEGORIES = category_predicates(
    CuriosityConfig(), yaml.safe_load((DATASETS_DIR / "MIMIC-IV.yaml").read_text())["predicates"]
)


def test_category_file_is_chosen_by_dataset_name_or_explicit_file(tmp_path):
    def request(name):
        (tmp_path / name / "metadata").mkdir(parents=True)
        (tmp_path / name / "metadata" / "dataset.json").write_text(json.dumps({"dataset_name": name}))
        return PreprocessDataRequest.from_arguments({"external_meds_dir": str(tmp_path / name),
                                                     "output_data_dir": str(tmp_path / "out")})

    path, source = Curiosity(CuriosityConfig())._category_file(request("MIMIC-IV"))
    assert path == DATASETS_DIR / "MIMIC-IV.yaml" and "dataset_name" in source
    with pytest.raises(ValueError, match="No Curiosity category file for dataset_name 'eICU'"):
        Curiosity(CuriosityConfig())._category_file(request("eICU"))
    explicit = Curiosity(CuriosityConfig(additional_predicates_file=DATASETS_DIR / "MIMIC-IV.yaml"))
    resolved = tmp_path / "resolved_feature_predicates.yaml"  # what the runner composes into staging
    composed = dataclasses.replace(request("other"), predicates_file=resolved)
    assert explicit._category_file(composed) == (resolved, "additional_predicates_file")


def test_category_file_must_define_every_configured_category():
    with pytest.raises(ValueError, match="curiosity_procedure"):
        category_predicates(CuriosityConfig(), {f"curiosity_{n}": {"code": "X"} for n in ("sex", "death", "encounter",
                                                                                           "diagnosis", "lab", "medication")})
    with pytest.raises(ValueError, match="plain predicate"):
        category_predicates(CuriosityConfig(categories=(CodeCategory("lab"),)), {"curiosity_lab": {"expr": "or(a, b)"}})


def _events(rows):
    return pl.DataFrame(rows, orient="row", schema={
        "subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.String, "numeric_value": pl.Float32,
    })


TRAIN = _events([
    (1, None, "GENDER//F", None),
    (1, T0 - timedelta(days=365 * 62), "MEDS_BIRTH", None),
    (1, T0, "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", None),
    *((1, T0 + timedelta(hours=k), "LAB//RESULT//50912//mg/dL", float(k)) for k in range(11)),
    (1, T0 + timedelta(days=3), "DIAGNOSIS//ICD//10//E1165", None),
    (1, T0 + timedelta(days=3), "LAB//220045//bpm", 80.0),  # chartevents: excluded category
    (1, T0 + timedelta(days=400), "MEDS_DEATH", None),
])


@pytest.fixture(scope="module")
def vocabulary():
    return _Vocabulary.fit(TRAIN.lazy(), CuriosityConfig(), MIMIC_CATEGORIES, None)


@pytest.mark.parametrize(("seconds", "expected"), [
    (59, []), (60, ["TIME//1m-5m"]), (300, ["TIME//5m-15m"]), (86400, ["TIME//1d-3d"]),
    (SIX_MONTHS - 1, ["TIME//3mt-6mt"]), (SIX_MONTHS, ["TIME//6mt"]), (400 * 86400, ["TIME//6mt"] * 2),
])
def test_gap_tokens_follow_ethos_buckets(seconds, expected):
    assert gap_tokens(seconds) == expected


def test_time_tokens_decode_to_geometric_bucket_means():
    assert _token_seconds("TIME//1m-5m") == pytest.approx(np.sqrt(60 * 300))
    assert _token_seconds("TIME//6mt") == SIX_MONTHS


def test_fit_keeps_paper_categories_and_train_deciles(vocabulary):
    assert "LAB//220045//bpm" not in vocabulary.codes()  # vitals/charting are excluded (paper Sec. 5.2.8)
    assert {"DIAGNOSIS//ICD//10//E1165", "MEDS_DEATH", "GENDER//F"} <= vocabulary.codes()
    edges = vocabulary.quantiles["LAB//RESULT//50912//mg/dL"]
    assert len(edges) == 11 and edges[0] == 0 and edges[-1] == 10 and edges[5] == pytest.approx(5.0)
    assert {"CODE//DIAGNOSIS//ICD//10//E11", "ICD//4-5//65"} <= set(vocabulary.tokens)


def test_vectorized_transform_matches_scalar_gaps_and_roundtrips(vocabulary):
    held_out = _events([
        (9, T0, "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", None),
        (9, T0 + timedelta(minutes=3), "LAB//RESULT//50912//mg/dL", 2.5),
        (9, T0 + timedelta(days=200), "DIAGNOSIS//ICD//10//E1165", None),
        (9, T0 + timedelta(days=201), "LAB//RESULT//99999//mg/dL", 1.0),  # unseen lab -> [UNK], no decile
    ])
    row = vocabulary.transform(held_out, None).row(0, named=True)
    tokens = [vocabulary.tokens[i] for i in row["tokens"]]
    assert tokens == [
        "[BOS]", "CODE//HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", *gap_tokens(180), "CODE//LAB//RESULT//50912//mg/dL",
        "Q//3", *gap_tokens(200 * 86400 - 180), "CODE//DIAGNOSIS//ICD//10//E11", "ICD//4-5//65",
        *gap_tokens(86400), "[UNK]",
    ]
    events, _, invalid = vocabulary.decode(row["tokens"][1:], T0, np.full(len(row["tokens"]), 0.5))
    assert [code for _, code, _ in events] == [
        "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", "LAB//RESULT//50912//mg/dL", "DIAGNOSIS//ICD//10//E1165",
    ]
    assert 2.0 <= events[1][2] <= 3.0 and invalid == 0
    assert all(time > T0 for time, _, _ in events), "generated events lie strictly after the prediction time"


def test_decode_counts_malformed_sequences(vocabulary):
    ids = vocabulary.ids
    tokens = [ids["ICD//4-5//65"], ids["Q//1"], ids["CODE//LAB//RESULT//50912//mg/dL"], ids["TIME//1h-2h"]]
    events, clock, invalid = vocabulary.decode(tokens, T0, np.zeros(len(tokens)))
    assert invalid == 3  # orphan ICD sub-token, orphan decile, lab without its decile
    assert events == [(T0 + timedelta(microseconds=1), "LAB//RESULT//50912//mg/dL", None)]
    assert clock == T0 + timedelta(seconds=np.sqrt(3600 * 7200))


def test_prompt_stops_at_prediction_time_and_states_the_elapsed_gap(vocabulary):
    history = vocabulary.transform(TRAIN, None).row(0, named=True)
    cutoff = T0 + timedelta(hours=5, minutes=30)
    prompt = [vocabulary.tokens[i] for i in _prompt(history, cutoff, vocabulary)]
    assert prompt.count("CODE//LAB//RESULT//50912//mg/dL") == 6  # hours 0..5 only
    assert "CODE//MEDS_DEATH" not in prompt and "CODE//DIAGNOSIS//ICD//10//E11" not in prompt
    assert prompt[-1] == "TIME//15m-1h"  # 30 minutes since the last real event


def test_capacity_and_dropout_control_the_qwen2_architecture():
    reference = _model.build_model(CuriosityConfig(), 7105, 0).config
    assert (reference.hidden_size, reference.num_attention_heads, reference.intermediate_size, reference.num_hidden_layers) == (768, 12, 3072, 6)
    assert sum(p.numel() for p in _model.build_model(CuriosityConfig(), 7105, 0).parameters()) == pytest.approx(62e6, rel=0.02)
    wide = _model.build_model(CuriosityConfig(capacity=2.0, dropout=0.1), 100, 0).config
    assert (wide.hidden_size, wide.num_attention_heads, wide.attention_dropout) == (1536, 24, 0.1)
    assert CuriosityConfig(capacity=0.01).hidden_size == 64  # one head minimum
    with pytest.raises(ValueError):
        CuriosityConfig(sequences_per_step=10, batch_size=4)


def test_optimizer_and_batching_settings_reach_training(monkeypatch):
    seen, sizes = {}, []
    real_adamw, real_loss = torch.optim.AdamW, _model._loss

    def adamw(params, **kwargs):
        seen.update(kwargs)
        return real_adamw(params, **kwargs)

    def loss(model, batch, pad_id, device):
        sizes.append(len(batch))
        return real_loss(model, batch, pad_id, device)

    monkeypatch.setattr(_model.torch.optim, "AdamW", adamw)
    monkeypatch.setattr(_model, "_loss", loss)
    config = CuriosityConfig(capacity=1 / 12, num_layers=1, learning_rate=1e-3, weight_decay=0.05, batch_size=2,
                             sequences_per_step=4, context_length=16, max_new_tokens=4, max_epochs=1, device="cpu")
    sequences = [np.arange(1, 30) % 9 + 1 for _ in range(4)]
    _, summary = _model.pretrain(sequences, [], config, 10, 0, 0, seed=0)
    assert seen == {"lr": 1e-3, "weight_decay": 0.05}
    windows = int(np.ceil(4 * 30 / 16))  # 4 subjects of 29 tokens + separator, packed into 16-token windows
    assert sizes == [2] * (windows // 2) and summary["optimizer_steps"] == int(np.ceil(windows / 4))


def test_generation_stops_at_horizon_or_stop_token():
    config = CuriosityConfig(capacity=1 / 12, num_layers=1, context_length=32, max_new_tokens=8, n_samples=6,
                             horizon_days=0.5, device="cpu")
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
    _check_generable(TaskExtractorConfig.load(TASK_FILE, PREDICATES_FILE), _Vocabulary.fit(
        _events([(1, T0, "ICU_ADMISSION//MICU", None), (1, T0 + timedelta(days=3), "ICU_DISCHARGE//MICU", None),
                 (2, T0, "MEDS_DEATH", None)]).lazy(), CuriosityConfig(), MIMIC_CATEGORIES, None))
    task = tmp_path / "task.yaml"
    task.write_text(TASK_FILE.read_text().replace('regex: "MEDS_DEATH.*"', 'regex: "^LAB//220045//"'))
    with pytest.raises(ValueError, match="cannot generate"):
        _check_generable(TaskExtractorConfig.load(task, PREDICATES_FILE), vocabulary)


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
