"""Curiosity-specific behavior: dataset-agnostic 0826 tokenization, internal temporal cutoffs, configuration effects,
sampling and Eq. 2 aggregation."""

import dataclasses
import json
import random
import subprocess
import sys
from datetime import datetime, timedelta
from statistics import NormalDist

import numpy as np
import polars as pl
import pytest
import torch
from aces.config import TaskExtractorConfig
from meds_algorithm.conformance.temporal import perturb_future

from curiosity_meds import Curiosity, CuriosityConfig
from curiosity_meds import _model
from curiosity_meds._tokenizer import LAB_BUCKETS, LAB_LEVELS, SIX_MONTHS, _token_seconds, _Vocabulary, age_token, gap_tokens
from curiosity_meds.curiosity_meds import _aggregate, _check_generable

from .e2e import PREDICATES_FILE, TASK, TASK_FILE, TEST_CONFIG, build_fixture

T0 = datetime(2150, 1, 1, 8)
CONFIG = CuriosityConfig(min_code_count=1, min_code_share=0.0, min_bucket_observations=5, follow_up_days=365.0)
SCHEMA = {"subject_id": pl.Int64, "time": pl.Datetime("us"), "code": pl.String, "numeric_value": pl.Float32,
          "text_value": pl.String, "route": pl.String, "race": pl.String}
LAB, TEXT_LAB, MED, DX = "LAB//RESULT//50912//mg/dL", "LAB//RESULT//51463//UNK", "MEDICATION//START//Heparin", "DIAGNOSIS//ICD//10//E1165"


def _events(rows):
    rows = [(*row, *([None] * (7 - len(row)))) for row in rows]
    return pl.DataFrame(rows, orient="row", schema=SCHEMA)


def _hours(h):
    return T0 + timedelta(hours=h)


def _keys(subject, *times):
    return pl.DataFrame({"subject_id": [subject] * len(times), "prediction_time": list(times)},
                        schema={"subject_id": pl.Int64, "prediction_time": pl.Datetime("us")})


TRAIN = _events([
    (1, None, "GENDER//F"),
    (1, T0 - timedelta(days=365.25 * 62), "MEDS_BIRTH"),
    (1, _hours(0), "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", None, None, None, "WHITE"),
    (1, _hours(0), "LAB//SPECIMEN_COLLECTED//50912//STAT"),  # a value-less code: just its token
    *((1, _hours(7 * k + 1), LAB, float(k)) for k in range(10)),  # 7 h apart: never in one 6-hour window
    *((1, _hours(k + 1), TEXT_LAB, None, value) for k, value in enumerate(["Neg", "Pos", "Neg", "Weird", "Pos", "Neg"])),
    (1, _hours(2), TEXT_LAB),  # a text code without a result -> missing-value token
    (1, _hours(6), MED, None, None, "IV"), (1, _hours(30), MED, None, None, "IV"), (1, _hours(31), MED, None, None, "PO"),
    (1, _hours(70), DX), (1, _hours(75), DX), (1, _hours(76), "DIAGNOSIS//ICD//10//Z999"),
    (2, None, "GENDER//M"),
    (2, T0 - timedelta(days=2), "MEDS_BIRTH"),
    (2, _hours(0), "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"),
    (2, _hours(1), "MEDS_DEATH"),
    (2, _hours(30), "HOSPITAL_DISCHARGE//DIED"),  # after the death date: dropped
])


def _race(frame):
    return frame.filter(pl.col("race").is_not_null()).sort("subject_id", "time").group_by("subject_id").agg(
        pl.col("race").first(), pl.col("time").first())


@pytest.fixture(scope="module")
def vocabulary():
    return _Vocabulary.fit(TRAIN.lazy(), CONFIG, _race(TRAIN).lazy())


def _tokens(vocabulary, sequence):
    return [vocabulary.tokens[i] for i in sequence]


def _sequences(vocabulary, frame, keys=None, config=CONFIG):
    events, subjects = vocabulary.events(frame, _race(frame), config)
    return vocabulary.sequences(events, subjects, keys)


# ---- time, age and value tokens ------------------------------------------------------------------------------------
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


def test_value_buckets_are_thirty_shared_percentile_levels(vocabulary):
    assert LAB_BUCKETS == 30 and LAB_LEVELS[0] == pytest.approx(0.001) and LAB_LEVELS[-1] == pytest.approx(0.999)
    widths = np.diff([NormalDist().inv_cdf(level) for level in LAB_LEVELS])
    assert np.allclose(widths, widths[0]) and len(widths) == 28
    edges = vocabulary.quantiles[LAB]
    assert len(edges) == 31 and edges[0] == 0 and edges[-1] == 9


# ---- dataset-agnostic tokenization ---------------------------------------------------------------------------------
def test_every_code_is_one_whole_token_with_its_value_tokens(vocabulary):
    first, second = (_tokens(vocabulary, s) for s in _sequences(vocabulary, TRAIN)["tokens"])
    assert first[:5] == ["[BOS]", "CODE//GENDER//F", "RACE//WHITE", "AGE//62y", "YEAR//2150-2154"]
    assert first[5:7] == ["CODE//HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", "CODE//LAB//SPECIMEN_COLLECTED//50912//STAT"]
    i = first.index(f"CODE//{LAB}")
    assert first[i + 1].startswith("LABQ//")  # a numeric code: its value bucket
    assert [first[j + 1] for j, t in enumerate(first) if t == f"CODE//{TEXT_LAB}"][:3] == [
        "RESULT//Neg", "RESULT//Pos", "RESULT//[MISSING]"]  # 1 h Neg; at 2 h Pos then no result (row order)
    assert first[first.index(f"CODE//{MED}") + 1] == "ATTR//ROUTE//IV"  # the optional route column
    assert f"CODE//{DX}" in first and not any(t.startswith(("ICD//", "ATC//")) for t in first)  # no hierarchy split
    # Death at the last instant of its date, then EOS; nothing after it.
    assert second[-2:] == ["CODE//MEDS_DEATH", "[EOS]"] and "CODE//HOSPITAL_DISCHARGE//DIED" not in second
    events, _ = vocabulary.events(TRAIN, _race(TRAIN), CONFIG)
    assert events.filter(pl.col("tokens").list.last() == vocabulary.ids["[EOS]"])["time"].to_list() == [
        datetime(2150, 1, 1, 23, 59, 59, 999999)]


def test_rare_codes_become_unk_and_rare_values_other():
    rare = dataclasses.replace(CONFIG, min_code_count=2)
    fitted = _Vocabulary.fit(TRAIN.lazy(), rare, _race(TRAIN).lazy())
    first = _tokens(fitted, _sequences(fitted, TRAIN, config=rare)["tokens"][0])
    assert first[1] == "[UNK]"  # GENDER//F occurs once in train: rare, an unknown header token (as in the paper)
    # Codes seen once: GENDER//F, the collection marker and Z999.
    assert "CODE//DIAGNOSIS//ICD//10//Z999" not in fitted.tokens and first.count("[UNK]") == 3
    assert "RESULT//Weird" not in fitted.tokens and "RESULT//[OTHER]" in first  # a rare result value
    assert "ATTR//ROUTE//PO" not in fitted.tokens  # a rare attribute value: no token
    floor = dataclasses.replace(CONFIG, min_code_count=1, min_code_share=0.5)
    assert len(_Vocabulary.fit(TRAIN.lazy(), floor, None).codes()) <= 1  # the share threshold applies too


def test_numeric_codes_are_downsampled_causally():
    """The latest value per numeric code in each 6-hour window; a value is hidden only once the later value of its
    window is visible, so prompts never depend on the future."""
    frame = _events([(21, _hours(k), "HR", 60.0 + k) for k in range(0, 60, 7)]
                    + [(22, _hours(1), "HR", 70.0), (22, _hours(3), "HR", 61.0), (22, _hours(30), "HR", 80.0)])
    fitted = _Vocabulary.fit(frame.lazy(), CONFIG, None)
    assert "HR" in fitted.quantiles
    early, late = (_tokens(fitted, s) for s in _sequences(fitted, frame, _keys(22, _hours(2), _hours(4)))["tokens"])
    assert early.count("CODE//HR") == 1 and late.count("CODE//HR") == 1  # 1 h, then 3 h replaces it
    assert early[early.index("CODE//HR") + 1] != late[late.index("CODE//HR") + 1]
    training = _tokens(fitted, _sequences(fitted, frame.filter(pl.col("subject_id") == 22))["tokens"][0])
    assert training.count("CODE//HR") == 2  # 1 h dropped (same window as 3 h); 30 h kept
    every = dataclasses.replace(CONFIG, downsample_hours=None)
    assert _tokens(fitted, _sequences(fitted, frame.filter(pl.col("subject_id") == 22), config=every)["tokens"][0]).count(
        "CODE//HR") == 3


def test_decode_roundtrips_codes_values_and_stops_at_eos(vocabulary):
    ids = vocabulary.ids
    names = ["TIME//1h-2h", f"CODE//{DX}", f"CODE//{LAB}", "LABQ//15", f"CODE//{TEXT_LAB}", "RESULT//Pos",
             f"CODE//{MED}", "ATTR//ROUTE//IV", "CODE//MEDS_DEATH", "[EOS]", "CODE//GENDER//F"]
    events, clock, invalid = vocabulary.decode([ids[n] for n in names], T0, np.full(len(names), 0.5))
    assert [e[1] for e in events] == [DX, LAB, TEXT_LAB, MED, "MEDS_DEATH"]
    assert 0 <= events[1][2] <= 9 and events[2][3] == "Pos" and invalid == 0 and all(e[0] > T0 for e in events)
    assert clock == T0 + timedelta(seconds=np.sqrt(3600 * 7200))
    orphan = [ids["LABQ//1"], ids["RESULT//Neg"]]
    assert vocabulary.decode(orphan, T0, np.zeros(2))[2] == 2


# ---- temporal cutoffs (internal invariance: nothing after T reaches the prompt at T) ---------------------------------
def test_prompt_uses_only_history_at_the_prediction_time():
    frame = _events([
        (7, None, "GENDER//F"), (7, T0 - timedelta(days=365.25 * 40), "MEDS_BIRTH"),
        (7, _hours(0), "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"),
        *((7, _hours(h), "HR", 60.0 + h) for h in (1, 2, 9, 16, 23, 30)),
        (7, _hours(500), "HOSPITAL_ADMISSION//URGENT//TRANSFER FROM HOSPITAL", None, None, None, "WHITE"),
        (7, _hours(600), "MEDS_DEATH"),
    ])
    fitted = _Vocabulary.fit(frame.lazy(), CONFIG, _race(frame).lazy())
    before, middle, after = (_tokens(fitted, s) for s in
                             _sequences(fitted, frame, _keys(7, _hours(-5), _hours(1.5), _hours(2.5)))["tokens"])
    assert before == ["[BOS]", "CODE//GENDER//F", "AGE//39y", "YEAR//2150-2154"]  # no event yet: timeline starts at T
    assert middle[-3:-1] == ["CODE//HR", middle[-2]] and middle[-2].startswith("LABQ//")  # 1 h value visible
    assert middle[-1] == "TIME//15m-1h"  # 30 minutes since it; the 2 h value is not yet recorded
    assert after.count("CODE//HR") == 1 and after[after.index("CODE//HR") + 1] != middle[-2]  # 2 h replaced 1 h
    assert not any(t.startswith("RACE//") or t in ("CODE//MEDS_DEATH", "[EOS]") for t in before + middle + after)
    training = _tokens(fitted, _sequences(fitted, frame)["tokens"][0])
    assert "RACE//WHITE" in training and training[-1] == "[EOS]"


def test_prompts_are_invariant_to_rewritten_futures(tmp_path):
    """Algorithm-level counterpart of the shared future-blind rerun: the shared check compares only probabilities,
    which a few-sample zero-shot estimate can leave unchanged. Here every prompt must be identical."""
    meds, _ = build_fixture(tmp_path / "fixture")
    index = meds / "task_labels" / TASK
    perturbed = perturb_future(meds, index, tmp_path / "perturbed", splits=["held_out"], seed=3)
    algorithm = Curiosity(TEST_CONFIG)
    train = pl.scan_parquet(sorted((meds / "data" / "train").rglob("*.parquet")))
    vocabulary = _Vocabulary.fit(train, TEST_CONFIG, algorithm._race(train))
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
                  rng.choice([None, "IV"]), rng.choice([None, "ASIAN"])) for _ in range(12)]
        history = base.filter(pl.col("time").is_null() | (pl.col("time") <= cutoff))
        a = _sequences(vocabulary, pl.concat([base, _events(extra)]), _keys(1, cutoff))
        b = _sequences(vocabulary, history, _keys(1, cutoff))
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




def test_tasks_needing_ungenerable_codes_are_rejected(tmp_path):
    icu = _events([(3, T0, "ICU_ADMISSION//MICU"), (3, T0 + timedelta(days=3), "ICU_DISCHARGE//MICU"),
                   (4, T0, "MEDS_DEATH"), (5, T0, "ICU_CHART//220052//mmHg", 70.0)])
    fitted = _Vocabulary.fit(icu.lazy(), CONFIG, None)
    _check_generable(TaskExtractorConfig.load(TASK_FILE, PREDICATES_FILE), fitted)
    task = tmp_path / "task.yaml"
    task.write_text(TASK_FILE.read_text().replace('regex: "MEDS_DEATH.*"', 'regex: "^ICU_CHART//220052//"'))
    _check_generable(TaskExtractorConfig.load(task, PREDICATES_FILE), fitted)  # any code of the training split
    without = _Vocabulary.fit(icu.filter(pl.col("subject_id") != 5).lazy(), CONFIG, None)
    with pytest.raises(ValueError, match="cannot generate"):
        _check_generable(TaskExtractorConfig.load(task, PREDICATES_FILE), without)

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




# ---- size presets --------------------------------------------------------------------------------------------------
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


