"""Primary behavior: pretrained without labels, Curiosity's generated futures carry the signal into the task.

In the fixture, a medication ordered before the prediction time (SIGNAL_DRUG vs CONTROL_DRUG) decides in-ICU
death. The model never sees a label: it must learn the code -> death continuation during pretraining, emit it
in sampled futures, and the ACES task must then score those futures. In the negative control, death is a coin
flip unrelated to the medication, so the two groups must receive similar probabilities.
"""

import polars as pl

from curiosity_meds._tokenizer import _Vocabulary

from .e2e import CONTROL_CODE, SIGNAL_CODE


def _group_means(result) -> dict[bool, float]:
    events = pl.read_parquet(result.inputs["external_meds_dir"] / "data" / "held_out")
    signal = events.filter(pl.col("code") == SIGNAL_CODE).select("subject_id").unique()
    predictions = pl.read_parquet(result.artifacts["predict_zero_shot_materialized"] / "predictions.parquet")
    grouped = predictions.with_columns(pl.col("subject_id").is_in(signal["subject_id"].implode()).alias("signal"))
    means = grouped.group_by("signal").agg(pl.col("predicted_boolean_probability").mean())
    return dict(means.iter_rows())


def test_primary_behavior(profile, signal_run, shuffled_run):
    assert profile == "zero_shot_materialized"
    vocabulary = _Vocabulary.load(signal_run.artifacts["preprocess_data"] / "vocabulary.json")
    assert {SIGNAL_CODE, CONTROL_CODE, "MEDS_DEATH"} <= vocabulary.codes(), "signal codes must be generable tokens"

    futures = pl.read_parquet(signal_run.artifacts["infer_trajectories"] / "events.parquet")
    assert futures.filter(pl.col("code") == "MEDS_DEATH").height > 0, "futures must contain generated deaths"

    signal = _group_means(signal_run)
    assert signal[True] - signal[False] > 0.5, f"signal not carried into generated futures: {signal}"
    shuffled = _group_means(shuffled_run)
    assert abs(shuffled[True] - shuffled[False]) < 0.3, f"negative control separated groups: {shuffled}"
