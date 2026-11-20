"""Curiosity (arXiv 2508.12104): a Qwen2 decoder pretrained on tokenized MEDS timelines whose Monte Carlo
futures are scored with the original ACES task (zero_shot_materialized)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
import polars as pl
from meds_algorithm import AlgorithmRepository, PayloadMetadata, RelaxationLedger, ZeroShotMaterializedAlgorithm
from meds_algorithm.requests import (
    InferTrajectoriesRequest,
    PredictZeroShotMaterializedRequest,
    PreprocessDataRequest,
    PretrainRequest,
)

from . import _model
from ._tokenizer import DEATH_CODE, SEP, _token_seconds, _Vocabulary, category_predicates
from .config import CuriosityConfig

DATA_FORMAT, MODEL_FORMAT, TRAJECTORY_FORMAT = "curiosity-tokens/v1", "curiosity-qwen2/v1", "curiosity-trajectories/v1"
PREDICTION_KEYS = ["subject_id", "prediction_time"]
_KEYS_PER_CHUNK = 256
_EPOCH = datetime(1970, 1, 1)
DATASETS_DIR = Path(__file__).parent / "datasets"  # one category file per MEDS dataset_name


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _payload(directory: Path) -> dict:
    return json.loads((Path(directory) / "payload.json").read_text())


def _shards(meds_dir: Path, split: str) -> list[Path]:
    return sorted((Path(meds_dir) / "data" / split).rglob("*.parquet"))


def _check_format(artifact, expected: str) -> None:
    found = artifact.read_manifest()["artifact"]["payload_format"]
    if found != expected:
        raise ValueError(f"{artifact.directory} has payload {found}, expected {expected}")


def _load_task(task):
    from aces.config import TaskExtractorConfig

    return TaskExtractorConfig.load(task.criteria_file, task.predicates_file)


def _future_predicates(task) -> set[str]:
    """Plain predicates the generated future must be able to express: every predicate referenced after the
    prediction time (normal form) for a future-only task, otherwise those of the label window."""
    from aces_relaxations import is_future_only, normalize

    if is_future_only(task):
        normalized = normalize(task)
        windows, plain, derived = normalized.windows.values(), normalized.plain_predicates, normalized.derived_predicates
    else:
        windows, plain, derived = [task.windows[task.label_window]], task.plain_predicates, task.derived_predicates
    stack = [name for window in windows for name in window.referenced_predicates | ({window.label} if window.label else set())]
    found = set()
    while stack:
        name = stack.pop()
        if name in derived:
            stack.extend(derived[name].input_predicates)
        elif name in plain and not plain[name].static:
            found.add(name)
    return found


def _check_generable(task, vocabulary: _Vocabulary) -> None:
    """Reject a task whose future-side predicates no generated event can satisfy (paper categories only)."""
    rows = []
    for code in sorted(vocabulary.codes()):
        edges = vocabulary.quantiles.get(code)
        values = [None] if edges is None else [*edges, *((a + b) / 2 for a, b in zip(edges, edges[1:]))]
        rows.extend((code, value) for value in values)
    generable = pl.DataFrame(rows, schema={"code": pl.String, "numeric_value": pl.Float64}, orient="row")
    plain = task.plain_predicates
    missing = []
    for name in sorted(_future_predicates(task)):
        if plain[name].other_cols:
            raise ValueError(f"Predicate {name!r} uses other_cols, which generated events do not carry")
        if not generable.select(plain[name].MEDS_eval_expr().any()).item():
            missing.append(name)
    if missing:
        raise ValueError(
            f"The task needs predicates {missing} after the prediction time, but the fitted vocabulary cannot "
            "generate any matching event (Curiosity keeps only the paper's event categories)"
        )


def _prompt(history: dict, prediction_time: datetime, vocabulary: _Vocabulary) -> list[int]:
    """The subject's tokens up to and including ``prediction_time``, then the elapsed gap to it."""
    prompt = [t for t, time in zip(history["tokens"], history["times"]) if time is None or time <= prediction_time]
    past = [time for time in history["times"] if time is not None and time <= prediction_time]
    if past:
        prompt += vocabulary.prompt_gap((prediction_time - max(past)).total_seconds())
    return prompt


def _aggregate(outcomes: pl.DataFrame, fallback: float) -> pl.DataFrame:
    """Per key: positives / resolved futures (paper Eq. 2, extended to futures outside the task population).

    A future is resolved when ACES realizes the task on it (``boolean_value`` not null) and it is positive, or
    negative with its label window ending no later than the generation (or the future ended in death).
    """
    resolved = pl.col("boolean_value") | (
        (pl.col("boolean_value") == False)  # noqa: E712 - null (unrealized) stays unresolved
        & ((pl.col("stop_reason") == "death") | (pl.col("label_window_end") <= pl.col("end_time")))
    )
    return (
        outcomes.with_columns(resolved.fill_null(False).alias("resolved"))
        .group_by(PREDICTION_KEYS)
        .agg(pl.len().alias("samples"), pl.col("resolved").sum().alias("resolved"),
             (pl.col("resolved") & pl.col("boolean_value").fill_null(False)).sum().alias("positive"))
        .with_columns(
            pl.when(pl.col("resolved") > 0).then(pl.col("positive") / pl.col("resolved"))
            .otherwise(pl.lit(fallback)).alias("predicted_boolean_probability")
        )
        .sort(PREDICTION_KEYS)
    )


class Curiosity(ZeroShotMaterializedAlgorithm):
    name = "curiosity_meds"
    config_type = CuriosityConfig

    def validate_inputs(self, request) -> None:
        super().validate_inputs(request)
        if isinstance(request, PretrainRequest | InferTrajectoriesRequest | PredictZeroShotMaterializedRequest):
            _check_format(request.data, DATA_FORMAT)
        if isinstance(request, InferTrajectoriesRequest):
            _check_format(request.pretrained, MODEL_FORMAT)
            if _payload(request.pretrained.directory)["vocabulary"] != _payload(request.data.directory)["vocabulary"]:
                raise ValueError("The pretrained model was fitted on another vocabulary")
        if isinstance(request, PredictZeroShotMaterializedRequest):
            _check_format(request.trajectories, TRAJECTORY_FORMAT)
            trajectories = _payload(request.trajectories.directory)
            if trajectories["vocabulary"] != _payload(request.data.directory)["vocabulary"]:
                raise ValueError("The trajectories were generated with another vocabulary")
            samples = pl.read_parquet(request.trajectories.directory / "samples.parquet", columns=PREDICTION_KEYS)
            requested = request.index.read().select(PREDICTION_KEYS)
            counts = samples.group_by(PREDICTION_KEYS).len()
            covered = requested.join(counts, on=PREDICTION_KEYS, how="left")
            if covered["len"].null_count() or (covered["len"] != trajectories["n_samples"]).any():
                raise ValueError("The trajectories do not cover every requested key with all samples")
            vocabulary = _Vocabulary.load(request.data.directory / "vocabulary.json")
            _check_generable(_load_task(request.task), vocabulary)

    # ---- preprocess_data -----------------------------------------------------------------------------------
    def preprocess_data(self, request: PreprocessDataRequest) -> PayloadMetadata:
        """Fit the vocabulary and deciles on the train split only, then tokenize every split with them."""
        config, output = self.config, request.output.directory
        splits = sorted(path.name for path in (Path(request.meds_dir) / "data").iterdir() if path.is_dir())
        if "train" not in splits or not _shards(request.meds_dir, "train"):
            raise FileNotFoundError(f"{request.meds_dir} has no train split")
        category_file, category_source = self._category_file(request)
        predicates = (yaml.safe_load(category_file.read_text()) or {}).get("predicates") or {}
        train = pl.scan_parquet(_shards(request.meds_dir, "train"))
        has_race = config.race_column is not None and config.race_column in train.collect_schema()
        vocabulary = _Vocabulary.fit(
            train.select("subject_id", "time", "code", "numeric_value"), config,
            category_predicates(config, predicates), self._race(train) if has_race else None,
        )
        vocabulary.save(output / "vocabulary.json")
        counts = {}
        for split in splits:
            for shard in _shards(request.meds_dir, split):
                frame = pl.read_parquet(shard)
                tokens = vocabulary.transform(
                    frame.select("subject_id", "time", "code", "numeric_value"),
                    self._race(frame.lazy()).collect() if has_race else None,
                )
                target = output / "tokens" / split / shard.relative_to(Path(request.meds_dir) / "data" / split)
                target.parent.mkdir(parents=True, exist_ok=True)
                tokens.write_parquet(target)
                subjects, length = counts.get(split, (0, 0))
                counts[split] = (subjects + tokens.height, length + int(tokens["tokens"].list.len().sum()))
        (output / "payload.json").write_text(json.dumps({
            "vocabulary": _digest(output / "vocabulary.json"),
            "vocabulary_size": len(vocabulary.tokens),
            "fitted_on": "train",
            "category_predicates": {"source": category_source, "path": str(category_file), "digest": _digest(category_file)},
            "meds_shards": {str(p.relative_to(request.meds_dir)): _digest(p) for s in splits for p in _shards(request.meds_dir, s)},
            "splits": {split: {"subjects": n, "tokens": t} for split, (n, t) in counts.items()},
        }, indent=1))
        return PayloadMetadata("curiosity-tokens", 1)

    def _category_file(self, request: PreprocessDataRequest) -> tuple[Path, str]:
        """The dataset category file: an explicit ``additional_predicates_file`` (composed by the runner), else
        the bundled ``datasets/<dataset_name>.yaml`` named by the MEDS dataset's own ``metadata/dataset.json``."""
        if self.config.additional_predicates_file is not None:
            return Path(request.predicates_file), "additional_predicates_file"
        metadata = Path(request.meds_dir) / "metadata" / "dataset.json"
        name = json.loads(metadata.read_text()).get("dataset_name") if metadata.is_file() else None
        bundled = DATASETS_DIR / f"{name}.yaml"
        if not name or not bundled.is_file():
            available = sorted(path.stem for path in DATASETS_DIR.glob("*.yaml"))
            raise ValueError(
                f"No Curiosity category file for dataset_name {name!r} (from {metadata}); bundled: {available}. "
                "Write one (see datasets/MIMIC-IV.yaml) and select it with additional_predicates_file."
            )
        return bundled, f"bundled datasets/{name}.yaml (metadata/dataset.json dataset_name)"

    def _race(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        column = self.config.race_column
        return (
            frame.filter(pl.col(column).is_not_null()).sort("subject_id", "time")
            .group_by("subject_id").agg(pl.col(column).first().cast(pl.String).alias("race"))
        )

    # ---- pretrain ------------------------------------------------------------------------------------------
    def pretrain(self, request: PretrainRequest) -> PayloadMetadata:
        """Next-token pretraining on packed training sequences; tuning sequences only for early stopping."""
        data = request.data.directory
        vocabulary = _Vocabulary.load(data / "vocabulary.json")
        ids = vocabulary.ids

        def sequences(split: str) -> list[np.ndarray]:
            files = sorted((data / "tokens" / split).rglob("*.parquet"))
            if not files:
                return []
            return [np.asarray(row, dtype=np.int64) for row in pl.read_parquet(files, columns=["tokens"])["tokens"].to_list()]

        model, summary = _model.pretrain(
            sequences("train"), sequences("tuning"), self.config, len(vocabulary.tokens), ids[SEP], ids["[PAD]"],
            request.execution.seed,
        )
        model.save_pretrained(request.output.directory / "model")
        (request.output.directory / "payload.json").write_text(json.dumps({
            "vocabulary": _payload(data)["vocabulary"], "training": summary,
        }, indent=1))
        return PayloadMetadata("curiosity-qwen2", 1)

    # ---- infer_trajectories --------------------------------------------------------------------------------
    def infer_trajectories(self, request: InferTrajectoriesRequest) -> PayloadMetadata:
        """``n_samples`` futures per key from its real history up to and including ``prediction_time``."""
        config, data = self.config, request.data.directory
        vocabulary = _Vocabulary.load(data / "vocabulary.json")
        ids = vocabulary.ids
        model = _model.load_model(request.pretrained.directory / "model", config)
        token_seconds = np.array([_token_seconds(t) if t.startswith("TIME//") else 0.0 for t in vocabulary.tokens])
        stop_ids = {ids[SEP]: "end_of_record"}
        if f"CODE//{DEATH_CODE}" in ids:
            stop_ids[ids[f"CODE//{DEATH_CODE}"]] = "death"
        keys = request.index.read().sort(PREDICTION_KEYS)
        files = sorted(p for split in request.index.splits for p in (data / "tokens" / split).rglob("*.parquet"))
        histories = pl.read_parquet(files).filter(pl.col("subject_id").is_in(keys["subject_id"].unique().implode()))
        histories = {row["subject_id"]: row for row in histories.iter_rows(named=True)}
        events, samples = [], []
        for subject_id, prediction_time in keys.iter_rows():
            if subject_id not in histories:
                raise ValueError(f"Subject {subject_id} is not in the prepared splits {list(request.index.splits)}")
            prompt = _prompt(histories[subject_id], prediction_time, vocabulary)
            seed = _model.key_seed(request.execution.seed, subject_id, (prediction_time - _EPOCH) // timedelta(microseconds=1))
            draws = np.random.default_rng(seed)
            for sample, (tokens, reason) in enumerate(_model.generate(model, prompt, config, seed, token_seconds, stop_ids)):
                decoded, clock, invalid = vocabulary.decode(tokens, prediction_time, draws.random(len(tokens)))
                events.extend((subject_id, prediction_time, sample, *event) for event in decoded)
                samples.append((subject_id, prediction_time, sample, reason, clock, len(tokens), invalid, len(prompt)))
        output = request.output.directory
        pl.DataFrame(events, orient="row", schema={
            "subject_id": pl.Int64, "prediction_time": pl.Datetime("us"), "sample": pl.Int32,
            "time": pl.Datetime("us"), "code": pl.String, "numeric_value": pl.Float64,
        }).write_parquet(output / "events.parquet")
        pl.DataFrame(samples, orient="row", schema={
            "subject_id": pl.Int64, "prediction_time": pl.Datetime("us"), "sample": pl.Int32, "stop_reason": pl.String,
            "end_time": pl.Datetime("us"), "generated_tokens": pl.Int32, "invalid_tokens": pl.Int32, "prompt_tokens": pl.Int32,
        }).write_parquet(output / "samples.parquet")
        (output / "payload.json").write_text(json.dumps({
            "vocabulary": _payload(data)["vocabulary"],
            "n_samples": config.n_samples,
            "sampling": {"temperature": config.temperature, "top_k": None, "top_p": None,
                         "key_seed": "sha256(seed:subject_id:prediction_time_us)"},
            "horizon_days": config.horizon_days,
            "max_new_tokens": config.max_new_tokens, "max_attempts": config.max_attempts,
            "prompt": "real events with time <= prediction_time, then the elapsed-gap time tokens",
            "time_units": "time tokens advance by the geometric mean of their bucket; generated events are placed "
                          "strictly after prediction_time",
            "stop_reasons": {"death": "MEDS_DEATH generated (nothing later)", "end_of_record": "separator generated",
                             "horizon": "generated time passed the horizon", "token_budget": "attempts exhausted"},
        }, indent=1))
        return PayloadMetadata("curiosity-trajectories", 1)

    # ---- predict_zero_shot_materialized --------------------------------------------------------------------
    def predict_zero_shot_materialized(self, request: PredictZeroShotMaterializedRequest) -> PayloadMetadata:
        """Evaluate the ACES task on real history plus each future; probability = positives / resolved futures."""
        from aces_relaxations import extract
        from aces_relaxations.extraction import extract_label_window_bounds

        config, task = self.config, _load_task(request.task)
        keys = request.index.read().select(PREDICTION_KEYS).sort(PREDICTION_KEYS)
        trajectories = request.trajectories.directory
        samples = pl.read_parquet(trajectories / "samples.parquet").join(keys, on=PREDICTION_KEYS, how="semi")
        samples = samples.sort(*PREDICTION_KEYS, "sample").with_row_index("pseudo_id").with_columns(pl.col("pseudo_id").cast(pl.Int64))
        generated = pl.read_parquet(trajectories / "events.parquet")
        meds = pl.scan_parquet(sorted((Path(request.meds_dir) / "data").rglob("*.parquet"))).select(
            "subject_id", "time", "code", "numeric_value"
        )
        outcomes = []
        for start in range(0, keys.height, _KEYS_PER_CHUNK):
            chunk = keys.slice(start, _KEYS_PER_CHUNK)
            chunk_samples = samples.join(chunk, on=PREDICTION_KEYS, how="semi")
            real = (
                meds.filter(pl.col("subject_id").is_in(chunk["subject_id"].unique().implode())).collect()
                .join(chunk_samples.select(*PREDICTION_KEYS, "pseudo_id"), on="subject_id")
                .filter(pl.col("time").is_null() | (pl.col("time") <= pl.col("prediction_time")))
            )
            future = generated.join(chunk_samples.select(*PREDICTION_KEYS, "sample", "pseudo_id"), on=[*PREDICTION_KEYS, "sample"])
            columns = [pl.col("pseudo_id").alias("subject_id"), "time", "code", pl.col("numeric_value").cast(pl.Float32)]
            frame = pl.concat([real.select(columns), future.select(columns)]).sort("subject_id", "time", nulls_last=False)
            anchors = chunk_samples.select(pl.col("pseudo_id").alias("subject_id"), "prediction_time")
            anchor = ["subject_id", "prediction_time"]
            # Another trigger could resolve to the same anchor: any positive realization counts as positive.
            labels = extract(task, frame).join(anchors, on=anchor, how="semi").group_by(anchor).agg(pl.col("boolean_value").max())
            bounds = extract_label_window_bounds(task, frame).group_by(anchor).agg(pl.col("label_window_end").max())
            outcomes.append(
                chunk_samples.join(labels, left_on=["pseudo_id", "prediction_time"], right_on=["subject_id", "prediction_time"], how="left")
                .join(bounds, left_on=["pseudo_id", "prediction_time"], right_on=["subject_id", "prediction_time"], how="left")
            )
        per_key = _aggregate(pl.concat(outcomes), config.fallback_probability)
        output = request.output.directory
        per_key.select(*PREDICTION_KEYS, "predicted_boolean_probability").write_parquet(output / "predictions.parquet")
        unresolved = per_key.filter(pl.col("resolved") == 0).select(*PREDICTION_KEYS, "samples", "resolved")
        unresolved.write_ipc(output / "unresolved_keys.arrow")
        resolved_fraction = float(per_key["resolved"].sum() / max(int(per_key["samples"].sum()), 1))
        self._ledger(trajectories, unresolved.height, per_key.height, resolved_fraction).write(output)
        return PayloadMetadata("curiosity-predictions", 1)

    def _ledger(self, trajectories: Path, unresolved: int, total: int, resolved_fraction: float) -> RelaxationLedger:
        generation = _payload(trajectories)
        n = generation["n_samples"]
        return RelaxationLedger(
            query_language="ACES task evaluated on each key's real history (time <= prediction_time) joined with "
                           "each Curiosity-generated future",
            decoded_quantity=f"positives / resolved futures among {n} sampled futures per key (paper Eq. 2 extended); "
                             f"{self.config.fallback_probability} when no future resolves the task",
            exact=False,
            steps=(
                {"name": "monte_carlo", "kind": "approximation", "direction": "unbiased",
                 "detail": f"{n} futures per key at temperature {generation['sampling']['temperature']}, no top-k/top-p"},
                {"name": "generation_limit", "kind": "approximation", "direction": "unknown",
                 "detail": f"horizon {generation['horizon_days']} days, {generation['max_attempts']} x "
                           f"{generation['max_new_tokens']} new tokens; a negative whose label window ends after "
                           "the future's last generated time (unless it ended in death) is unresolved"},
                {"name": "task_population", "kind": "algorithm_assumption", "direction": "unknown",
                 "detail": "futures that violate the task's post-prediction criteria or never realize its windows "
                           "are excluded: estimates P(label | the future stays in the task population); only the "
                           "label window, not criteria windows, is checked against the generation end"},
                {"name": "fallback", "kind": "algorithm_assumption", "direction": "unknown",
                 "detail": f"{unresolved} of {total} keys had no resolved future (unresolved_keys.arrow); "
                           f"{resolved_fraction:.3f} of all futures resolved"},
            ),
        )


# Declarative implementation/completion metadata, never fitted algorithms or mutable runtime state.
IMPLEMENTATIONS = {
    "zero_shot_materialized": Curiosity,
}
PROFILES = tuple(IMPLEMENTATIONS)
STUB_PROFILES: set[str] = set()


def create_repository():
    return AlgorithmRepository(IMPLEMENTATIONS, repo=Path(__file__).resolve().parents[2])
