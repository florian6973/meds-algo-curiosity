"""Algorithm-owned profile execution and fitted-feature inspection adapters."""

import json
import random
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from aces.config import TaskExtractorConfig
from aces_relaxations import extract
from meds_algorithm.conformance.signal import add_split_canaries
from meds_algorithm.conformance.types import E2EResult, MEDSDevRun

from curiosity_meds import CuriosityConfig
from curiosity_meds._tokenizer import _Vocabulary
from curiosity_meds.curiosity_meds import DATASETS_DIR, create_repository

FIXTURES = Path(__file__).parent / "fixtures"
TASK_FILE = FIXTURES / "in_icu_mortality.yaml"
PREDICATES_FILE = FIXTURES / "predicates.yaml"
TASK = "in_icu_mortality"
SIGNAL_CODE, CONTROL_CODE = "MEDICATION//START//SIGNAL_DRUG", "MEDICATION//START//CONTROL_DRUG"
STEPS = ("preprocess_data", "pretrain", "infer_trajectories", "predict_zero_shot_materialized")

# A two-head, two-layer Curiosity (capacity 2/12 of Curiosity-S) on CPU, with a short context and budget.
TEST_CONFIG = CuriosityConfig(
    capacity=2 / 12, num_layers=2, learning_rate=3e-3, batch_size=8, sequences_per_step=8, context_length=128,
    max_epochs=25, early_stopping_patience=5, warmup_fraction=0.05, device="cpu",
    n_samples=8, max_new_tokens=48, max_attempts=3, horizon_days=60.0,
    additional_predicates_file=DATASETS_DIR / "MIMIC-IV.yaml",  # explicit route; MEDS-DEV runs use dataset_name
)
# Real data (MIMIC-IV demo): the paper's defaults (Curiosity-S, n=20, 3 x 2,000 tokens, 365 d) on the GPU when
# available; 8-sequence optimizer steps because the demo's ~100 subjects fill only a few 8,192-token windows.
DEMO_CONFIG = CuriosityConfig(sequences_per_step=8)


def build_fixture(root: Path, *, seed: int = 0, shuffle_outcomes: bool = False, sizes=(1000, 100, 40)):
    """ICU admissions in MIMIC-IV-MEDS code conventions. A medication ordered one hour into the ICU stay decides
    in-ICU death (SIGNAL_DRUG) or discharge (CONTROL_DRUG) 60-90 h after ICU admission; ``shuffle_outcomes`` makes the
    outcome a coin flip instead (negative control). Labels come from the fixture task through ACES."""
    root = Path(root)
    rng = random.Random(seed)
    rows, splits, subject = [], [], 0
    base = datetime(2150, 1, 1)
    for split, size in zip(("train", "tuning", "held_out"), sizes, strict=True):
        for _ in range(size):
            subject += 1
            admit = base + timedelta(days=rng.randrange(365), hours=rng.randrange(24))
            icu = admit + timedelta(hours=2)
            signal = rng.random() < 0.5
            dies = (rng.random() < 0.5) if shuffle_outcomes else signal
            stay = timedelta(hours=rng.randint(60, 90))
            race = rng.choice(["WHITE", "BLACK/AFRICAN AMERICAN", "ASIAN"])
            events = [
                (None, f"GENDER//{rng.choice('FM')}", None, None),
                (admit - timedelta(days=365 * rng.randint(40, 85)), "MEDS_BIRTH", None, None),
                (admit, "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM", None, race),
                (icu, "ICU_ADMISSION//Medical Intensive Care Unit (MICU)", None, None),
                (icu + timedelta(hours=1), SIGNAL_CODE if signal else CONTROL_CODE, None, None),
                *((icu + timedelta(hours=h), "LAB//RESULT//50912//mg/dL", round(rng.gauss(1.2, 0.4), 2), None)
                  for h in (0, 24)),  # the second lab is at the prediction time (ICU admission + 24 h)
            ]
            if rng.random() < 0.5:  # an earlier admission with a billed diagnosis
                prior = admit - timedelta(days=365)
                events += [
                    (prior, "HOSPITAL_ADMISSION//URGENT//TRANSFER FROM HOSPITAL", None, race),
                    (prior + timedelta(days=3), "HOSPITAL_DISCHARGE//HOME", None, None),
                    (prior + timedelta(days=3), f"DIAGNOSIS//ICD//10//{rng.choice(['E1165', 'I10', 'J449'])}", None, None),
                ]
            end = icu + stay
            if dies:
                events.append((end, "MEDS_DEATH", None, None))
            else:
                events += [(end, "ICU_DISCHARGE//Medical Intensive Care Unit (MICU)", None, None),
                           (end + timedelta(days=1), "HOSPITAL_DISCHARGE//HOME", None, None)]
            rows += [(subject, split, *event) for event in events]
            splits.append((subject, split))
    frame = pl.DataFrame(rows, orient="row", schema={
        "subject_id": pl.Int64, "split": pl.String, "time": pl.Datetime("us"), "code": pl.String,
        "numeric_value": pl.Float32, "race": pl.String,
    }).sort("subject_id", "time", nulls_last=False, maintain_order=True)
    task = TaskExtractorConfig.load(TASK_FILE, PREDICATES_FILE)
    labels = extract(task, frame.select("subject_id", "time", "code", "numeric_value"))
    split_of = pl.DataFrame(splits, orient="row", schema={"subject_id": pl.Int64, "split": pl.String})
    for split in ("train", "tuning", "held_out"):
        (root / "data" / split).mkdir(parents=True, exist_ok=True)
        frame.filter(pl.col("split") == split).drop("split").write_parquet(root / "data" / split / "0.parquet")
        (root / "task_labels" / TASK).mkdir(parents=True, exist_ok=True)
        labels.join(split_of.filter(pl.col("split") == split), on="subject_id", how="semi").write_parquet(
            root / "task_labels" / TASK / f"{split}.parquet"
        )
    (root / "metadata").mkdir(exist_ok=True)
    frame.select(pl.col("code").unique().sort()).write_parquet(root / "metadata" / "codes.parquet")
    split_of.write_parquet(root / "metadata" / "subject_splits.parquet")
    # Synthetic, but in MIMIC-IV-MEDS 0.1.0 code conventions, so it uses the bundled MIMIC-IV category file.
    (root / "metadata" / "dataset.json").write_text(json.dumps({"dataset_name": "MIMIC-IV", "meds_version": "0.4"}))
    return root, add_split_canaries(root)


def run_e2e(
    tmp_path,
    *,
    profile: str,
    seed: int = 0,
    external_meds_dir=None,
    external_labels_dir=None,
    external_index_dir=None,
    external_evaluation_labels_dir=None,
    external_predicates_file=None,
    external_task_criteria_file=None,
    splits=("held_out",),
    config: CuriosityConfig | None = None,
    shuffle_outcomes: bool = False,
):
    """preprocess_data -> pretrain -> infer_trajectories -> predict through the core runner, one fresh algorithm
    instance per step. Without ``external_meds_dir`` it builds the synthetic fixture and its task."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = config or (TEST_CONFIG if external_meds_dir is None else DEMO_CONFIG)
    canaries = {}
    if external_meds_dir is None:
        external_meds_dir, canaries = build_fixture(tmp_path / "fixture", seed=seed, shuffle_outcomes=shuffle_outcomes)
        external_labels_dir = external_labels_dir or external_meds_dir / "task_labels" / TASK
        external_task_criteria_file = external_task_criteria_file or TASK_FILE
        external_predicates_file = external_predicates_file or PREDICATES_FILE
    if external_labels_dir is None and external_index_dir is None:
        raise ValueError("an index (or labels) directory is required with an external MEDS dataset")
    external_index_dir = external_index_dir or external_labels_dir
    external_evaluation_labels_dir = external_evaluation_labels_dir or external_index_dir
    common = {"seed": seed, "config_file": str(config.save(tmp_path / "config.json"))}
    artifacts = {step: tmp_path / name for step, name in zip(STEPS, ("data", "pretrained", "trajectories", "predictions"))}
    repository = create_repository()
    repository.run("preprocess_data", {
        "external_meds_dir": str(external_meds_dir), "output_data_dir": str(artifacts["preprocess_data"]), **common,
    }, profile=profile)
    repository.run("pretrain", {
        "input_data_dir": str(artifacts["preprocess_data"]),
        "output_pretrained_algorithm_dir": str(artifacts["pretrain"]), **common,
    }, profile=profile)
    repository.run("infer_trajectories", {
        "input_data_dir": str(artifacts["preprocess_data"]),
        "input_pretrained_algorithm_dir": str(artifacts["pretrain"]),
        "external_index_dir": str(external_index_dir), "splits": list(splits),
        "output_trajectories_dir": str(artifacts["infer_trajectories"]), **common,
    }, profile=profile)
    predict = {
        "input_data_dir": str(artifacts["preprocess_data"]),
        "input_trajectories_dir": str(artifacts["infer_trajectories"]),
        "external_meds_dir": str(external_meds_dir),
        "external_index_dir": str(external_index_dir), "splits": list(splits),
        "external_task_criteria_file": str(external_task_criteria_file),
        "output_predictions_dir": str(artifacts["predict_zero_shot_materialized"]),
        "external_evaluation_labels_dir": str(external_evaluation_labels_dir), **common,
    }
    if external_predicates_file is not None:
        predict["external_predicates_file"] = str(external_predicates_file)
    repository.run("predict_zero_shot_materialized", predict, profile=profile)
    inputs = {
        "seed": seed,
        "external_meds_dir": Path(external_meds_dir),
        "external_index_dir": Path(external_index_dir),
        "external_evaluation_labels_dir": Path(external_evaluation_labels_dir),
        "external_task_criteria_file": Path(external_task_criteria_file),
        "config": config,
        "shuffle_outcomes": shuffle_outcomes,
    }
    if external_predicates_file is not None:
        inputs["external_predicates_file"] = Path(external_predicates_file)
    return E2EResult(artifacts=artifacts, labels_dir=Path(external_index_dir), splits=list(splits), inputs=inputs,
                     canaries=canaries)


def fitted_codes(artifacts: Mapping[str, Path]) -> set[str]:
    """Every MEDS code with a token or decile statistic in the train-fitted vocabulary."""
    return _Vocabulary.load(Path(artifacts["preprocess_data"]) / "vocabulary.json").codes()


def prepare_meds_dev_run(meds_dev, tmp_path, *, profile: str):
    """Pretrain and predict a small fixture through MEDS-DEV's full mode with model.yaml's default config.

    The defaults are the paper's (Curiosity-S, n = 20, 3 x 2,000 new tokens), run on the GPU when available; the
    fixture is kept small so the run checks installation and command wiring, not learning.
    """
    dataset, _ = build_fixture(Path(tmp_path) / "fixture", sizes=(200, 20, 10))
    labels_dir = dataset / "task_labels" / TASK
    output_dir = Path(tmp_path) / "meds-dev-run"
    args = [
        "mode=full",
        "dataset_type=full",
        f"dataset_dir={dataset}",
        f"labels_dir={labels_dir}",
        "dataset_name=synthetic_icu",
        f"task_name={TASK}",
        f"task_criteria_path={TASK_FILE}",
        f"predicates_path={PREDICATES_FILE}",
        f"output_dir={output_dir}",
    ]
    predictions_dir = output_dir / "synthetic_icu" / TASK / "predict" / "predictions"
    return MEDSDevRun(args=args, predictions_dir=predictions_dir, labels_dir=labels_dir, splits=["held_out"])
