"""Run with the installed wheel's Python using -I; no source checkout on sys.path.

This checks packaging, not scientific correctness. Extend for required algorithm resources.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from importlib.resources import files
from pathlib import Path

from curiosity_meds import Curiosity, CuriosityConfig
from curiosity_meds.curiosity_meds import create_repository


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestration", action="store_true")
    args = parser.parse_args()
    assert Curiosity.__module__.startswith("curiosity_meds.")
    assert CuriosityConfig.__module__.startswith("curiosity_meds.")
    repository = create_repository()
    # Bundled defaults: the default config, its dependencies, the tokenizer and Qwen2.
    from aces_relaxations import extract, normalize  # noqa: F401 - the task evaluation dependency must import
    from curiosity_meds._model import build_model

    import polars as pl
    from curiosity_meds._tokenizer import _Vocabulary

    config = CuriosityConfig()
    events = pl.DataFrame({"subject_id": [1] * 12, "time": [f"2100-01-01 {h:02d}:00" for h in range(12)],
                           "code": ["HR"] * 12, "numeric_value": [80.0 + h for h in range(12)]}).with_columns(
                               pl.col("time").str.to_datetime())
    assert "CODE//HR" in _Vocabulary.fit(events.lazy(), config, None).tokens
    from curiosity_meds._tokenizer import EXAMPLE_HIERARCHIES, load_hierarchy

    assert load_hierarchy(EXAMPLE_HIERARCHIES / "MIMIC-IV.yaml")  # the example hierarchy file ships in the wheel
    with tempfile.TemporaryDirectory() as directory:
        assert CuriosityConfig.load(config.save(Path(directory) / "config.json")) == config
    assert build_model(CuriosityConfig(capacity=1 / 12, num_layers=1), 32, 0).config.hidden_size == 64
    resources = files("curiosity_meds") / "resources"
    assert (resources / "runs" / "run.yaml").is_file()
    for profile in repository.profiles:
        assert (resources / "algorithms" / profile / "model.yaml").is_file()
    executable = Path(sys.executable).parent / ("curiosity-meds" + (".exe" if sys.platform == "win32" else ""))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        result = subprocess.run([str(executable), "profiles"], cwd=root, check=True, text=True, capture_output=True)
        assert result.stdout.splitlines() == list(repository.profiles)
        for profile in repository.profiles:
            result = subprocess.run([str(executable), "commands", "--profile", profile], cwd=root, check=True, text=True, capture_output=True)
            assert result.stdout.splitlines() == list(repository.profile_commands(profile))
        if args.orchestration:
            # A preprocessing-only dry run requires no fitted artifacts or task-specific labels.
            # Algorithms with mandatory configuration fields should add representative overrides here.
            (root / "inputs").mkdir()
            (root / "predicates.yaml").write_text("predicates: {}\n", encoding="utf-8")
            result = subprocess.run(
                [str(executable), "run", "--dry-run", f"profile={next(iter(repository.profiles))}",
                 "stages=[preprocess_data]", f"output_dir={root / 'planned'}",
                 f"inputs.meds_dir={root / 'inputs'}", f"inputs.predicates_file={root / 'predicates.yaml'}"],
                cwd=root, check=True, text=True, capture_output=True,
            )
            plan = json.loads(result.stdout)
            assert [stage["name"] for stage in plan["stages"]] == ["preprocess_data"]
            assert not (root / "planned").exists()
    print("Installed imports, commands, packaged resources and requested dry-run checks passed.")


if __name__ == "__main__":
    main()
