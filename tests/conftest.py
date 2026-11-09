from pathlib import Path

import pytest
import torch
from meds_algorithm.conformance.cases import AlgorithmCase

from curiosity_meds.config import CodeCategory, CuriosityConfig
from curiosity_meds.curiosity_meds import STUB_PROFILES, create_repository

from .e2e import TEST_CONFIG, fitted_codes, prepare_meds_dev_run, run_e2e

# Many-core aarch64 CPUs run this tiny model fastest single-threaded; it also keeps CPU runs reproducible.
torch.set_num_threads(1)


@pytest.fixture(scope="session")
def algorithm_case():
    return AlgorithmCase(
        root=Path(__file__).resolve().parents[1],
        name="curiosity_meds",
        create_repository=create_repository,
        configurations=(
            CuriosityConfig(),  # paper defaults: Curiosity-S, MIMIC-IV categories, n=20
            TEST_CONFIG,
            CuriosityConfig(capacity=2.0, categories=(CodeCategory("all", ".*", top_k=5000, numeric=True),),
                            race_column=None, horizon_days=730.0, n_samples=64),
        ),
        stub_profiles=STUB_PROFILES,
        run_e2e=run_e2e,
        fitted_codes=fitted_codes,
        prepare_meds_dev_run=prepare_meds_dev_run,
    )


@pytest.fixture(scope="session", params=create_repository().profiles)
def profile(request):
    return request.param


@pytest.fixture(scope="session")
def signal_run(tmp_path_factory):
    """One real lifecycle on the synthetic ICU fixture, shared by the behavior and artifact tests."""
    return run_e2e(tmp_path_factory.mktemp("signal"), profile="zero_shot_materialized")


@pytest.fixture(scope="session")
def shuffled_run(tmp_path_factory):
    return run_e2e(tmp_path_factory.mktemp("shuffled"), profile="zero_shot_materialized", shuffle_outcomes=True)
