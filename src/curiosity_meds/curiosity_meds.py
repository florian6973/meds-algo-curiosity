"""Algorithm-owned algorithm. Implement typed methods; core owns staging and publication."""

from pathlib import Path

from meds_algorithm import AlgorithmRepository, PayloadMetadata
from meds_algorithm import ZeroShotMaterializedAlgorithm
from meds_algorithm.requests import PreprocessDataRequest
from meds_algorithm.requests import PretrainRequest
from meds_algorithm.requests import InferTrajectoriesRequest
from meds_algorithm.requests import PredictZeroShotMaterializedRequest

from .config import CuriosityConfig


class Curiosity(ZeroShotMaterializedAlgorithm):
    name = "curiosity_meds"
    config_type = CuriosityConfig

    # Extend validate_inputs(request) for payload/vocabulary compatibility. Call super() unless
    # explicitly documenting support for compatible external algorithms or different configurations.

    def preprocess_data(self, request: PreprocessDataRequest) -> PayloadMetadata:
        """Write payload into request.output.directory; return its format and version."""
        raise NotImplementedError("Implement Curiosity.preprocess_data")



    def pretrain(self, request: PretrainRequest) -> PayloadMetadata:
        """Write payload into request.output.directory; return its format and version."""
        raise NotImplementedError("Implement Curiosity.pretrain")







    def infer_trajectories(self, request: InferTrajectoriesRequest) -> PayloadMetadata:
        """Write payload into request.output.directory; return its format and version."""
        raise NotImplementedError("Implement Curiosity.infer_trajectories")

    def predict_zero_shot_materialized(self, request: PredictZeroShotMaterializedRequest) -> PayloadMetadata:
        """Write payload into request.output.directory; return its format and version."""
        raise NotImplementedError("Implement Curiosity.predict_zero_shot_materialized")



# Declarative implementation/completion metadata, never fitted algorithms or mutable runtime state.
IMPLEMENTATIONS = {
    "zero_shot_materialized": Curiosity,
}
PROFILES = tuple(IMPLEMENTATIONS)
STUB_PROFILES = set(PROFILES)


def create_repository():
    return AlgorithmRepository(IMPLEMENTATIONS, repo=Path(__file__).resolve().parents[2])
