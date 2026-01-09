"""
Objective Registry - Maps FitnessObjective enums to their implementation classes.

Uses auto-registration via BaseObjective.__init_subclass__ - each objective class
registers itself when imported by declaring `objective_type = FitnessObjective.XXX`.

Usage:
    from Objectives.registry import get_objective, ensure_all_registered

    # Ensure all objectives are registered (call once at startup)
    ensure_all_registered()

    # Create an objective instance
    objective = get_objective(FitnessObjective.UTMOS, config, model_data, device, embedding_data)
"""

from typing import Type, Optional
import inspect
from Datastructures.enum import FitnessObjective
from Datastructures.dataclass import ModelData, EmbeddingData
from Objectives.base import BaseObjective


def ensure_all_registered():
    """
    Import all objective modules to trigger auto-registration.
    Call this once at startup to ensure all objectives are available.
    """
    # Naturalness objectives
    from Objectives.Naturalness.PhonemeCountObjective import PhonemeCountObjective
    from Objectives.Naturalness.UtmosObjective import UtmosObjective
    from Objectives.Naturalness.PPLObjective import PPLObjective
    from Objectives.Naturalness.PESQObjective import PesqObjective

    # InterpolationVector objectives
    from Objectives.InterpolationVector.L1Objective import L1Objective
    from Objectives.InterpolationVector.L2Objective import L2Objective

    # Target objectives
    from Objectives.Target.WerTargetObjective import WerTargetObjective
    from Objectives.Target.SbertTargetObjective import SbertTargetObjective
    from Objectives.Target.TextEmbTargetObjective import TextEmbTargetObjective
    from Objectives.Target.WhisperProbObjective import WhisperProbObjective
    from Objectives.Target.Wav2VecDifferentObjective import Wav2VecDifferentObjective
    from Objectives.Target.Wav2VecAsrObjective import Wav2VecAsrObjective

    # GroundTruth objectives
    from Objectives.GroundTruth.WerGtObjective import WerGtObjective
    from Objectives.GroundTruth.SbertGtObjective import SbertGtObjective
    from Objectives.GroundTruth.TextEmbGtObjective import TextEmbGtObjective
    from Objectives.GroundTruth.Wav2VecSimilarObjective import Wav2VecSimilarObjective


def get_objective(
    objective_enum: FitnessObjective,
    config,
    model_data: ModelData,
    device: str = None,
    embedding_data: Optional[EmbeddingData] = None
) -> BaseObjective:
    """
    Factory function to create an objective instance.

    Args:
        objective_enum: The FitnessObjective enum value
        config: Configuration data (ConfigData)
        model_data: Shared model data container (ModelData)
        device: Device to use ('cuda' or 'cpu')
        embedding_data: Optional embedding data for text/audio similarity objectives

    Returns:
        An instance of the appropriate BaseObjective subclass
    """
    objective_cls = BaseObjective.get_class(objective_enum)

    # Check if the objective accepts embedding_data parameter
    sig = inspect.signature(objective_cls.__init__)
    params = sig.parameters

    if 'embedding_data' in params:
        return objective_cls(config, model_data, device=device, embedding_data=embedding_data)
    else:
        return objective_cls(config, model_data, device=device)


def get_all_objective_enums() -> list[FitnessObjective]:
    """Returns a list of all registered FitnessObjective enums."""
    return BaseObjective.get_all_registered_enums()
