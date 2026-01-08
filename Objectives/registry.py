"""
Objective Registry - Maps FitnessObjective enums to their implementation classes.

Uses lazy imports to avoid loading dependencies until objectives are actually needed.

Usage:
    from Objectives.registry import get_objective

    # Create an objective instance (lazy loads the class)
    objective = get_objective(FitnessObjective.UTMOS, config, model_data, device, embedding_data)
"""

from typing import Type, Optional
from Datastructures.enum import FitnessObjective
from Datastructures.dataclass import ModelData, EmbeddingData
from Objectives.base import BaseObjective


def _get_objective_class(objective_enum: FitnessObjective) -> Type[BaseObjective]:
    """
    Lazily import and return the objective class for the given enum.
    This avoids loading all dependencies at module import time.
    """

    # Naturalness objectives
    if objective_enum == FitnessObjective.PHONEME_COUNT:
        from Objectives.Naturalness.PhonemeCountObjective import PhonemeCountObjective
        return PhonemeCountObjective

    elif objective_enum == FitnessObjective.UTMOS:
        from Objectives.Naturalness.UtmosObjective import UtmosObjective
        return UtmosObjective

    elif objective_enum == FitnessObjective.PPL:
        from Objectives.Naturalness.PPLObjective import PPLObjective
        return PPLObjective

    elif objective_enum == FitnessObjective.PESQ:
        from Objectives.Naturalness.PESQObjective import PesqObjective
        return PesqObjective

    # InterpolationVector objectives
    elif objective_enum == FitnessObjective.L1:
        from Objectives.InterpolationVector.L1Objective import L1Objective
        return L1Objective

    elif objective_enum == FitnessObjective.L2:
        from Objectives.InterpolationVector.L2Objective import L2Objective
        return L2Objective

    # Target objectives
    elif objective_enum == FitnessObjective.WER_TARGET:
        from Objectives.Target.WerTargetObjective import WerTargetObjective
        return WerTargetObjective

    elif objective_enum == FitnessObjective.SBERT_TARGET:
        from Objectives.Target.SbertTargetObjective import SbertTargetObjective
        return SbertTargetObjective

    elif objective_enum == FitnessObjective.TEXT_EMB_TARGET:
        from Objectives.Target.TextEmbTargetObjective import TextEmbTargetObjective
        return TextEmbTargetObjective

    elif objective_enum == FitnessObjective.WHISPER_PROB:
        from Objectives.Target.WhisperProbObjective import WhisperProbObjective
        return WhisperProbObjective

    elif objective_enum == FitnessObjective.WAV2VEC_DIFFERENT:
        from Objectives.Target.Wav2VecDifferentObjective import Wav2VecDifferentObjective
        return Wav2VecDifferentObjective

    elif objective_enum == FitnessObjective.WAV2VEC_ASR:
        from Objectives.Target.Wav2VecAsrObjective import Wav2VecAsrObjective
        return Wav2VecAsrObjective

    # GroundTruth objectives
    elif objective_enum == FitnessObjective.WER_GT:
        from Objectives.GroundTruth.WerGtObjective import WerGtObjective
        return WerGtObjective

    elif objective_enum == FitnessObjective.SBERT_GT:
        from Objectives.GroundTruth.SbertGtObjective import SbertGtObjective
        return SbertGtObjective

    elif objective_enum == FitnessObjective.TEXT_EMB_GT:
        from Objectives.GroundTruth.TextEmbGtObjective import TextEmbGtObjective
        return TextEmbGtObjective

    elif objective_enum == FitnessObjective.WAV2VEC_SIMILAR:
        from Objectives.GroundTruth.Wav2VecSimilarObjective import Wav2VecSimilarObjective
        return Wav2VecSimilarObjective

    else:
        raise ValueError(f"Unknown objective: {objective_enum}")

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
    objective_cls = _get_objective_class(objective_enum)

    # Check if the objective accepts embedding_data parameter
    import inspect
    sig = inspect.signature(objective_cls.__init__)
    params = sig.parameters

    if 'embedding_data' in params:
        return objective_cls(config, model_data, device=device, embedding_data=embedding_data)
    else:
        return objective_cls(config, model_data, device=device)


# For backwards compatibility, provide a way to get all registered objectives
def get_all_objective_enums() -> list[FitnessObjective]:
    """Returns a list of all supported FitnessObjective enums."""
    return [
        # Naturalness
        FitnessObjective.PHONEME_COUNT,
        FitnessObjective.UTMOS,
        FitnessObjective.PPL,
        FitnessObjective.PESQ,
        # InterpolationVector
        FitnessObjective.L1,
        FitnessObjective.L2,
        # Target
        FitnessObjective.WER_TARGET,
        FitnessObjective.SBERT_TARGET,
        FitnessObjective.TEXT_EMB_TARGET,
        FitnessObjective.WHISPER_PROB,
        FitnessObjective.WAV2VEC_DIFFERENT,
        FitnessObjective.WAV2VEC_ASR,
        # GroundTruth
        FitnessObjective.WER_GT,
        FitnessObjective.SBERT_GT,
        FitnessObjective.TEXT_EMB_GT,
        FitnessObjective.WAV2VEC_SIMILAR,
    ]
