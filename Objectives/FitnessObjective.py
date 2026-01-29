from enum import Enum

# Naturalness objectives
from Objectives.Naturalness.UtmosObjective import UtmosObjective
from Objectives.Naturalness.PPLObjective import PPLObjective
from Objectives.Naturalness.PESQObjective import PesqObjective

# InterpolationVector objectives
from Objectives.InterpolationVector.L1Objective import L1Objective
from Objectives.InterpolationVector.L2Objective import L2Objective

# Target objectives
from Objectives.Target.WerTargetObjective import WerTargetObjective
from Objectives.Target.MerTargetObjective import MerTargetObjective
from Objectives.Target.SbertTargetObjective import SbertTargetObjective
from Objectives.Target.TextEmbTargetObjective import TextEmbTargetObjective
from Objectives.Target.WhisperProbObjective import WhisperProbObjective
from Objectives.Target.Wav2VecDifferentObjective import Wav2VecDifferentObjective
from Objectives.Target.Wav2VecAsrObjective import Wav2VecAsrObjective

# GroundTruth objectives
from Objectives.GroundTruth.WerGtObjective import WerGtObjective
from Objectives.GroundTruth.MerGtObjective import MerGtObjective
from Objectives.GroundTruth.SbertGtObjective import SbertGtObjective
from Objectives.GroundTruth.TextEmbGtObjective import TextEmbGtObjective
from Objectives.GroundTruth.Wav2VecSimilarObjective import Wav2VecSimilarObjective
from Objectives.GroundTruth.VocabOverlapObjective import VocabOverlapObjective

class FitnessObjective(Enum):
    # ==== Increase Naturalness ====
    UTMOS = UtmosObjective
    PPL = PPLObjective
    PESQ = PesqObjective

    # ==== Interpolation Vector Restrictions ====
    L1 = L1Objective
    L2 = L2Objective

    # ==== Optimize Text Towards Target ====
    WER_TARGET = WerTargetObjective
    MER_TARGET = MerTargetObjective
    SBERT_TARGET = SbertTargetObjective
    TEXT_EMB_TARGET = TextEmbTargetObjective
    WHISPER_PROB = WhisperProbObjective

    # ==== Optimize Text Away From Ground-Truth ====
    WER_GT = WerGtObjective
    MER_GT = MerGtObjective
    SBERT_GT = SbertGtObjective
    TEXT_EMB_GT = TextEmbGtObjective
    SET_OVERLAP = VocabOverlapObjective

    # ==== Optimize Audio Similarity ====
    WAV2VEC_SIMILAR = Wav2VecSimilarObjective
    WAV2VEC_DIFFERENT = Wav2VecDifferentObjective
    WAV2VEC_ASR = Wav2VecAsrObjective
