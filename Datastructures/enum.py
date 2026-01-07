from enum import Enum, auto

class AttackMode(Enum):
    TARGETED = "targeted"
    NOISE_UNTARGETED = "noise-untargeted"
    UNTARGETED = "untargeted"

class FitnessObjective(Enum):

    # ==== Increase Naturalness ====
    PHONEME_COUNT = auto()
    UTMOS = auto()
    PPL = auto()
    PESQ = auto()

    # ==== Interpolation Vector Restrictions ====
    L1 = auto()
    L2 = auto()

    # ==== Optimize Text Towards Target ====
    WER_TARGET = auto()
    SBERT_TARGET = auto()
    TEXT_EMB_TARGET = auto()
    WHISPER_PROB = auto()

    # ==== Optimize Text Away From Ground-Truth ====
    WER_GT = auto()
    SBERT_GT = auto()
    TEXT_EMB_GT = auto()

    # ==== Optimize Audio Similarity ====
    WAV2VEC_SIMILAR = auto()
    WAV2VEC_DIFFERENT = auto()
    WAV2VEC_ASR = auto()