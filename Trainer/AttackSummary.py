"""
Standalone utility: compute a standardised attack-result JSON from an adversarial audio file.

Works across all conda environments (styletts2, smack, whisper_attack).
Required packages: pesq, sentence-transformers, torchaudio (>=0.12), openai-whisper.

Usage example:
    from Trainer.AttackSummary import compute_attack_summary

    summary = compute_attack_summary(
        adversarial_audio_path="outputs/results/SMACK/.../best_smack.wav",
        gt_audio_path="Scripts/Adversarial/HarvardAudios/harvard_audio_1.wav",
        gt_text="The birch canoe slid on the smooth planks.",
        attack_method="SMACK",
        num_generations=82,
        pop_size=163,
        elapsed_time_seconds=412.3,
        output_path="outputs/results/SMACK/.../smack_summary.json",
        sentence_id=1,
        asr_model=whisper_instance,   # Models.whisper.Whisper; loaded if None
    )
"""

import os
import re
import json
import datetime

import torch
import soundfile as sf
import torchaudio.functional as AF
import whisper as _whisper_lib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pesq import pesq as _pesq
from sentence_transformers import SentenceTransformer, util as st_util

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

_STOPWORDS  = set(stopwords.words('english'))
_LEMMATIZER = WordNetLemmatizer()
_SBERT_MODEL = 'all-MiniLM-L6-v2'


# ---------------------------------------------------------------------------
# Internal helpers (mirrors RunLogger / SetOverlapObjective logic exactly)
# ---------------------------------------------------------------------------

def _lemmatize_word(word: str) -> str:
    for pos in ('a', 'v', 'n', 'r'):
        lemma = _LEMMATIZER.lemmatize(word, pos=pos)
        if lemma != word:
            return lemma
    return word


def _compute_set_overlap(gt_text: str, asr_text: str) -> float:
    """Fraction of GT content-word lemmas surviving in ASR output. 0.0=success, 1.0=fail."""
    clean_gt = re.sub(r'[^\w\s]', '', gt_text.lower())
    gt_words  = {_lemmatize_word(w) for w in set(clean_gt.split()) - _STOPWORDS}
    if not gt_words:
        return 0.0
    clean_asr = re.sub(r'[^\w\s]', '', (asr_text or '').lower())
    asr_words = {_lemmatize_word(w) for w in set(clean_asr.split()) - _STOPWORDS}
    return round(min(len(gt_words & asr_words) / len(gt_words), 1.0), 6)


def _load_16k(path: str) -> torch.Tensor:
    """Load a WAV file and return a [1, T] float32 tensor resampled to 16 kHz."""
    arr, sr = sf.read(path, dtype='float32')
    t = torch.from_numpy(arr).float()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if sr != 16000:
        t = AF.resample(t, sr, 16000)
    return t  # [1, T] @ 16 kHz


def _transcribe_16k(raw_whisper_model, audio_16k: torch.Tensor, device: str) -> str:
    """Transcribe a [1, T] 16 kHz tensor using the raw openai-whisper model."""
    import string
    arr = _whisper_lib.pad_or_trim(audio_16k.squeeze().numpy())
    mel = _whisper_lib.log_mel_spectrogram(
        torch.from_numpy(arr).to(device),
        n_mels=raw_whisper_model.dims.n_mels,
    ).unsqueeze(0)
    opts = _whisper_lib.DecodingOptions(without_timestamps=True, temperature=0)
    results = _whisper_lib.decode(raw_whisper_model, mel, opts)
    if not isinstance(results, list):
        results = [results]
    text = results[0].text if results else ''
    return text.translate(str.maketrans('', '', string.punctuation)).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_attack_summary(
    adversarial_audio_path: str,
    gt_audio_path: str,
    gt_text: str,
    # Attack metadata
    attack_method: str,
    num_generations: int,
    pop_size: int,
    elapsed_time_seconds: float,
    # Optional context
    output_path: str = None,
    sentence_id: int = None,
    run_id: int = None,
    device: str = None,
    # Pre-computed transcriptions — provide these to skip Whisper re-inference.
    # IMPORTANT: for TTS/Waveform, always pass these from the optimization-time
    # values (text_best / gt_asr_text), because cuBLAS non-determinism means
    # re-running Whisper on the saved WAV may produce a different transcription.
    whisper_transcription: str = None,
    gt_transcription: str = None,
    # Pre-loaded models — pass to avoid reloading between sentences
    # asr_model: Models.whisper.Whisper instance (uses .model internally)
    asr_model=None,
    sbert_model=None,
    squim_model=None,
    # Any extra method-specific fields stored verbatim under "extra"
    extra: dict = None,
) -> dict:
    """
    Compute a standardised attack-result JSON from two WAV files and run metadata.

    Args:
        adversarial_audio_path: Path to the adversarial output WAV.
        gt_audio_path:          Path to the ground truth WAV (PESQ / UTMOS reference).
        gt_text:                Original sentence text (SetOverlap + SBERT anchor).
        attack_method:          "TTS", "Waveform", "SMACK", or "PGD".
        num_generations:        Number of optimizer generations executed.
        pop_size:               Population size.
        elapsed_time_seconds:   Total wall-clock time of the attack in seconds.
        output_path:            If given, save the JSON here.
        sentence_id:            Sentence index (1-based).
        run_id:                 Run index within the sentence.
        device:                 Torch device; auto-detected if None.
        whisper_transcription:  Pre-computed Whisper transcription of the adversarial
                                audio. If provided, Whisper is NOT re-run on the file.
                                For TTS/Waveform this MUST be the value stored during
                                optimization to avoid cuBLAS non-determinism issues.
        gt_transcription:       Pre-computed Whisper transcription of the GT audio.
                                Same caveat applies.
        asr_model:              Models.whisper.Whisper instance; its .model is used
                                for transcription. A new Whisper(tiny) is loaded if
                                neither this nor whisper_transcription is provided.
        sbert_model:            Pre-loaded SentenceTransformer; loaded if None.
        squim_model:            Pre-loaded torchaudio SQUIM_SUBJECTIVE; loaded if None.
        extra:                  Additional method-specific fields stored under "extra".

    Returns:
        dict with the standardised summary structure.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Load and resample audio to 16 kHz ---
    adv_16k = _load_16k(adversarial_audio_path)
    gt_16k  = _load_16k(gt_audio_path)

    # --- Whisper transcriptions ---
    # Use pre-computed values when provided (required for TTS/Waveform to avoid
    # cuBLAS non-determinism: the same audio can transcribe differently depending
    # on the CUDA context / batch size it was first seen in).
    if whisper_transcription is None or gt_transcription is None:
        if asr_model is None:
            from Models.whisper import Whisper
            asr_model = Whisper(device=device)
        raw_whisper = asr_model.model

        if whisper_transcription is None:
            whisper_transcription = _transcribe_16k(raw_whisper, adv_16k, device)
        if gt_transcription is None:
            gt_transcription = _transcribe_16k(raw_whisper, gt_16k, device)

    # --- PESQ (wideband, 16 kHz) ---
    # Raw range:     -0.5 (worst) → 4.5 (perfect)
    # Fitness range:  1.0 (worst) → 0.0 (perfect)  — consistent with PESQObjective
    try:
        pesq_raw = float(_pesq(16000, gt_16k.squeeze().numpy(), adv_16k.squeeze().numpy(), 'wb'))
    except Exception:
        pesq_raw = -0.5
    pesq_fitness = round(max(0.0, min(1.0, 1.0 - (pesq_raw + 0.5) / 5.0)), 6)

    # --- UTMOS (torchaudio SQUIM_SUBJECTIVE, predicts MOS [1, 5]) ---
    if squim_model is None:
        from torchaudio.pipelines import SQUIM_SUBJECTIVE
        squim_model = SQUIM_SUBJECTIVE.get_model().to(device)
        squim_model.eval()

    with torch.no_grad():
        utmos_adv = round(float(squim_model(adv_16k.to(device), gt_16k.to(device))[0].item()), 4)
        utmos_gt  = round(float(squim_model(gt_16k.to(device),  gt_16k.to(device))[0].item()), 4)

    # --- SetOverlap (mirrors SetOverlapObjective exactly) ---
    set_overlap = _compute_set_overlap(gt_text, whisper_transcription)

    # --- SBERT semantic similarity ---
    if sbert_model is None:
        sbert_model = SentenceTransformer(_SBERT_MODEL, device=device)

    embs = sbert_model.encode([gt_text, whisper_transcription], convert_to_tensor=True)
    sbert_similarity = round(float(st_util.cos_sim(embs[0], embs[1]).item()), 6)

    # --- Assemble ---
    avg_time_per_gen = (
        round(elapsed_time_seconds / num_generations, 2) if num_generations > 0 else None
    )

    summary = {
        'metadata': {
            'attack_method':  attack_method,
            'sentence_id':    sentence_id,
            'run_id':         run_id,
            'timestamp':      datetime.datetime.now().isoformat(),
        },
        'text_data': {
            'ground_truth_text':     gt_text,
            'gt_transcription':      gt_transcription,
            'whisper_transcription': whisper_transcription,
        },
        'metrics': {
            'pesq_raw':          round(pesq_raw, 4),  # -0.5 (bad) → 4.5 (perfect)
            'pesq_fitness':      pesq_fitness,         #  1.0 (bad) → 0.0 (perfect)
            'utmos_adversarial': utmos_adv,            # predicted MOS [1, 5]
            'utmos_gt':          utmos_gt,
            'set_overlap':       set_overlap,          #  0.0 = success, 1.0 = fail
            'sbert_similarity':  sbert_similarity,     #  0.0 = different, 1.0 = identical
        },
        'efficiency': {
            'num_generations':         num_generations,
            'pop_size':                pop_size,
            'elapsed_time_seconds':    round(elapsed_time_seconds, 2),
            'avg_time_per_generation': avg_time_per_gen,
        },
    }

    if extra:
        summary['extra'] = extra

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

    return summary
