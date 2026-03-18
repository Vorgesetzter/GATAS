import whisper
import torch
import string
import torchaudio.functional as torchaudio_functional


def load_whisper_model(model_name="tiny", device=None):
    """Load a Whisper model for SMACK framework."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return Whisper(device=device)


class Whisper:
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = whisper.load_model("tiny", device=self.device)

    def inference(self, audio_batch):
        # Force deterministic cuBLAS/cuDNN kernels for the entire ASR pipeline so that
        # results are bit-identical regardless of batch size (optimization batch vs
        # batch_size=1 at final inference). Without this, borderline adversarial audio
        # can produce different transcriptions depending on the CUDA tiling strategy.
        prev_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except TypeError:
            torch.use_deterministic_algorithms(True)

        try:
            # 2. Prepare audio tensors — already at 16 kHz (StyleTTS2 resamples internally)
            audio_tensor_asr = whisper.pad_or_trim(audio_batch)

            # 3. Create Mel spectrogram
            mel_batch = whisper.log_mel_spectrogram(audio_tensor_asr, n_mels=self.model.dims.n_mels).to(self.device)

            # 4. Run ASR decoding (without_timestamps reduces hallucination on padded silence)
            # temperature=0 forces greedy decoding for deterministic results
            # Note: fallback thresholds only apply to whisper.transcribe(), not whisper.decode()
            decode_options = whisper.DecodingOptions(without_timestamps=True, temperature=0)
            results = whisper.decode(self.model, mel_batch, decode_options)

            # 5. Process ASR results (handle single vs batch)
            if not isinstance(results, list):
                results = [results]
            asr_texts = [r.text for r in results]
            # clean_texts = [re.sub(r'[^a-zA-Z\s]', '', t).strip() for t in asr_texts]
            clean_texts = [t.translate(str.maketrans('', '', string.punctuation)).strip() for t in asr_texts]

        finally:
            torch.use_deterministic_algorithms(prev_deterministic)

        return clean_texts, mel_batch
