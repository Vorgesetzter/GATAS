import sys
import torch
import librosa
from pathlib import Path

# Add project root for Whisper import
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Models.whisper import load_whisper_model


# Load once at module import time — reloading on every call was the bottleneck
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = load_whisper_model("base", device=_device)


def whisper_ASR(audio_file):
    # Load audio at native sample rate
    audio, sr = librosa.load(audio_file, sr=None)
    audio_tensor = torch.from_numpy(audio).float()

    # Call inference method (handles resampling internally)
    clean_texts, _ = _model.inference(audio_tensor)

    text = clean_texts[0].upper() if clean_texts else ""

    # Normalize: keep only alphanumeric and spaces
    text = "".join(c for c in text if c.isalnum() or c.isspace())

    return text if text else "NA"


# For testing purposes
if __name__ == "__main__":
    audio_file = sys.argv[1]
    result = whisper_ASR(audio_file)
    print(f'Whisper ASR Result: {result}')
