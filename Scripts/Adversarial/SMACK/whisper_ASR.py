import sys
import torch
from pathlib import Path

# Add project root for Whisper import
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Models.whisper import load_whisper_model


def whisper_ASR(audio_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_whisper_model("base", device=device)

    result = model.transcribe(audio_file, language="en")
    text = result.get("text", "").strip().upper()

    # Normalize: keep only alphanumeric and spaces
    text = "".join(c for c in text if c.isalnum() or c.isspace())

    return text if text else "NA"


# For testing purposes
if __name__ == "__main__":
    audio_file = sys.argv[1]
    result = whisper_ASR(audio_file)
    print(f'Whisper ASR Result: {result}')
