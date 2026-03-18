"""
Generate reference audio files for Harvard sentences using StyleTTS2.
Outputs are saved at 16 kHz (resampled from StyleTTS2's native 24 kHz)
for use by the SMACK attack pipeline.

Run from project root in the styletts2 conda environment:
    python Scripts/Adversarial/SMACK/generate_harvard_audios.py --start 1 --end 100
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import argparse
import torch
import librosa
import soundfile as sf

from Datastructures.harvard_sentences import HARVARD_SENTENCES
from Models.styletts2 import StyleTTS2


OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'HarvardAudios'
)


def main():
    parser = argparse.ArgumentParser(description='Generate Harvard sentence reference audios via StyleTTS2')
    parser.add_argument('--start', type=int, default=1, help='First sentence index (1-based)')
    parser.add_argument('--end', type=int, default=100, help='Last sentence index (1-based, inclusive)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Generating audios for sentences {args.start} → {args.end}")
    print(f"Output directory: {OUTPUT_DIR}")
    print('=' * 60)

    print("Loading StyleTTS2...")
    tts_model = StyleTTS2(device=device)
    print("StyleTTS2 loaded.\n")

    for sentence_id in range(args.start, args.end + 1):
        sentence_text = HARVARD_SENTENCES[sentence_id - 1]
        output_path = os.path.join(OUTPUT_DIR, f'harvard_audio_{sentence_id}.wav')

        if os.path.exists(output_path):
            print(f"[{sentence_id:3d}] Already exists, skipping: {output_path}")
            continue

        noise = torch.randn(1, 1, 256).to(device)
        audio_tensor = tts_model.inference(sentence_text, noise).flatten()
        audio_numpy = audio_tensor.cpu().detach().numpy()

        # Resample from StyleTTS2's 24 kHz to 16 kHz for SMACK
        audio_16k = librosa.resample(audio_numpy, orig_sr=24000, target_sr=16000)
        sf.write(output_path, audio_16k, 16000)

        print(f"[{sentence_id:3d}] Saved: {output_path}  |  {sentence_text}")

    print("\n[Done]")


if __name__ == '__main__':
    main()
