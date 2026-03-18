"""
Generate reference audio files for Harvard sentences using StyleTTS2.
Outputs are saved at 16 kHz (resampled from StyleTTS2's native 24 kHz).

Run from project root in the styletts2 conda environment:
    python Scripts/Adversarial/generate_harvard_audios.py --start 1 --end 100
    python Scripts/Adversarial/generate_harvard_audios.py --start 1 --end 100 --output_dir /custom/path
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import librosa
import soundfile as sf

from Datastructures.harvard_sentences import HARVARD_SENTENCES
from Models.styletts2 import StyleTTS2
import torch


DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'HarvardAudios'
)


def main():
    parser = argparse.ArgumentParser(description='Generate Harvard sentence reference audios via StyleTTS2')
    parser.add_argument('--start', type=int, default=1, help='First sentence index (1-based)')
    parser.add_argument('--end', type=int, default=100, help='Last sentence index (1-based, inclusive)')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save generated audio files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Generating audios for sentences {args.start} → {args.end}")
    print(f"Output directory: {args.output_dir}")
    print('=' * 60)

    print("Loading StyleTTS2...")
    tts_model = StyleTTS2(device=device)
    print("StyleTTS2 loaded.\n")

    for sentence_id in range(args.start, args.end + 1):
        sentence_text = HARVARD_SENTENCES[sentence_id - 1]
        output_path = os.path.join(args.output_dir, f'harvard_audio_{sentence_id}.wav')

        embeddings_path = os.path.join(args.output_dir, f'harvard_audio_{sentence_id}.pt')

        if os.path.exists(output_path) and os.path.exists(embeddings_path):
            print(f"[{sentence_id:3d}] Already exists, skipping: {output_path}")
            continue

        noise = torch.randn(1, 1, 256).to(device)
        embeddings = tts_model.extract_embeddings(tts_model.preprocess_text(sentence_text), noise)
        audio_tensor = tts_model.inference_on_embedding(embeddings).flatten()
        audio_numpy = audio_tensor.cpu().detach().numpy()

        audio_16k = librosa.resample(audio_numpy, orig_sr=24000, target_sr=16000)
        sf.write(output_path, audio_16k, 16000)
        torch.save(embeddings, embeddings_path)

        print(f"[{sentence_id:3d}] Saved: {output_path}  |  {sentence_text}")

    print("\n[Done]")


if __name__ == '__main__':
    main()
