"""
Post-hoc attack summary computation (styletts2 environment).

Each attack script (SMACK, PGD) writes an attack_metadata.json per sentence
directory during the attack run. This script scans those directories, loads
models once, and computes the full metric summary (PESQ, UTMOS, SetOverlap,
SBERT, Whisper transcription) for every sentence.

Run from project root in the styletts2 conda environment:
    python Scripts/Adversarial/compute_attack_summary.py --results_dir outputs/results/SMACK/20260319_1200
"""

import os
import sys
import json
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from Trainer.AttackSummary import compute_attack_summary


def main():
    parser = argparse.ArgumentParser(description='Post-hoc attack summary computation')
    parser.add_argument('--results_dir', required=True,
                        help='Path to attack results directory (contains sentence_* subdirs)')
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"[!] Results directory not found: {results_dir}")
        sys.exit(1)

    sentence_dirs = sorted([
        d for d in Path(results_dir).iterdir()
        if d.is_dir() and d.name.startswith('sentence_')
    ])

    if not sentence_dirs:
        print(f"[!] No sentence_* directories found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(sentence_dirs)} sentence directories in {results_dir}")
    print('=' * 60)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from Models.whisper import Whisper
    from sentence_transformers import SentenceTransformer
    from torchaudio.pipelines import SQUIM_SUBJECTIVE

    print("Loading models...")
    asr_model   = Whisper(device=device)
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    squim_model = SQUIM_SUBJECTIVE.get_model().to(device)
    squim_model.eval()
    print("Models loaded.\n")

    for sentence_dir in sentence_dirs:
        metadata_path = sentence_dir / 'attack_metadata.json'
        if not metadata_path.exists():
            print(f"[{sentence_dir.name}] No attack_metadata.json, skipping")
            continue

        with open(metadata_path) as f:
            meta = json.load(f)

        attack_method = meta['attack_method']
        adv_path      = str(sentence_dir / meta['adversarial_audio'])
        gt_path       = str(sentence_dir / 'ground_truth.wav')
        output_path   = str(sentence_dir / f"{attack_method.lower()}_summary.json")

        if not os.path.exists(adv_path):
            print(f"[{sentence_dir.name}] Adversarial audio not found: {adv_path}, skipping")
            continue
        if not os.path.exists(gt_path):
            print(f"[{sentence_dir.name}] Ground truth not found: {gt_path}, skipping")
            continue

        print(f"[{sentence_dir.name}] Computing summary...")
        compute_attack_summary(
            adversarial_audio_path=adv_path,
            gt_audio_path=gt_path,
            gt_text=meta['gt_text'],
            attack_method=attack_method,
            num_generations=meta['num_generations'],
            pop_size=meta['pop_size'],
            elapsed_time_seconds=meta['elapsed_time_seconds'],
            output_path=output_path,
            sentence_id=meta.get('sentence_id'),
            extra=meta.get('extra'),
            asr_model=asr_model,
            sbert_model=sbert_model,
            squim_model=squim_model,
            device=device,
        )
        print(f"[{sentence_dir.name}] Saved: {output_path}")

    print("\n[Done]")


if __name__ == '__main__':
    main()
