"""
Adversarial PGD — Harvard Sentences Experiment

Loads pre-generated reference audios and runs the PGD untargeted attack
(from whisper_attack) on each Harvard sentence.

Generate reference audios first (styletts2 env):
    python Scripts/Adversarial/generate_harvard_audios.py --start 1 --end 100

Then run this script (whisper_attack env) from project root:
    python Scripts/Adversarial/PGD/adversarial_pgd_harvard.py --start 1 --end 100
"""

import os
import sys
import json
import shutil
import argparse
import datetime
import subprocess
import soundfile as sf
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Datastructures.harvard_sentences import HARVARD_SENTENCES


NB_ITER = 200
SEED = 235
SNR = 35
MODEL_LABEL = 'base'

AUDIO_DIR = str(project_root / 'Scripts' / 'Adversarial' / 'HarvardAudios')
WHISPER_ATTACK_DIR = str(project_root / 'whisper_attack')


def create_csv(sentence_ids, output_dir):
    csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, 'harvard.csv')

    with open(csv_path, 'w') as f:
        f.write('ID,duration,wav,wrd\n')
        for sid in sentence_ids:
            audio_path = os.path.join(AUDIO_DIR, f'harvard_audio_{sid}.wav')
            if not os.path.exists(audio_path):
                print(f"[{sid:3d}] Audio not found, skipping: {audio_path}")
                continue
            duration = sf.info(audio_path).duration
            text = HARVARD_SENTENCES[sid - 1].upper()
            f.write(f'sentence_{sid:03d},{duration:.3f},{audio_path},{text}\n')

    return csv_path


def run_pgd_attack(output_dir):
    cmd = [
        sys.executable, 'run_attack.py',
        'attack_configs/whisper/pgd.yaml',
        f'--root={output_dir}',
        f'--data_folder={output_dir}',
        '--data_csv_name=harvard',
        f'--model_label={MODEL_LABEL}',
        f'--nb_iter={NB_ITER}',
        '--load_audio=False',
        f'--seed={SEED}',
        '--attack_name=pgd_harvard',
        f'--snr={SNR}',
        '--skip_prep=True',
    ]
    subprocess.run(cmd, cwd=WHISPER_ATTACK_DIR, check=True)


def organize_outputs(sentence_ids, output_dir, elapsed_time_seconds=None, n_sentences=1):
    save_path = os.path.join(
        output_dir, 'attacks', 'pgd_harvard',
        f'whisper-{MODEL_LABEL}-{SNR}', str(SEED), 'save'
    )
    timestamp = os.path.basename(output_dir)

    for sid in sentence_ids:
        adv_src = os.path.join(save_path, f'sentence_{sid:03d}.wav')
        gt_src = os.path.join(AUDIO_DIR, f'harvard_audio_{sid}.wav')

        if not os.path.exists(adv_src):
            print(f"[{sid:3d}] Adversarial audio not found, skipping")
            continue

        sentence_dir = os.path.join(output_dir, f'sentence_{sid:03d}')
        os.makedirs(sentence_dir, exist_ok=True)

        adv_dst = os.path.join(sentence_dir, 'best_pgd.wav')
        gt_dst  = os.path.join(sentence_dir, 'ground_truth.wav')
        shutil.copy(adv_src, adv_dst)
        shutil.copy(gt_src,  gt_dst)

        sentence_elapsed = elapsed_time_seconds / n_sentences if elapsed_time_seconds else None
        with open(os.path.join(sentence_dir, 'attack_metadata.json'), 'w') as f:
            json.dump({
                'attack_method': 'PGD',
                'sentence_id': sid,
                'gt_text': HARVARD_SENTENCES[sid - 1],
                'adversarial_audio': 'best_pgd.wav',
                'num_generations': NB_ITER,
                'pop_size': 1,
                'elapsed_time_seconds': round(sentence_elapsed or 0.0, 2),
                'extra': {
                    'model': f'whisper-{MODEL_LABEL}',
                    'snr': SNR,
                    'seed': SEED,
                    'timestamp': timestamp,
                },
            }, f, indent=2)

        print(f"[{sid:3d}] Saved to {sentence_dir}")


def main():
    parser = argparse.ArgumentParser(description='PGD Untargeted Attack — Harvard Sentences')
    parser.add_argument('--start', type=int, default=1, help='First sentence index (1-based)')
    parser.add_argument('--end', type=int, default=100, help='Last sentence index (1-based, inclusive)')
    args = parser.parse_args()

    sentence_ids = list(range(args.start, args.end + 1))

    print(f"Sentences: {args.start} → {args.end}")
    print(f"Model: whisper-{MODEL_LABEL} | SNR: {SNR} | Iterations: {NB_ITER} | Seed: {SEED}")
    print('=' * 60)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = str(project_root / 'outputs' / 'results' / 'PGD' / timestamp)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = create_csv(sentence_ids, output_dir)
    print(f"CSV created: {csv_path}\n")

    import time
    t0 = time.time()
    run_pgd_attack(output_dir)
    elapsed = time.time() - t0

    organize_outputs(sentence_ids, output_dir, elapsed_time_seconds=elapsed, n_sentences=len(sentence_ids))

    with open('/tmp/pgd_last_output.txt', 'w') as f:
        f.write(output_dir)

    print('\n[Done]')


if __name__ == '__main__':
    main()
