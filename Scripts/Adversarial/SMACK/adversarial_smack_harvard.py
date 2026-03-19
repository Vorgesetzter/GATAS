"""
Adversarial SMACK — Harvard Sentences Experiment

Loads pre-generated reference audio from HarvardAudios/ and runs the SMACK
untargeted attack (genetic + gradient) on each Harvard sentence.

Generate reference audios first (styletts2 env):
    python Scripts/Adversarial/SMACK/generate_harvard_audios.py --start 1 --end 100

Then run this script (smack env) from the SMACK directory:
    python adversarial_smack_harvard.py --start 1 --end 10
"""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

import sys
import json
import time
import argparse
import datetime
import soundfile as sf
from pathlib import Path

# Add project root for Datastructures, and SMACK dir for genetic/gradient
project_root = Path(__file__).resolve().parent.parent.parent.parent
smack_dir = str(project_root / 'SMACK')
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if smack_dir not in sys.path:
    sys.path.insert(0, smack_dir)

from Datastructures.harvard_sentences import HARVARD_SENTENCES
from genetic import GeneticAlgorithm
from gradient import GradientEstimation
from synthesis import audio_synthesis


POPULATION_SIZE = 163
GENETIC_ITERATIONS = 82
GRADIENT_ITERATIONS = 41
TARGET_MODEL = 'whisperASR'

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'HarvardAudios')


def run_attack(reference_audio: str, reference_text: str, output_dir: str):
    """Run genetic + gradient attack for one sentence."""
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    ga = GeneticAlgorithm(reference_audio, reference_text, TARGET_MODEL, POPULATION_SIZE)
    fittest_individual = ga.run(GENETIC_ITERATIONS)

    print("Genetic algorithm finished. Launching gradient estimation.\n")

    gradient_estimator = GradientEstimation(
        reference_audio, reference_text, TARGET_MODEL,
        sigma=0.1, learning_rate=0.01, K=POPULATION_SIZE
    )
    p_refined = gradient_estimator.refine_prosody_vector(fittest_individual, GRADIENT_ITERATIONS)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"Attack finished. Time: {hours}h {minutes}m {seconds:.2f}s\n")

    return p_refined, elapsed


def main():
    parser = argparse.ArgumentParser(description='SMACK Untargeted Attack — Harvard Sentences')
    parser.add_argument('--start', type=int, default=1,
                        help='First Harvard sentence index (1-based)')
    parser.add_argument('--end', type=int, default=10,
                        help='Last Harvard sentence index (1-based, inclusive)')
    args = parser.parse_args()

    print(f"Sentences: {args.start} → {args.end}")
    print(f"Population: {POPULATION_SIZE} | Genetic: {GENETIC_ITERATIONS} | Gradient: {GRADIENT_ITERATIONS}")
    print(f"Audio directory: {AUDIO_DIR}")
    print('=' * 60)

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_base = os.path.join(
        str(project_root), 'outputs', 'results', 'SMACK', run_timestamp
    )

    for sentence_id in range(args.start, args.end + 1):
        sentence_text = HARVARD_SENTENCES[sentence_id - 1]
        reference_audio = os.path.join(AUDIO_DIR, f'harvard_audio_{sentence_id}.wav')

        if not os.path.exists(reference_audio):
            print(f"[{sentence_id:3d}] Reference audio not found, skipping: {reference_audio}")
            continue

        print(f"\n{'='*60}")
        print(f"[Sentence {sentence_id}] {sentence_text}")
        print('=' * 60)

        sentence_dir = os.path.join(output_base, f'sentence_{sentence_id:03d}')
        os.makedirs(sentence_dir, exist_ok=True)

        p_refined, elapsed = run_attack(reference_audio, sentence_text, sentence_dir)

        # Synthesize adversarial audio from the refined prosody vector
        # ETTS (WaveGlow) outputs at 22050 Hz — resample to 16 kHz for consistency
        import librosa
        audio_numpy = audio_synthesis(p_refined.reshape(-1, 32), reference_audio, sentence_text)
        audio_16k = librosa.resample(audio_numpy.astype('float32'), orig_sr=22050, target_sr=16000)
        adv_path = os.path.join(sentence_dir, 'best_smack.wav')
        gt_dst    = os.path.join(sentence_dir, 'ground_truth.wav')
        sf.write(adv_path, audio_16k, 16000)

        import shutil
        shutil.copy(reference_audio, gt_dst)

        with open(os.path.join(sentence_dir, 'attack_metadata.json'), 'w') as f:
            json.dump({
                'attack_method': 'SMACK',
                'sentence_id': sentence_id,
                'gt_text': sentence_text,
                'adversarial_audio': 'best_smack.wav',
                'num_generations': GENETIC_ITERATIONS + GRADIENT_ITERATIONS,
                'pop_size': POPULATION_SIZE,
                'elapsed_time_seconds': round(elapsed, 2),
            }, f, indent=2)

    with open('/tmp/smack_last_output.txt', 'w') as f:
        f.write(output_base)

    print("\n[Done]")


if __name__ == '__main__':
    main()
