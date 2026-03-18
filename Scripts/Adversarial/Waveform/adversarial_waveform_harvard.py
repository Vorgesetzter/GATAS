"""
Adversarial Waveform — Harvard Sentences Experiment (Vertex AI Custom Job entry point)

Waveform-space NSGA-II baseline: perturbs the raw audio waveform with additive noise
instead of manipulating TTS embeddings.

Usage:
    python Scripts/adversarial_waveform_harvard.py [args]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import signal
import argparse
import datetime
import numpy as np
import torch
import nltk

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from google.cloud import storage

from Datastructures.dataclass import ModelData
from Trainer.EnvironmentLoader import EnvironmentLoader
from Trainer.WaveformAdversarialTrainer import WaveformAdversarialTrainer
from Trainer.RunLogger import RunLogger
from Optimizer.pymoo_optimizer import PymooOptimizer
from pymoo.algorithms.moo.nsga2 import NSGA2


from Datastructures.harvard_sentences import HARVARD_SENTENCES


def upload_folder_to_gcs(local_folder: str, bucket_name: str, gcs_prefix: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            gcs_path = os.path.join(gcs_prefix, os.path.relpath(local_path))
            bucket.blob(gcs_path).upload_from_filename(local_path)
    print(f"[GCS] Uploaded {local_folder} → gs://{bucket_name}/{gcs_prefix}/")


def initialize_parser():
    parser = argparse.ArgumentParser(description="Adversarial TTS — Harvard Sentences")
    parser.add_argument("--sentence_start", type=int, default=1, help="Start position (1-indexed) in the sentence list")
    parser.add_argument("--sentence_end", type=int, default=5, help="End position (1-indexed) in the sentence list")
    parser.add_argument("--loop_count", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=100)
    parser.add_argument("--pop_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--noise_scale", type=float, default=0.05)
    parser.add_argument("--mode", type=str, default="TARGETED")
    parser.add_argument("--target_text", type=str, default="")
    parser.add_argument("--objectives", type=str, default="PESQ=0.2, SET_OVERLAP=0.5")
    parser.add_argument("--seed_target", action="store_true", default=False)
    parser.add_argument("--seed_gt", action="store_true", default=False)
    parser.add_argument("--min_generations", type=int, default=0)
    parser.add_argument("--save_spectrograms", action="store_true", default=True)
    parser.add_argument("--save_graphs", action="store_true", default=True)
    parser.add_argument("--gcs_bucket", type=str, default="thesis-data-2026")
    parser.add_argument("--gcs_prefix", type=str, default="outputs")
    parser.add_argument("--upload_gcs", action="store_true", default=False)
    return parser


def main():
    parser = initialize_parser()
    args = parser.parse_args()

    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _handle_sigterm)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | GPUs: {torch.cuda.device_count()}")

    loader = EnvironmentLoader(device)
    tts_model, asr_model = loader.load_required_models()
    print("Models loaded.")

    sentence_ids = list(range(args.sentence_start, args.sentence_end + 1))

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Run timestamp: {run_timestamp}")
    print(f"{'='*60}")
    print(f"  mode:               {args.mode}")
    print(f"  sentence positions: {args.sentence_start} → {args.sentence_end}  →  IDs: {sentence_ids}")
    print(f"  runs per sentence:  {args.loop_count}")
    print(f"  generations:        {args.num_generations}")
    print(f"  pop_size:           {args.pop_size}")
    print(f"  batch_size:         {args.batch_size}")
    print(f"  objectives:         {args.objectives}")
    print(f"  noise_scale:        {args.noise_scale}")
    print(f"  seed_target:        {args.seed_target}")
    print(f"  min_generations:    {args.min_generations}")
    print(f"{'='*60}")

    all_summaries = []

    try:
        for sentence_id in sentence_ids:
            sentence_text = HARVARD_SENTENCES[sentence_id - 1]
            target_text = args.target_text if args.target_text else HARVARD_SENTENCES[sentence_id % len(HARVARD_SENTENCES)]
            print(f"\n{'='*60}")
            print(f"[Sentence {sentence_id}] {sentence_text}")
            if args.mode == "TARGETED":
                print(f"[Target]     {target_text}")
            print(f"{'='*60}")

            for run_id in range(args.loop_count):
                print(f"\n  [Run {run_id + 1}/{args.loop_count}]")

                run_args = argparse.Namespace(
                    ground_truth_text=sentence_text,
                    target_text=target_text,
                    loop_count=1,
                    num_generations=args.num_generations,
                    pop_size=args.pop_size,
                    iv_scalar=0.0,
                    size_per_phoneme=1,
                    num_rms_candidates=1,
                    batch_size=args.batch_size,
                    notify=False,
                    subspace_optimization=False,
                    mode=args.mode,
                    objectives=args.objectives,
                )
                config_data = loader.load_configuration(run_args)

                audio_gt, audio_target, audio_embedding_gt, audio_embedding_target, gt_rms, target_rms = loader.generate_audio_data(
                    config_data.mode, config_data.text_gt, config_data.text_target, tts_model
                )

                objectives_dict = loader.initialize_objectives(
                    active_objectives=config_data.active_objectives,
                    model_data=ModelData(tts_model=tts_model, asr_model=asr_model),
                    text_gt=config_data.text_gt,
                    text_target=config_data.text_target,
                    mode=config_data.mode,
                    audio_gt=audio_gt,
                )

                trainer = WaveformAdversarialTrainer(
                    tts_model, asr_model, config_data.thresholds, objectives_dict, audio_gt, device,
                    mode=config_data.mode, target_audio=audio_target,
                )
                target_rms = trainer.target_audio.pow(2).mean().sqrt().item()
                logger = RunLogger(
                    config_data.active_objectives, tts_model, asr_model, None, device
                )

                waveform_bounds = (0, 1)
                optimizer = PymooOptimizer(
                    bounds=waveform_bounds,
                    algorithm=NSGA2,
                    algo_params={"pop_size": args.pop_size},
                    num_objectives=len(config_data.active_objectives),
                    solution_shape=(
                        audio_gt.shape[-1],
                    ),
                )

                if args.seed_target or args.seed_gt:
                    n_var = audio_gt.shape[-1]
                    initial_pop = np.random.uniform(waveform_bounds[0], waveform_bounds[1], (args.pop_size, n_var))
                    if args.seed_target:
                        initial_pop[1] = waveform_bounds[1]  # Anchor: pure target → SET_OVERLAP = 0
                    if args.seed_gt:
                        initial_pop[0] = waveform_bounds[0]  # Anchor: pure GT → PESQ ≈ 0
                    optimizer.update_problem((n_var,), sampling=initial_pop)

                fitness_data, archive_data, generation_count, elapsed_time_total, interrupted, generation_found = trainer.run_full_iteration(optimizer, args.num_generations, args.pop_size, args.batch_size, min_generations=args.min_generations)

                if fitness_data:
                    folder_path = logger.setup_multi_sentence_directory(sentence_id, run_id, run_timestamp)
                    summary = logger.save_results_run(
                        optimizer=optimizer,
                        fitness_data=fitness_data,
                        archive_data=archive_data,
                        generation_count=generation_count,
                        elapsed_time_total=elapsed_time_total,
                        audio_gt=audio_gt,
                        audio_target=audio_target,
                        config_data=config_data,
                        folder_path=folder_path,
                        sentence_id=sentence_id,
                        run_id=run_id,
                        run_timestamp=run_timestamp,
                        num_generations=args.num_generations,
                        generation_found=generation_found,
                        save_spectrograms=args.save_spectrograms,
                        save_graphs=args.save_graphs,
                        seed_target=args.seed_target,
                        seed_gt=args.seed_gt,
                        min_generations=args.min_generations,
                        gt_rms=gt_rms,
                        target_rms=target_rms,
                    )
                    all_summaries.append(summary)
                    if args.upload_gcs:
                        upload_folder_to_gcs(folder_path, args.gcs_bucket, args.gcs_prefix)

                torch.cuda.empty_cache()

                if interrupted:
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n[!] Experiment stopped early. All completed runs have been saved.")

    finally:
        RunLogger.aggregate_results(all_summaries, output_dir=os.path.join("outputs", "results", run_timestamp))
        if args.upload_gcs:
            upload_folder_to_gcs("outputs", args.gcs_bucket, args.gcs_prefix)
        print("\n[Done]")


if __name__ == "__main__":
    main()
