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


# =============================================================================
# Harvard Sentence Lists (IEEE, Rothauser et al. 1969)
# Source: https://www.cs.columbia.edu/~hgs/audio/harvard.html
# =============================================================================
HARVARD_SENTENCES = [
    # List 1
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "A large size in stockings is hard to sell.",
    # List 2
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.",
    "The girl at the booth sold fifty bonds.",
    # List 3
    "The small pup gnawed a hole in the sock.",
    "The fish twisted and turned on the bent hook.",
    "Press the pants and sew a button on the vest.",
    "The swan dive was far short of perfect.",
    "The beauty of the view stunned the young boy.",
    "Two blue fish swam in the tank.",
    "Her purse was full of useless trash.",
    "The colt reared and threw the tall rider.",
    "It snowed, rained, and hailed the same morning.",
    "Read verse out loud for pleasure.",
    # List 4
    "Hoist the load to your left shoulder.",
    "Take the winding path to reach the lake.",
    "Note closely the size of the gas tank.",
    "Wipe the grease off his dirty face.",
    "Mend the coat before you go out.",
    "The wrist was badly strained and hung limp.",
    "The stray cat gave birth to kittens.",
    "The young girl gave no clear response.",
    "The meal was cooked before the bell rang.",
    "What joy there is in living.",
    # List 5
    "A king ruled the state in the early days.",
    "Ship maps are different from those for planes.",
    "Dimes showered down from all sides.",
    "The two met while playing on the sand.",
    "The ink stain dried on the finished page.",
    "The walled town was seized without a fight.",
    "The lease ran out in sixteen weeks.",
    "A tame squirrel makes a nice pet.",
    "The horn of the car woke the sleeping cop.",
    "The heart beat strongly and with firm strokes.",
    # List 6
    "The pearl was worn in a thin silver ring.",
    "The fruit peel was cut in thick slices.",
    "The Navy attacked the big task force.",
    "See the cat glaring at the scared mouse.",
    "There are more than two factors here.",
    "The hat brim was wide and too droopy.",
    "The lawyer tried to lose his case.",
    "The grass curled around the fence post.",
    "Cut the pie into large parts.",
    "Men strive but seldom get rich.",
    # List 7
    "Always close the barn door tight.",
    "He ran half way to the hardware store.",
    "The clock struck to mark the third period.",
    "A small creek cut across the field.",
    "Cars and busses stalled in snow drifts.",
    "The set of china hit the floor with a crash.",
    "This is a grand season for hikes on the road.",
    "The dune rose from the edge of the water.",
    "Those words were the cue for the actor to leave.",
    "A yacht slid around the point into the bay.",
    # List 8
    "The two boys' voices echoed in the hall.",
    "The gold vase is both rare and costly.",
    "The knife was hung inside its bright sheath.",
    "The rarest spice comes from the Far East.",
    "The roof should be tilted at a sharp slant.",
    "A siege will take a lot out of the troops.",
    "The green moss grew over the stone well.",
    "The shelves were bare of both jam or crackers.",
    "Every word and phrase he speaks is true.",
    "The bombs left most of the town in ruins.",
    # List 9
    "Stop whistling and watch the boys march.",
    "Jerk the rope and the bell rings weakly.",
    "A clean neck means a neat collar.",
    "A new wax protects the deep scratch.",
    "The cup cracked and spilled its contents.",
    "Guess the results from the first scores.",
    "A salt pickle tastes fine with ham.",
    "The just claim got the right verdict.",
    "The purple tie was ten years old.",
    "The tree top waved in a graceful way.",
    # List 10
    "The spot on the blotter was made by green ink.",
    "Mud was spattered on the front of his white shirt.",
    "The cigar burned a hole in the desk top.",
    "The empty flask stood on the tin tray.",
    "A speedy man can beat this track mark.",
    "He broke a new shoelace that day.",
    "The coffee in the mug was too hot to drink.",
    "The birch trees were bare and lonely.",
    "The petals fall with every light breeze.",
    "Bring your problems to the wise chief.",
]


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
    parser.add_argument("--sentence_start", type=int, default=1,
                        help="Start position (1-indexed) in the randomly sampled sentence list")
    parser.add_argument("--sentence_end", type=int, default=5,
                        help="End position (1-indexed) in the randomly sampled sentence list")
    parser.add_argument("--sample_size", type=int, default=20,
                        help="Total number of sentences to sample from 1–100")
    parser.add_argument("--sample_seed", type=int, default=42,
                        help="Random seed for sentence sampling")
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

    sampled_ids = sorted(np.random.default_rng(args.sample_seed).choice(100, size=args.sample_size, replace=False) + 1)
    sentence_ids = [int(i) for i in sampled_ids[args.sentence_start - 1 : args.sentence_end]]

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Run timestamp: {run_timestamp}")
    print(f"{'='*60}")
    print(f"  mode:               {args.mode}")
    print(f"  sample seed:        {args.sample_seed}  (sample size: {args.sample_size})")
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
                        initial_pop[0] = waveform_bounds[1]  # Anchor: pure target → SET_OVERLAP = 0
                    if args.seed_gt:
                        initial_pop[1] = waveform_bounds[0]  # Anchor: pure GT → PESQ ≈ 0
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
                    upload_folder_to_gcs(folder_path, args.gcs_bucket, args.gcs_prefix)

                torch.cuda.empty_cache()

                if interrupted:
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n[!] Experiment stopped early. All completed runs have been saved.")

    finally:
        RunLogger.aggregate_results(all_summaries, output_dir=os.path.join("outputs", "results", run_timestamp))
        upload_folder_to_gcs("outputs", args.gcs_bucket, args.gcs_prefix)
        print("\n[Done]")


if __name__ == "__main__":
    main()
