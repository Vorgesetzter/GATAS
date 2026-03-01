"""
Adversarial Harvard Sentences — Combined TTS Embedding + Waveform Experiment

For each sentence/run:
  1. Synthesizes audio_gt once via StyleTTS2
  2. Runs the TTS embedding-space NSGA-II method on that audio
  3. Runs the waveform-space NSGA-II baseline on the SAME audio_gt

Both methods therefore start from an identical waveform, ensuring a fair comparison.

Results are saved separately:
  outputs/results/tts/{timestamp}/sentence_XXX/run_N/
  outputs/results/waveform/{timestamp}/sentence_XXX/run_N/

Usage:
    python Scripts/adversarial_harvard.py [args]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import signal
import argparse
import datetime
import torch
import nltk

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from google.cloud import storage

from Datastructures.dataclass import ModelData
from Trainer.EnvironmentLoader import EnvironmentLoader
from Trainer.AdversarialTrainer import AdversarialTrainer
from Trainer.WaveformAdversarialTrainer import WaveformAdversarialTrainer
from Trainer.VectorManipulator import VectorManipulator
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
    parser.add_argument("--harvard_sentences_start", type=int, default=1)
    parser.add_argument("--harvard_sentences_end", type=int, default=10)
    parser.add_argument("--loop_count", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=100)
    parser.add_argument("--pop_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=100)
    # TTS embedding method
    parser.add_argument("--iv_scalar", type=float, default=0.5)
    parser.add_argument("--size_per_phoneme", type=int, default=1)
    parser.add_argument("--subspace_optimization", action="store_true")
    parser.add_argument("--num_rms_candidates", type=int, default=1)
    parser.add_argument("--seed_target", action="store_true", default=False)
    # Waveform method
    parser.add_argument("--noise_scale", type=float, default=0.05)
    parser.add_argument("--mode", type=str, default="TARGETED")
    parser.add_argument("--target_text", type=str, default="")
    parser.add_argument("--objectives", type=str, default="PESQ=0.2, SET_OVERLAP=0.5")
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

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Run timestamp: {run_timestamp}")
    print(f"{'='*60}")
    print(f"  mode:               {args.mode}")
    print(f"  sentences:          {args.harvard_sentences_start} → {args.harvard_sentences_end}")
    print(f"  runs per sentence:  {args.loop_count}")
    print(f"  generations:        {args.num_generations}")
    print(f"  pop_size:           {args.pop_size}")
    print(f"  batch_size:         {args.batch_size}")
    print(f"  objectives:         {args.objectives}")
    print(f"  iv_scalar:          {args.iv_scalar}")
    print(f"  size_per_phoneme:   {args.size_per_phoneme}")
    print(f"  noise_scale:        {args.noise_scale}")
    print(f"{'='*60}")

    tts_summaries = []
    waveform_summaries = []

    try:
        for sentence_id in range(args.harvard_sentences_start, args.harvard_sentences_end + 1):
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
                    iv_scalar=args.iv_scalar,
                    size_per_phoneme=args.size_per_phoneme,
                    batch_size=args.batch_size,
                    notify=False,
                    subspace_optimization=args.subspace_optimization,
                    mode=args.mode,
                    objectives=args.objectives,
                    num_rms_candidates=args.num_rms_candidates,
                )
                config_data = loader.load_configuration(run_args)

                # Synthesize audio_gt ONCE — shared by both methods for identical starting point
                audio_gt, audio_target, audio_embedding_gt, audio_embedding_target = loader.generate_audio_data(
                    config_data.mode, config_data.text_gt, config_data.text_target, tts_model,
                    num_rms_candidates=config_data.num_rms_candidates,
                )

                # ── TTS Embedding Method ──────────────────────────────────
                print(f"\n  [Run {run_id + 1}] TTS Embedding Method")

                tts_objectives = loader.initialize_objectives(
                    active_objectives=config_data.active_objectives,
                    model_data=ModelData(tts_model=tts_model, asr_model=asr_model),
                    text_gt=config_data.text_gt,
                    text_target=config_data.text_target,
                    mode=config_data.mode,
                    audio_gt=audio_gt,
                )

                vector_manipulator = VectorManipulator(
                    audio_embedding_gt=audio_embedding_gt,
                    h_text_target=audio_embedding_target.h_text,
                    config_data=config_data,
                )

                tts_trainer = AdversarialTrainer(
                    tts_model, asr_model, config_data.thresholds, tts_objectives, vector_manipulator, device
                )
                tts_logger = RunLogger(
                    config_data.active_objectives, tts_model, asr_model, vector_manipulator, device
                )
                tts_solution_shape = (
                    int(audio_embedding_gt.input_length.detach().cpu().item()), args.size_per_phoneme,
                )
                tts_optimizer = PymooOptimizer(
                    bounds=(0, 1),
                    algorithm=NSGA2,
                    algo_params={"pop_size": args.pop_size},
                    num_objectives=len(config_data.active_objectives),
                    solution_shape=tts_solution_shape,
                )

                if args.seed_target:
                    import numpy as np
                    n_var = int(np.prod(tts_solution_shape))
                    initial_pop = np.random.uniform(0, 1, (args.pop_size, n_var))
                    initial_pop[0] = 1.0  # Anchor: pure noise target → SET_OVERLAP = 0
                    tts_optimizer.update_problem(tts_solution_shape, sampling=initial_pop)

                fitness_data, archive_data, generation_count, elapsed_time_total, interrupted = \
                    tts_trainer.run_full_iteration(tts_optimizer, args.num_generations, args.pop_size, args.batch_size)

                if fitness_data:
                    folder_path = tts_logger.setup_multi_sentence_directory(
                        sentence_id, run_id, run_timestamp, base_path="outputs/results/tts"
                    )
                    summary = tts_logger.save_results_run(
                        optimizer=tts_optimizer,
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
                        save_spectrograms=args.save_spectrograms,
                        save_graphs=args.save_graphs,
                    )
                    tts_summaries.append(summary)
                    upload_folder_to_gcs(folder_path, args.gcs_bucket, args.gcs_prefix)

                torch.cuda.empty_cache()

                if interrupted:
                    raise KeyboardInterrupt

                # ── Waveform Method ───────────────────────────────────────
                print(f"\n  [Run {run_id + 1}] Waveform Method")

                waveform_objectives = loader.initialize_objectives(
                    active_objectives=config_data.active_objectives,
                    model_data=ModelData(tts_model=tts_model, asr_model=asr_model),
                    text_gt=config_data.text_gt,
                    text_target=config_data.text_target,
                    mode=config_data.mode,
                    audio_gt=audio_gt,
                )

                waveform_trainer = WaveformAdversarialTrainer(
                    tts_model, asr_model, config_data.thresholds, waveform_objectives, audio_gt, device,
                    mode=config_data.mode, target_audio=audio_target,
                )
                waveform_logger = RunLogger(
                    config_data.active_objectives, tts_model, asr_model, None, device
                )
                waveform_bounds = (0, 1) if args.mode == "TARGETED" else (-args.noise_scale, args.noise_scale)
                waveform_optimizer = PymooOptimizer(
                    bounds=waveform_bounds,
                    algorithm=NSGA2,
                    algo_params={"pop_size": args.pop_size},
                    num_objectives=len(config_data.active_objectives),
                    solution_shape=(audio_gt.shape[-1],),
                )

                fitness_data, archive_data, generation_count, elapsed_time_total, interrupted = \
                    waveform_trainer.run_full_iteration(waveform_optimizer, args.num_generations, args.pop_size, args.batch_size)

                if fitness_data:
                    folder_path = waveform_logger.setup_multi_sentence_directory(
                        sentence_id, run_id, run_timestamp, base_path="outputs/results/waveform"
                    )
                    summary = waveform_logger.save_results_run(
                        optimizer=waveform_optimizer,
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
                        save_spectrograms=args.save_spectrograms,
                        save_graphs=args.save_graphs,
                    )
                    waveform_summaries.append(summary)
                    upload_folder_to_gcs(folder_path, args.gcs_bucket, args.gcs_prefix)

                torch.cuda.empty_cache()

                if interrupted:
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n[!] Experiment stopped early. All completed runs have been saved.")

    finally:
        RunLogger.aggregate_results(tts_summaries, output_dir=os.path.join("outputs", "results", "tts", run_timestamp))
        RunLogger.aggregate_results(waveform_summaries, output_dir=os.path.join("outputs", "results", "waveform", run_timestamp))
        upload_folder_to_gcs("outputs", args.gcs_bucket, args.gcs_prefix)
        print("\n[Done]")


if __name__ == "__main__":
    main()
