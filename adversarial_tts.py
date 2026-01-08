"""
Adversarial TTS - Main entry point for optimization.

This script runs multi-objective optimization to generate adversarial
audio that sounds like one text but is transcribed as another.
"""

import torch
import argparse
from tqdm import tqdm
import cProfile
import os

# Import specialized modules
from Trainer.ModelLoader import initialize_environment, load_optimizer
from Adversarial_TTS.core_logic import run_optimization_generation
from Adversarial_TTS.reporting import finalize_run


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Adversarial TTS Optimization")

    # String parameters
    parser.add_argument(
        "--ground_truth_text", type=str,
        default="I think the NFL is lame and boring",
        help="The ground truth text input."
    )
    parser.add_argument(
        "--target_text", type=str,
        default="The Seattle Seahawks are the best Team in the world",
        help="The target text input."
    )

    # Numeric parameters
    parser.add_argument("--loop_count", type=int, default=1, help="Number of optimization loops.")
    parser.add_argument("--num_generations", type=int, default=4, help="Generations per loop.")
    parser.add_argument("--pop_size", type=int, default=4, help="Population size.")
    parser.add_argument("--iv_scalar", type=float, default=0.5, help="Interpolation vector scalar.")
    parser.add_argument("--size_per_phoneme", type=int, default=1, help="Size per phoneme.")
    parser.add_argument("--batch_size", type=int, default=-1, help="Batch size (-1 for full batch).")

    # Boolean parameters
    parser.add_argument(
        "--notify", action="store_true",
        help="Send WhatsApp notification on completion."
    )
    parser.add_argument(
        "--subspace_optimization", action="store_true",
        help="Enable subspace optimization for embedding vector."
    )

    # Enum/Selection parameters
    parser.add_argument(
        "--mode", type=str, default="TARGETED",
        choices=["TARGETED", "UNTARGETED", "NOISE_UNTARGETED"],
        help="Attack mode."
    )
    parser.add_argument(
        "--ACTIVE_OBJECTIVES", nargs="+", type=str,
        default=["PESQ", "WHISPER_PROB"],
        help="List of active objectives (e.g. PESQ WER_GT UTMOS)."
    )
    parser.add_argument(
        "--thresholds", nargs='*', type=str,
        default=["PESQ=0.3", "WHISPER_PROB=0.25"],
        help="Early stopping thresholds. Format: OBJ=Val"
    )

    return parser.parse_args()


def main():
    """Main entry point for adversarial TTS optimization."""

    # 1. Parse arguments and set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parse_arguments()

    # 2. Initialize environment (models, data, objectives)
    result = initialize_environment(args, device)
    config_data, model_data, audio_data, embedding_data, objective_manager = result

    # Safety check if initialization failed
    if config_data is None:
        print("Initialization failed. Exiting.")
        return

    print(f"\nStarting Optimization Loop...")

    # 3. Main optimization loop
    for iteration in tqdm(range(config_data.loop_count), desc="Total Progress"):

        # Run the generation loop with ObjectiveManager
        fitness_data, progress_bar, stop_optimization, gen = run_optimization_generation(
            config_data=config_data,
            model_data=model_data,
            audio_data=audio_data,
            embedding_data=embedding_data,
            objective_manager=objective_manager,
            iteration=iteration,
            device=device
        )

        # 4. Finalize and save results
        finalize_run(
            config_data, fitness_data, model_data, audio_data, progress_bar, gen, device
        )

        # Reset optimizer for next iteration
        model_data.optimizer = load_optimizer(audio_data, config_data)


if __name__ == "__main__":
    main()
