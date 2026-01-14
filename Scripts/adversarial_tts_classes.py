"""
Adversarial TTS - Class-based entry point for optimization.

This script uses the refactored class-based architecture:
- EnvironmentLoader: Handles argument parsing, model loading, and environment setup
- AdversarialTrainer: Runs the optimization loop (returns results)
- RunLogger: Handles all output and logging (called separately)

Usage:
    python adversarial_tts_classes.py --ground_truth_text "Hello world" --target_text "Goodbye"
"""

import torch
import os
import argparse
from Datastructures.enum import AttackMode, FitnessObjective

os.chdir("..") # Since we are in Scripts Folder

# Import class-based modules
from Trainer.EnvironmentLoader import EnvironmentLoader
from Trainer.AdversarialTrainer import AdversarialTrainer
from Trainer.RunLogger import RunLogger

# Import Pymoo components
from Optimizer.pymoo_optimizer import PymooOptimizer
from pymoo.algorithms.moo.nsga2 import NSGA2

def initialize_parser():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Adversarial TTS Optimization")

    # String parameters
    parser.add_argument("--ground_truth_text", type=str, default="I think the NFL is lame and boring", help="The ground truth text input.")
    parser.add_argument("--target_text", type=str, default="The Seattle Seahawks are the best Team in the world", help="The target text input.")

    # Numeric parameters
    parser.add_argument("--loop_count", type=int, default=1, help="Number of optimization loops.")
    parser.add_argument("--num_generations", type=int, default=4, help="Generations per loop.")
    parser.add_argument("--pop_size", type=int, default=4, help="Population size.")
    parser.add_argument("--iv_scalar", type=float, default=0.5, help="Interpolation vector scalar.")
    parser.add_argument("--size_per_phoneme", type=int, default=1, help="Size per phoneme.")
    parser.add_argument("--batch_size", type=int, default=-1, help="Batch size (-1 for full batch).")

    # Boolean parameters
    parser.add_argument("--notify", action="store_true", help="Send WhatsApp notification on completion.")
    parser.add_argument("--subspace_optimization", action="store_true", help="Enable subspace optimization for embedding vector.")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU support (requires multiple CUDA devices).")

    # Enum/Selection parameters
    parser.add_argument("--mode", type=str.upper, default="TARGETED", choices=AttackMode._member_names_, help="Attack mode.")
    parser.add_argument("--ACTIVE_OBJECTIVES", nargs="+", type=str.upper, default=["PESQ", "WER_GT"], choices=FitnessObjective._member_names_, help="List of active objectives (e.g. PESQ WER_GT UTMOS).")
    parser.add_argument("--thresholds", nargs='*', type=str, default=["PESQ=0.3", "WER_GT=0.5"], help="Early stopping thresholds. Format: OBJ=Val")

    return parser

def main():

    parser = initialize_parser()
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load environment
    loader = EnvironmentLoader(device)
    config_data, model_data, audio_data, objective_manager = loader.initialize(args)

    # 2. Create trainer
    trainer = AdversarialTrainer(config_data, model_data, audio_data, objective_manager, device)

    # 3. Run optimization loops
    for loop_iteration in range(config_data.loop_count):
        print(f"\n[Loop {loop_iteration + 1}/{config_data.loop_count}] Starting optimization loop...")

        # Initialize fresh optimizer for this cycle
        optimizer = PymooOptimizer(
            bounds=(0, 1),
            algorithm=NSGA2,
            algo_params={"pop_size": config_data.pop_size},
            num_objectives=len(config_data.active_objectives),
            solution_shape=(audio_data.input_lengths.detach().cpu().item(), config_data.size_per_phoneme),
        )

        fitness_data, generation_count, elapsed_time_total = trainer.run_full_iteration(optimizer)

        # 4. Log Results
        logger = RunLogger(optimizer, config_data, model_data, audio_data, fitness_data, generation_count, elapsed_time_total, device)
        logger.finalize_run()

        print(f"Results saved to: {logger.folder_path}")


if __name__ == "__main__":
    main()
