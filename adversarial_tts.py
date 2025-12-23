import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Import Pymoo components
from _pymoo_optimizer import PymooOptimizer
from pymoo.algorithms.moo.nsga2 import NSGA2

# Import your new specialized modules
from model_loader import initialize_environment
from core_logic import run_optimization_generation
from reporting import finalize_run
from _helper import FitnessObjective, AttackMode


def parse_arguments():
    parser = argparse.ArgumentParser(description="Adversarial TTS Optimization Executable")

    # String parameters
    parser.add_argument("--ground_truth_text", type=str, default="I think the NFL is lame and boring",
                        help="The ground truth text input.")
    parser.add_argument("--target_text", type=str, default="The Seattle Seahawks are the best Team in the world",
                        help="The target text input.")

    # Numeric parameters
    parser.add_argument("--loop_count", type=int, default=1, help="The loop count to use.")
    parser.add_argument("--num_generations", type=int, default=150,
                        help="Number of generations for the optimizer.")
    parser.add_argument("--pop_size", type=int, default=100,
                        help="Population size.")
    parser.add_argument("--iv_scalar", type=float, default=0.5,
                        help="Interpolation vector scalar.")
    parser.add_argument("--size_per_phoneme", type=int, default=1,
                        help="Size per phoneme.")

    # Boolean parameters
    parser.add_argument("--notify", action="store_true",
                        help="If set, sends a WhatsApp notification upon completion.")

    # Enum/Selection parameters
    parser.add_argument("--mode", type=str, default="TARGETED",
                        choices=["TARGETED", "UNTARGETED", "NOISE_UNTARGETED"],
                        help="Attack mode (case sensitive).")

    parser.add_argument("--ACTIVE_OBJECTIVES", nargs="+", type=str,
                        default=["PESQ", "WER_GT"],
                        help="List of active objectives (e.g. PESQ WER_GT UTMOS).")

    parser.add_argument("--thresholds", nargs='*', type=str, default=[],
                        help="Early stopping thresholds. Format: OBJ=Val (e.g. --thresholds PESQ=0.35 WER_GT=0.05)")

    return parser.parse_args()


def main():
    # 1. Parse Arguments and set Device
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. Initialize Environment
    # This handles Enums, Thresholds, Model Loading, and Reference Data generation
    models, data = initialize_environment(args, device)

    # Safety check if initialization failed (e.g., invalid Enum name)
    if models is None or data is None:
        return

    # 3. Setup Optimizer
    # Extract phoneme count from the StyleTTS2 processing done in the loader
    phoneme_count = data['input_lengths'].detach().cpu().item()

    optimizer = PymooOptimizer(
        bounds=(0, 1),
        algorithm=NSGA2,
        algo_params={"pop_size": args.pop_size},
        num_objectives=len(data['ACTIVE_OBJECTIVES']),
        solution_shape=(phoneme_count, args.size_per_phoneme),
    )

    print(f"Starting Optimization Loop...")

    # 4. Main Loop
    for iteration in tqdm(range(args.loop_count), desc="Total Progress"):

        # Run the generation loop (Core Logic)
        fitness_history, mean_model, progress_bar, stop_optimization, gen = run_optimization_generation(
            optimizer,
            iteration,
            models,
            data,
            args,
            device
        )

        # Prepare runtime context for the reporting module
        run_context = {
            "fitness_history": fitness_history,
            "mean_model": mean_model,
            "progress_bar": progress_bar,
            "current_gen": gen,
            "stop_optimization": stop_optimization,
            "active_objectives": data['ACTIVE_OBJECTIVES'],
            "objective_order": data['OBJECTIVE_ORDER'],
            "thresholds": data['THRESHOLDS']
        }

        # 5. Finalize and Save Results
        # This creates folders, saves audio, generates graphs, and sends notifications
        finalize_run(
            optimizer,
            models,
            args,
            run_context,
            data,
            device
        )

        # Break outer loop if early stopping was triggered in the inner loop
        if stop_optimization:
            break

if __name__ == "__main__":
    main()