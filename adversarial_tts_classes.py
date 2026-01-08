"""
Adversarial TTS - Class-based entry point for optimization.

This script uses the refactored class-based architecture:
- EnvironmentLoader: Handles model loading and environment setup
- AdversarialTrainer: Runs the optimization loop (returns results)
- RunLogger: Handles all output and logging (called separately)

Usage:
    python adversarial_tts_classes.py --ground_truth_text "Hello world" --target_text "Goodbye"
"""

import torch
import argparse

# Import class-based modules
from Trainer.ModelLoader import EnvironmentLoader
from Trainer.AdversarialTrainer import AdversarialTrainer
from Trainer.RunLogger import RunLogger


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Adversarial TTS Optimization (Class-based)")

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
    parser.add_argument(
        "--multi_gpu", action="store_true",
        help="Enable multi-GPU support (requires multiple CUDA devices)."
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
    """Main entry point using class-based architecture."""

    # 1. Parse arguments and set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parse_arguments()

    print("=" * 60)
    print(" ADVERSARIAL TTS - Class-Based Architecture")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Ground Truth: {args.ground_truth_text}")
    print(f"Target Text:  {args.target_text}")
    print("=" * 60)

    # 2. Initialize environment using EnvironmentLoader
    loader = EnvironmentLoader(args, device)
    config, models, audio, embeds, objective_manager = loader.initialize()

    if config is None:
        print("[ERROR] Initialization failed. Exiting.")
        return

    # 3. Create trainer (optimization only)
    trainer = AdversarialTrainer(
        config=config,
        models=models,
        audio=audio,
        embeds=embeds,
        objective_manager=objective_manager,
        device=device
    )

    # 4. Run optimization - returns results without logging
    results = trainer.run()

    # 5. Log results separately (can be re-run with different settings)
    logger = RunLogger(config, models, audio, device)

    for i, result in enumerate(results):
        print(f"\n[Logging] Saving results for loop {i + 1}...")
        logger.finalize_run(
            result.fitness_data,
            result.generation_count,
            result.elapsed_time
        )

    print("\n" + "=" * 60)
    print(" OPTIMIZATION COMPLETE")
    print(f" Results saved to: {logger.folder_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
