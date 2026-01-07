import torch
import argparse
from tqdm import tqdm
import cProfile
import os

# Import your new specialized modules
from Adversarial_TTS.model_loader import initialize_environment, load_optimizer
from Adversarial_TTS.core_logic import run_optimization_generation
from Adversarial_TTS.reporting import finalize_run

def parse_arguments():
    parser = argparse.ArgumentParser(description="Adversarial TTS Optimization Executable")

    # String parameters
    parser.add_argument("--ground_truth_text", type=str, default="I think the NFL is lame and boring", help="The ground truth text input.")
    parser.add_argument("--target_text", type=str, default="The Seattle Seahawks are the best Team in the world", help="The target text input.")

    # Numeric parameters
    parser.add_argument("--loop_count", type=int, default=1, help="The loop count to use.")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of generations for the optimizer.")
    parser.add_argument("--pop_size", type=int, default=4, help="Population size.")
    parser.add_argument("--iv_scalar", type=float, default=0.5, help="Interpolation vector scalar.")
    parser.add_argument("--size_per_phoneme", type=int, default=1, help="Size per phoneme.")
    parser.add_argument("--batch_size", type=int, default=-1, help="Batch size for inference. >1 for full batch.")

    # Boolean parameters
    parser.add_argument("--notify", action="store_true", help="If set, sends a WhatsApp notification upon completion.")
    parser.add_argument("--subspace_optimization", action="store_true", help="If set, does subspace optimization for the embedding vector.")

    # Enum/Selection parameters
    parser.add_argument("--mode", type=str, default="TARGETED", choices=["TARGETED", "UNTARGETED", "NOISE_UNTARGETED"], help="Attack mode (case sensitive).")
    parser.add_argument("--ACTIVE_OBJECTIVES", nargs="+", type=str, default=["PESQ", "WHISPER_PROB"], help="List of active objectives (e.g. PESQ WER_GT UTMOS).")
    parser.add_argument("--thresholds", nargs='*', type=str, default=["PESQ=0.3 ", "WHISPER_PROB=0.25"], help="Early stopping thresholds. Format: OBJ=Val (e.g. --thresholds PESQ=0.35 WER_GT=0.05)")

    return parser.parse_args()

def main():
    # 1. Parse Arguments and set Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parse_arguments()

    # 2. Initialize Environment
    # This handles Enums, Thresholds, Model Loading, and Reference Data generation
    config_data, model_data, audio_data, embedding_data = initialize_environment(args, device)

    # Safety check if initialization failed (e.g., invalid Enum name)
    if config_data is None:
        return

    print(f"Starting Optimization Loop...")

    # 4. Main Loop
    for iteration in tqdm(range(config_data.loop_count), desc="Total Progress"):
        profiler = cProfile.Profile()
        profiler.enable()

        # Run the generation loop (Core Logic)
        fitness_data, progress_bar, stop_optimization, gen = run_optimization_generation(
            config_data, model_data, audio_data, embedding_data, iteration, device
        )

        # 5. Finalize and Save Results
        # This creates folders, saves audio, generates graphs, and sends notifications
        folder_path = finalize_run(config_data, fitness_data, model_data, audio_data, progress_bar, gen, device)

        model_data.optimizer = load_optimizer(audio_data, config_data)

        profiler.disable()
        profiler.dump_stats(os.path.join(folder_path, "performance_data.prof"))

if __name__ == "__main__":
    main()