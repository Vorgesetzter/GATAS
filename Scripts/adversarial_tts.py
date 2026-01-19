"""
Adversarial TTS - Class-based entry point for optimization.

This script uses the refactored class-based architecture:
- EnvironmentLoader: Handles argument parsing, model loading, and environment setup
- AdversarialTrainer: Runs the optimization loop (returns results)
- RunLogger: Handles all output and logging (called separately)

Usage:
    python adversarial_tts.py --ground_truth_text "Hello world" --target_text "Goodbye"
"""

import torch
import os
import argparse
from Datastructures.enum import AttackMode
from Objectives.FitnessObjective import FitnessObjective
import matplotlib.pyplot as plt

from Trainer import VectorManipulator

os.chdir("..") # Since we are in Scripts Folder

# Import class-based modules
from Trainer.EnvironmentLoader import EnvironmentLoader
from Trainer.AdversarialTrainer import AdversarialTrainer
from Trainer.RunLogger import RunLogger
from Trainer.GraphPlotter import GraphPlotter
from Trainer.VectorManipulator import VectorManipulator
from Models.styletts2 import StyleTTS2

from helper import write_run_summary

# Import Pymoo components
from Optimizer.pymoo_optimizer import PymooOptimizer
from pymoo.algorithms.moo.nsga2 import NSGA2

from Datastructures.dataclass import ModelData

def initialize_parser():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Adversarial TTS Optimization")

    # String parameters
    parser.add_argument("--ground_truth_text", type=str, default="I think the NFL is lame and boring", help="The ground truth text input.")
    parser.add_argument("--target_text", type=str, default="The Seattle Seahawks are the best Team in the world", help="The target text input.")

    # Numeric parameters
    parser.add_argument("--loop_count", type=int, default=1, help="Number of optimization loops.")
    parser.add_argument("--num_generations", type=int, default=1, help="Generations per loop.")
    parser.add_argument("--pop_size", type=int, default=1, help="Population size.")
    parser.add_argument("--iv_scalar", type=float, default=0.5, help="Interpolation vector scalar.")
    parser.add_argument("--size_per_phoneme", type=int, default=1, help="Size per phoneme.")
    parser.add_argument("--batch_size", type=int, default=-1, help="Batch size (-1 for full batch).")

    # Boolean parameters
    parser.add_argument("--notify", action="store_true", help="Send WhatsApp notification on completion.")
    parser.add_argument("--subspace_optimization", action="store_true", help="Enable subspace optimization for embedding vector.")

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
    config_data = loader.load_configuration(args)
    config_data.print_summary()

    tts_model, asr_model = loader.load_required_models()

    audio_gt, audio_target, audio_embedding_gt, audio_embedding_target = loader.generate_audio_data(config_data.mode, config_data.text_gt, config_data.text_target, tts_model)

    # 4. Initialize Objectives
    objectives = loader.initialize_objectives(
        active_objectives=config_data.active_objectives,
        model_data=ModelData(tts_model=tts_model, asr_model=asr_model),
        text_gt=config_data.text_gt,
        text_target=config_data.text_target,
        mode=config_data.mode,
        audio_gt=audio_gt,
    )

    vector_manipulator = VectorManipulator(audio_embedding_gt, audio_embedding_target.h_text, config_data)

    # 3. Create Trainer and Logger
    trainer = AdversarialTrainer(tts_model, asr_model, config_data.thresholds, objectives, vector_manipulator, device)
    logger = RunLogger(config_data.active_objectives, tts_model, asr_model, vector_manipulator, device)

    # 4. Run optimization loops
    for loop_iteration in range(config_data.loop_count):
        print(f"\n[Loop {loop_iteration + 1}/{config_data.loop_count}] Starting optimization loop...")

        # Initialize fresh optimizer for this cycle
        optimizer = PymooOptimizer(
            bounds=(0, 1),
            algorithm=NSGA2,
            algo_params={"pop_size": config_data.pop_size},
            num_objectives=len(config_data.active_objectives),
            solution_shape=(audio_embedding_gt.input_length.detach().cpu().item(), config_data.size_per_phoneme),
        )

        fitness_data, generation_count, elapsed_time_total = trainer.run_full_iteration(optimizer, config_data.num_generations, config_data.pop_size, config_data.batch_size)

        # 5. Log Results
        folder_path = logger.setup_output_directory()
        logger.save_fitness_history(fitness_data)

        best_candidate = logger.select_best_candidate(optimizer.best_candidates, config_data.thresholds)
        audio_best, text_best, audio_embedding_best = logger.run_final_inference(best_candidate)

        logger.save_audios(audio_gt, audio_target, audio_best)
        logger.save_torch_state(text_best, audio_embedding_best, best_candidate, config_data)

        # 6. Generate Graphs
        graph_plotter = GraphPlotter(config_data.active_objectives, generation_count, folder_path, fitness_data)
        graph_plotter.generate_hypervolume_graph()
        graph_plotter.generate_pareto_population_graph()
        graph_plotter.generate_mean_population_graph()
        graph_plotter.generate_minimal_population_graph()
        plt.close('all')

        # 7. Write Summary to Terminal
        write_run_summary(folder_path, text_best, best_candidate, generation_count, elapsed_time_total, config_data)

        print("[Log] Finished saving all results")


if __name__ == "__main__":
    main()
