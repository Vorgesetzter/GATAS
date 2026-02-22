"""
AdversarialTrainer - Optimization-only trainer for adversarial TTS.

This class handles ONLY the optimization logic for a single cycle:
1. Initialize optimizer with fresh interpolation vectors
2. Run generations until complete or threshold met
3. Return optimization results

The outer loop (loop_count iterations) is handled by the calling code.
Logging and visualization should be handled separately via RunLogger.
"""

import time
import torch
import numpy as np
from tqdm.auto import tqdm

# Local imports
from Datastructures.dataclass import ObjectiveContext
from Objectives.FitnessObjective import FitnessObjective
from Objectives.base import BaseObjective

from Trainer.VectorManipulator import VectorManipulator


class AdversarialTrainer:

    def __init__(
        self,
        tts_model,
        asr_model,
        thresholds: dict[FitnessObjective, float],
        objectives: dict[FitnessObjective, BaseObjective],
        vector_manipulator: VectorManipulator,
        device: str
    ):
        # Store state
        self.tts_model = tts_model
        self.asr_model = asr_model.module if isinstance(asr_model, torch.nn.DataParallel) else asr_model
        self.thresholds = thresholds
        self.objectives = objectives
        self.vector_manipulator = vector_manipulator
        self.device = device

    def evaluate_batch(self, context: ObjectiveContext) -> dict[FitnessObjective, list[float]]:
        """Evaluate all objectives on a batch."""
        scores = {}
        for obj_enum, objective in self.objectives.items():
            try:
                scores[obj_enum] = objective.calculate_score(context)
            except Exception as e:
                print(f"[ERROR] {obj_enum.name} evaluation failed: {e}")
        return scores

    def run_full_iteration(self, optimizer, num_generations, pop_size, batch_size) -> tuple[list[np.ndarray], int, float]:
        """
        Run a single optimization cycle through all generations.

        Initializes a fresh optimizer, runs until generations complete or
        threshold is met, and returns the results.

        Returns:
            tuple: (history, generations_run, total_inference_time)
                - history: List of matrices (one per generation)
                - generations_run: Integer count of generations completed
                - total_inference_time: Float seconds
        """

        fitness_history = []
        archive_history = []
        gen = -1
        elapsed_time_total = 0.0

        print("Press Ctrl+C to stop training early and save results.")

        try:
            # 1. Create the progress bar object
            with tqdm(range(num_generations), desc="Generations", leave=False) as pbar:

                for gen in pbar:
                    # --- Existing Logic ---
                    fitness_score_per_objective, stop_optimization, elapsed_time, audio_per_individual = self.run_one_generation(optimizer, pop_size, batch_size)
                    fitness_arrays = [np.array(scores) for scores in fitness_score_per_objective]

                    optimizer.assign_fitness(fitness_arrays, audio_per_individual)
                    optimizer.update()

                    generation_matrix = np.column_stack(fitness_arrays)
                    fitness_history.append(generation_matrix)

                    archive_snapshot = np.array([list(c.fitness) for c in optimizer.best_candidates])
                    archive_history.append(archive_snapshot)

                    elapsed_time_total += elapsed_time

                    # Calculate stats for this generation
                    # axis=0 means "down the column" (across all individuals)
                    current_means = generation_matrix.mean(axis=0)
                    current_mins = np.array([list(c.fitness) for c in optimizer.best_candidates]).min(axis=0)

                    # Format the string (e.g., "WER: 0.15 (Avg: 0.40) | PESQ: ...")
                    stats_parts = []
                    for idx, obj in enumerate(self.objectives):
                        stats_parts.append(
                            f"{obj.name}: {current_mins[idx]:.4f} (Avg: {current_means[idx]:.4f})"
                        )

                    status_message = " | ".join(stats_parts)

                    # Print a permanent log line (Preserves history)
                    pbar.write(f"[Gen {gen + 1}] {status_message}")

                    if stop_optimization:
                        pbar.write(f"\n[!] Early Stopping at Generation {gen + 1} (Thresholds met).")
                        break

        except KeyboardInterrupt:
            print(f"\n[!] Manual Stop triggered at Generation {gen + 1}. Saving results so far...")

        # Clean up
        torch.cuda.empty_cache()

        return fitness_history, archive_history, gen+1, elapsed_time_total

    def run_one_generation(self, optimizer, pop_size, batch_size) -> tuple[list[list[float]], bool, float, list]:

        stop_optimization = False
        total_elapsed_time = 0

        # Create list to store scores for this generation
        fitness_per_objective: list[list[float]] = [[] for _ in self.objectives]
        audio_per_individual: list = []

        # Get current population from optimizer
        interpolation_vectors_full = torch.from_numpy(optimizer.get_x_current()).to(self.device).float()

        # Process batches
        for batch_idx in range(0, pop_size, batch_size):
            batch_stop, batch_fitness_per_objective, elapsed_time, batch_audio = self._process_batch(
                batch_idx,
                batch_size,
                interpolation_vectors_full,
            )
            total_elapsed_time += elapsed_time

            for obj_idx, score in enumerate(batch_fitness_per_objective):
                fitness_per_objective[obj_idx].extend(score)

            audio_per_individual.extend(batch_audio)

            if batch_stop:
                stop_optimization = True

        return fitness_per_objective, stop_optimization, total_elapsed_time, audio_per_individual

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _process_batch(self, batch_idx: int, batch_size: int, interpolation_vectors_full: torch.Tensor) -> tuple[bool, list[list[float]], float, list]:
        """
        Returns:
            stop_optimization (bool): True if early stopping met.
            batch_scores_list (list[list[float]]): Scores for this batch, grouped by objective.

        Args:
            batch_idx: Starting index of this batch
            interpolation_vectors_full: Full population tensor

        Returns:
            True if early stopping criteria met, False otherwise
        """
        start_time = time.time()

        # 1. TTS Inference
        interpolation_vectors_batch = interpolation_vectors_full[batch_idx: batch_idx + batch_size]
        current_batch_size, interpolation_vectors, audio_embedding_data_mixed = self.vector_manipulator.interpolate(interpolation_vectors_batch)
        audio_mixed_batch = self.tts_model.inference_on_embedding(audio_embedding_data_mixed)

        # 2. ASR Inference
        asr_texts, mel_batch = self.asr_model.inference(audio_mixed_batch)

        # 3. Create context and evaluate objectives
        context = ObjectiveContext(
            audio_mixed_batch=audio_mixed_batch,
            asr_texts=asr_texts,
            interpolation_vectors=interpolation_vectors,
            mel_batch=mel_batch
        )
        batch_scores_dict = self.evaluate_batch(context)

        end_time = time.time()

        # 4. Collect scores and check early stopping (vectorized)
        batch_scores_list = []

        # Build score matrix: [batch_size, num_objectives]
        score_matrix = np.zeros((current_batch_size, len(self.objectives)), dtype=np.float32)

        for obj_idx, obj in enumerate(self.objectives):
            scores = np.array(batch_scores_dict[obj], dtype=np.float32)

            # Store in matrix for check
            score_matrix[:, obj_idx] = scores

            # Store in list for return
            batch_scores_list.append(scores.tolist())

        # Vectorized early stopping check
        stop_optimization = self._check_early_stopping_batch(score_matrix)
        elapsed_time = end_time - start_time

        # Store each individual's audio on CPU so it can be retrieved later without
        # re-synthesis (re-synthesis is non-deterministic due to cumsum_cuda and other
        # CUDA ops, which can flip Whisper's output on borderline adversarial audio).
        audio_list = [audio_mixed_batch[i].detach().cpu() for i in range(current_batch_size)]

        return stop_optimization, batch_scores_list, elapsed_time, audio_list

    def _check_early_stopping_batch(self, score_matrix: np.ndarray) -> bool:
        """
        Check if any individual in the batch meets all threshold criteria.

        Args:
            score_matrix: [batch_size, num_objectives] array of scores

        Returns:
            True if any individual meets all thresholds, False otherwise
        """
        if not self.thresholds:
            return False

        # Build threshold array aligned with active_objectives order
        threshold_mask = []  # Which objectives have thresholds
        threshold_values = []

        for obj_idx, obj in enumerate(self.objectives):
            if obj in self.thresholds:
                threshold_mask.append(obj_idx)
                threshold_values.append(self.thresholds[obj])

        if not threshold_mask:
            return False

        # Extract only columns with thresholds
        relevant_scores = score_matrix[:, threshold_mask]
        thresholds = np.array(threshold_values, dtype=np.float32)

        # Check if any row has ALL scores <= thresholds
        meets_thresholds = relevant_scores <= thresholds  # [batch, num_thresholds]
        any_meets_all = np.any(np.all(meets_thresholds, axis=1))

        return bool(any_meets_all)
