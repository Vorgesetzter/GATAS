"""
AdversarialTrainer - Optimization-only trainer for adversarial TTS.

This class handles ONLY the optimization logic:
1. Main optimization loop (generation-based)
2. Audio generation from interpolation vectors
3. ASR transcription and fitness evaluation

Logging and visualization should be handled separately via RunLogger.
"""

import time
import sys
import re
import torch
import numpy as np
import whisper
import torchaudio.functional as torchaudio_functional
from tqdm.auto import tqdm
from whisper.tokenizer import get_tokenizer
from dataclasses import dataclass

# Local imports
from Datastructures.dataclass import ConfigData, ModelData, AudioData, EmbeddingData, StepContext, FitnessData
from Datastructures.enum import FitnessObjective
from Objectives.manager import ObjectiveManager
from Trainer.ModelLoader import load_optimizer


@dataclass
class OptimizationResult:
    """Result from a single optimization cycle."""
    fitness_data: FitnessData
    generation_count: int
    elapsed_time: float
    stopped_early: bool


class AdversarialTrainer:
    """
    Optimization-only trainer for adversarial TTS.

    This class focuses purely on running the optimization loop.
    Use RunLogger separately to handle logging and visualization.

    Usage:
        trainer = AdversarialTrainer(config, models, audio, embeds, objective_manager, device)
        results = trainer.run()  # Returns list of OptimizationResult

        # Then use RunLogger separately
        logger = RunLogger(config, models, audio, device)
        for result in results:
            logger.finalize_run(result.fitness_data, result.generation_count, result.elapsed_time)
    """

    def __init__(
        self,
        config: ConfigData,
        models: ModelData,
        audio: AudioData,
        embeds: EmbeddingData,
        objective_manager: ObjectiveManager,
        device: str
    ):
        # Store state
        self.config = config
        self.models = models
        self.audio = audio
        self.embeds = embeds
        self.objective_manager = objective_manager
        self.device = device

        # Prepare Whisper components
        self._real_asr_model = self._get_real_asr_model(self.models.asr_model)
        self._target_tokens_template = None

        if FitnessObjective.WHISPER_PROB in self.config.active_objectives:
            self._target_tokens_template = self._prepare_whisper_tokens()

    def run(self) -> list[OptimizationResult]:
        """
        Run the full optimization process.

        Returns:
            List of OptimizationResult, one per loop iteration.
        """
        print(f"[Info] Starting Training for {self.config.loop_count} loops...")

        results = []

        for loop_idx in range(self.config.loop_count):
            print(f"\n--- Optimization Loop {loop_idx + 1}/{self.config.loop_count} ---")

            # Run optimization
            start_time = time.time()
            fitness_data, gen_count, stopped_early = self._optimize_one_cycle(loop_idx)
            elapsed_time = time.time() - start_time

            # Store result
            result = OptimizationResult(
                fitness_data=fitness_data,
                generation_count=gen_count,
                elapsed_time=elapsed_time,
                stopped_early=stopped_early
            )
            results.append(result)

            if stopped_early:
                print(f"[Info] Early stopping triggered. Ending optimization.")
                break

            # Reset optimizer for next loop (if not last)
            if loop_idx < self.config.loop_count - 1:
                self._reset_state()

        return results

    def run_single_cycle(self, iteration: int = 0) -> OptimizationResult:
        """
        Run a single optimization cycle.

        Args:
            iteration: Iteration number for display purposes.

        Returns:
            OptimizationResult with fitness data and timing.
        """
        start_time = time.time()
        fitness_data, gen_count, stopped_early = self._optimize_one_cycle(iteration)
        elapsed_time = time.time() - start_time

        return OptimizationResult(
            fitness_data=fitness_data,
            generation_count=gen_count,
            elapsed_time=elapsed_time,
            stopped_early=stopped_early
        )

    def _optimize_one_cycle(self, iteration: int) -> tuple[FitnessData, int, bool]:
        """
        Runs one full optimization cycle (all generations).

        Returns:
            Tuple of (FitnessData, generation_count, stopped_early)
        """
        # History tracking
        pareto_fitness_history = []
        mean_fitness_history = []
        total_fitness_history = []
        stop_optimization = False

        progress_bar = tqdm(
            range(self.config.num_generations),
            desc=f"Generation Loop {iteration + 1}",
            leave=False
        )

        gen = 0
        options = whisper.DecodingOptions()

        for gen in progress_bar:
            # Per-generation score tracking
            gen_scores: dict[FitnessObjective, list[float]] = {
                obj: [] for obj in self.config.active_objectives
            }

            # Get current population from optimizer
            interpolation_vectors_full = torch.from_numpy(
                self.models.optimizer.get_x_current()
            ).to(self.device).float()

            # Process batches
            for batch_idx in range(0, self.config.pop_size, self.config.batch_size):
                batch_stop = self._process_batch(
                    batch_idx,
                    interpolation_vectors_full,
                    options,
                    gen_scores
                )

                if batch_stop:
                    stop_optimization = True

            # End of generation processing
            gen_mean, gen_total, total_fitness = self._compute_generation_stats(gen, gen_scores)

            # Update optimizer
            self.models.optimizer.assign_fitness(gen_total)
            self.models.optimizer.update()

            # Capture Pareto front
            current_front = np.array([c.fitness for c in self.models.optimizer.best_candidates])

            # Add to history
            mean_fitness_history.append(gen_mean)
            total_fitness_history.append(total_fitness)
            pareto_fitness_history.append(current_front)


            if stop_optimization:
                print(f"\n[!] Early Stopping at Generation {gen + 1} (Thresholds met).")
                break

        fitness_data = FitnessData(mean_fitness_history, pareto_fitness_history, total_fitness_history)
        return fitness_data, gen + 1, stop_optimization

    def _process_batch(
        self,
        batch_idx: int,
        interpolation_vectors_full: torch.Tensor,
        options: whisper.DecodingOptions,
        gen_scores: dict
    ) -> bool:
        """
        Process a single batch through TTS -> ASR -> Fitness evaluation.

        Args:
            batch_idx: Starting index of this batch
            interpolation_vectors_full: Full population tensor
            options: Whisper decoding options
            gen_scores: Dict to collect scores into (modified in-place)

        Returns:
            True if early stopping criteria met, False otherwise
        """
        stop_optimization = False

        # 1. TTS Inference
        audio_mixed_batch, current_batch_size, interpolation_vectors = \
            self.models.tts_model.inference_on_interpolation_vectors(
                interpolation_vectors_full,
                batch_idx,
                self.config.batch_size,
                self.config,
                self.audio
            )

        # 2. Prepare audio tensors (single conversion from numpy)
        audio_tensor_full = torch.from_numpy(audio_mixed_batch).to(self.device)
        audio_tensor_asr = audio_tensor_full.squeeze(1).float()
        audio_tensor_asr = torchaudio_functional.resample(audio_tensor_asr, 24000, 16000)
        audio_tensor_asr = whisper.pad_or_trim(audio_tensor_asr)

        # 3. Create Mel spectrogram
        mel_batch = whisper.log_mel_spectrogram(
            audio_tensor_asr, n_mels=self._real_asr_model.dims.n_mels
        ).to(self.device)

        # 4. Compute WHISPER_PROB (if active)
        batch_whisper_values = self._compute_whisper_prob_batch(mel_batch, current_batch_size)

        # 5. Run ASR decoding
        results = whisper.decode(self._real_asr_model, mel_batch, options)

        # 6. Process ASR results
        asr_texts = [r.text for r in results]
        clean_texts = [re.sub(r'[^a-zA-Z\s]', '', t).strip() for t in asr_texts]

        # 7. Create StepContext
        context = StepContext(
            audio_mixed=audio_tensor_full,
            asr_text=asr_texts,
            clean_text=clean_texts,
            interpolation_vector=interpolation_vectors,
            whisper_prob=batch_whisper_values
        )

        # 8. Evaluate objectives
        batch_scores = self.objective_manager.evaluate_batch(context, self.audio)

        # 9. Collect scores and check early stopping
        for i in range(current_batch_size):
            current_ind_scores: dict[FitnessObjective, float] = {}

            if len(clean_texts[i]) < 2:
                # Garbage text penalty
                for obj in self.config.active_objectives:
                    gen_scores[obj].append(1.0)
                    current_ind_scores[obj] = 1.0
            else:
                for obj in self.config.active_objectives:
                    score = batch_scores[obj][i]
                    gen_scores[obj].append(score)
                    current_ind_scores[obj] = score

            # Early stopping check
            if self._check_early_stopping(current_ind_scores):
                stop_optimization = True

        return stop_optimization

    def _reset_state(self):
        """Reset state between optimization loops."""
        self.models.optimizer = load_optimizer(self.audio, self.config)
        torch.cuda.empty_cache()

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_real_asr_model(self, asr_model):
        """Extract actual model from DataParallel wrapper if needed."""
        if isinstance(asr_model, torch.nn.DataParallel):
            return asr_model.module
        return asr_model

    def _prepare_whisper_tokens(self) -> torch.Tensor:
        """Prepare tokenized target text for WHISPER_PROB computation."""
        tokenizer = get_tokenizer(self._real_asr_model.is_multilingual)
        target_ids = (
            list(tokenizer.sot_sequence) +
            tokenizer.encode(self.config.text_target) +
            [tokenizer.eot]
        )
        return torch.tensor([target_ids]).to(self.device)

    def _compute_whisper_prob_batch(self, mel_batch: torch.Tensor, batch_size: int) -> list:
        """Compute WHISPER_PROB fitness values for a batch."""
        if FitnessObjective.WHISPER_PROB not in self.config.active_objectives:
            return [None] * batch_size

        if self._target_tokens_template is None:
            return [None] * batch_size

        target_tokens_batch = self._target_tokens_template.expand(batch_size, -1)

        with torch.no_grad():
            logits = self.models.asr_model(mel_batch, target_tokens_batch)

        # Remove start and end token
        logits_shifted = logits[:, :-1, :]
        targets_shifted = target_tokens_batch[:, 1:]

        # Cross-entropy loss per token
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        raw_losses = loss_fct(
            logits_shifted.reshape(-1, logits_shifted.size(-1)),
            targets_shifted.reshape(-1)
        )

        # Average loss over sentence length
        sample_losses = raw_losses.reshape(batch_size, -1).mean(dim=1)

        # Convert to fitness score (0.0 = best, 1.0 = worst)
        probs = torch.exp(-sample_losses)
        vals = 1.0 - probs

        return vals.detach().cpu().tolist()

    def _check_early_stopping(self, current_ind_scores: dict) -> bool:
        """Check if current individual meets all threshold criteria."""
        if not self.config.thresholds:
            return False

        for obj in self.config.active_objectives:
            if obj in self.config.thresholds:
                if current_ind_scores[obj] > self.config.thresholds[obj]:
                    return False

        return True

    def _compute_generation_stats(self, gen: int, gen_scores: dict):
        """Compute statistics for the current generation."""
        gen_mean: dict[str, float] = {"Generation": gen}
        gen_total: list[np.ndarray] = []

        for obj in self.config.objective_order:
            if obj not in self.config.active_objectives:
                continue

            arr = np.array(gen_scores[obj], dtype=float)
            gen_mean[obj.name] = float(np.mean(arr))
            gen_total.append(arr)

        total_fitness = np.column_stack(gen_total)

        return gen_mean, gen_total, total_fitness


# Backward-compatible function wrapper
def run_optimization_generation(
    config_data: ConfigData,
    model_data: ModelData,
    audio_data: AudioData,
    embedding_data: EmbeddingData,
    objective_manager: ObjectiveManager,
    iteration: int,
    device: str
):
    """
    Backward-compatible wrapper that runs one optimization cycle.

    This function creates a temporary trainer and runs a single cycle.
    For new code, prefer using AdversarialTrainer directly.
    """
    trainer = AdversarialTrainer(
        config=config_data,
        models=model_data,
        audio=audio_data,
        embeds=embedding_data,
        objective_manager=objective_manager,
        device=device
    )

    # Run one cycle
    result = trainer.run_single_cycle(iteration)

    # Create a dummy progress bar for backward compatibility
    progress_bar = tqdm(range(config_data.num_generations), desc="", leave=False)
    progress_bar.n = result.generation_count
    progress_bar.refresh()

    return result.fitness_data, progress_bar, result.stopped_early, result.generation_count - 1
