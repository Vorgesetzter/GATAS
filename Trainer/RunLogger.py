import os
import datetime
import platform
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Local Imports
from Datastructures.dataclass import ConfigData, ModelData, AudioData, FitnessData, BestMixedAudio
from Datastructures.enum import AttackMode
from Trainer.GraphPlotter import GraphPlotter
from helper import adjustInterpolationVector, send_whatsapp_notification

class RunLogger:
    def __init__(self, config: ConfigData, models: ModelData, audio: AudioData, device):
        """
        Initializes the logger and immediately creates the output directory.
        """
        self.config = config
        self.models = models
        self.audio = audio

        # Helper: Get device from model automatically
        self.device = device

        # Initialize Directory Immediately
        self.folder_path = self._setup_output_directory()

        self.graph_plotter = GraphPlotter(self.folder_path, self.config.active_objectives, self.config.num_generations)

    def _setup_output_directory(self) -> str:
        """Creates the timestamped output folder and returns its path."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

        objectives_str = "_".join([obj.name for obj in self.config.active_objectives])

        folder_path = os.path.join("outputs/", objectives_str, timestamp)
        os.makedirs(folder_path, exist_ok=True)

        print(f"[Info] Output directory initialized: {folder_path}")
        return folder_path

    # In RunLogger.py, inside log_metrics
    def log_metrics(self, loop_idx, fitness_data):
        if not fitness_data.mean_fitness:
            return

        # current_mean is likely: {"PHONEME_COUNT": 0.5, "PESQ": 1.2, ...}
        current_mean = fitness_data.mean_fitness[-1]

        if isinstance(current_mean, dict):
            # Pick the first objective to display in the progress bar / log
            # Or specify a specific one like current_mean.get("PESQ", 0.0)
            first_key = list(current_mean.keys())[0]
            score_val = current_mean[first_key]
            score_str = f"{first_key}: {score_val:.4f}"
        else:
            # Fallback if it's just a float or list
            score_val = current_mean[0] if isinstance(current_mean, (list, np.ndarray)) else current_mean
            score_str = f"{score_val:.4f}"

        # Use tqdm.write to keep the progress bar at the bottom
        tqdm.write(f"[Gen {loop_idx}] Mean Metrics -> {score_str}")

    def finalize_run(self, fitness: FitnessData, gen_count: int, elapsed_time: float):
        """
        Saves all results to the already initialized self.folder_path.
        """

        # 1. Visualization
        self.graph_plotter.generate_all_visualizations(fitness)

        # 2. Save Baselines
        self._save_baseline_audio()

        # 3. Process Best Candidate
        best_candidate = self._select_best_candidate(self.models.optimizer.best_candidates)
        best_mixed_audio = self._run_final_inference(best_candidate)

        # 4. Save Results
        sf.write(os.path.join(self.folder_path, "best_candidate.wav"), best_mixed_audio.audio, samplerate=24000)

        self._save_torch_state(best_mixed_audio, best_candidate)
        self._write_run_summary(best_mixed_audio, best_candidate, gen_count, elapsed_time)

        # 5. Notify
        if self.config.notify:
            send_whatsapp_notification()

    # ================= INTERNAL HELPERS =================
    def _save_baseline_audio(self):
        """Saves GT and Target audio if they exist."""
        sf.write(os.path.join(self.folder_path, "ground_truth.wav"), self.audio.audio_gt, samplerate=24000)

        if self.audio.audio_target is not None:
            sf.write(os.path.join(self.folder_path, "target.wav"), self.audio.audio_target, samplerate=24000)

    def _select_best_candidate(self, candidates):
        if not candidates:
            raise ValueError("Candidate list is empty.")

        f = np.array([c.fitness for c in candidates])

        # 1. Identify which candidates satisfy ALL thresholds
        satisfied_mask = np.ones(len(candidates), dtype=bool)

        if self.config.thresholds:
            for i, obj in enumerate(self.config.active_objectives):
                if obj in self.config.thresholds:
                    limit = self.config.thresholds[obj]
                    # Assuming fitness is minimization (Lower is better)
                    satisfied_mask &= (f[:, i] <= limit)

        # 2. Logic: If some satisfy thresholds, only pick from those.
        # Otherwise, pick from everyone (fallback).
        if np.any(satisfied_mask):
            eligible_indices = np.where(satisfied_mask)[0]
            eligible_fitness = f[satisfied_mask]
        else:
            eligible_indices = np.arange(len(candidates))
            eligible_fitness = f

        # 3. From the eligible ones, find the Knee Point (closeness to origin)
        # We normalize here to ensure one objective doesn't dominate the distance
        norms = np.linalg.norm(eligible_fitness, axis=1)
        best_local_idx = np.argmin(norms)

        return candidates[eligible_indices[best_local_idx]]

    def _run_final_inference(self, best_candidate):

        phoneme_count = int(self.audio.input_lengths.item())

        best_vector = torch.from_numpy(best_candidate.solution).to(self.device).float()
        best_vector = best_vector.view(phoneme_count, self.config.size_per_phoneme)
        best_vector = adjustInterpolationVector(best_vector, self.config.random_matrix, self.config.subspace_optimization)

        if self.config.mode in [AttackMode.NOISE_UNTARGETED, AttackMode.TARGETED]:
            h_text_mixed = (1.0 - best_vector) * self.audio.h_text_gt + best_vector * self.audio.h_text_target
        else:
            h_text_mixed = self.audio.h_text_gt + self.config.iv_scalar * best_vector

        with torch.no_grad():
            audio_best = self.models.tts_model.inference_on_embedding(
                self.audio.input_lengths,
                self.audio.text_mask,
                self.audio.h_bert_gt,
                h_text_mixed,
                self.audio.style_vector_acoustic,
                self.audio.style_vector_prosodic
            )

        asr_text = self.models.asr_model.transcribe(audio_best)["text"]

        return BestMixedAudio(
            audio=audio_best,
            text=asr_text,
            h_text=h_text_mixed,
            h_bert=self.audio.h_bert_gt
        )

    def _save_torch_state(self, best_mixed, candidate):
        """
        Saves all tensors required to reconstruct the adversarial audio
        without re-running the optimization.
        """
        state_dict = {
            # 1. Metadata for reference
            "metadata": {
                "attack_mode": self.config.mode.name,
                "text_gt": self.config.text_gt,
                "text_target": self.config.text_target,
                "asr_transcription": best_mixed.text,
                "fitness_scores": dict(zip([obj.name for obj in self.config.active_objectives],
                                           candidate.fitness.tolist()))
            },

            # 2. The Solution (Raw Optimization Result)
            "solution_vector": torch.from_numpy(candidate.solution).float().cpu(),

            # 3. Structural requirements for reconstruction
            "random_matrix": self.config.random_matrix.cpu(),
            "size_per_phoneme": self.config.size_per_phoneme,
            "iv_scalar": self.config.iv_scalar,

            # 4. Model Inputs (AudioData)
            # These are required by the TTS decoder to generate the audio
            "input_lengths": self.audio.input_lengths.cpu(),
            "text_mask": self.audio.text_mask.cpu(),
            "style_vector_acoustic": self.audio.style_vector_acoustic.cpu(),
            "style_vector_prosodic": self.audio.style_vector_prosodic.cpu(),

            # 5. The Mixed Embeddings (The final "Adversarial" embeddings)
            "h_text_mixed": best_mixed.h_text.cpu(),
            "h_bert_mixed": best_mixed.h_bert.cpu()
        }

        save_path = os.path.join(self.folder_path, "reconstruction_pack.pt")
        torch.save(state_dict, save_path)
        print(f"[Log] Torch state saved for reproducibility: {save_path}")

    def _write_run_summary(self, best_mixed, candidate, gen_count, elapsed_time):

        # 1. System Metadata
        os_info = f"{platform.system()} {platform.release()}"
        gpu_info = "CPU Only"
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_info = f"{torch.cuda.get_device_name(0)} ({vram:.2f} GB VRAM)"

        avg_per_gen = elapsed_time / gen_count if gen_count > 0 else 0

        summary_path = os.path.join(self.folder_path, "run_summary.txt")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write(" ADVERSARIAL TTS OPTIMIZATION REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write("--- [1] INPUT DATA ---\n")
            # Assuming these paths are stored in your config or audio_data
            f.write(f"GT Text:      {self.config.text_gt}\n")
            f.write(f"Target Text:  {self.config.text_target if self.config.text_target else '[NONE]'}\n")

            f.write("\n--- [2] CLI ARGUMENTS & CONFIG ---\n")
            f.write(f"Attack Mode:       {self.config.mode.name}\n")
            f.write(f"Objectives:        {', '.join([obj.name for obj in self.config.active_objectives])}\n")
            f.write(f"Population Size:   {self.config.pop_size}\n")
            f.write(f"Size Per Phoneme:  {self.config.size_per_phoneme}\n")
            f.write(f"IV Scalar:         {self.config.iv_scalar}\n")
            f.write(f"Subspace Opt:      {self.config.subspace_optimization}\n")

            if self.config.thresholds:
                t_str = ", ".join([f"{k.name} <= {v}" for k, v in self.config.thresholds.items()])
                f.write(f"Early Stopping:    {t_str}\n")
            else:
                f.write(f"Early Stopping:    Off (Ran full duration)\n")

            f.write("\n--- [3] PERFORMANCE & HARDWARE ---\n")
            f.write(f"Hardware:          {gpu_info}\n")
            f.write(f"OS/CPU:            {os_info} | {platform.processor()}\n")
            f.write(f"Gens Completed:    {gen_count}\n")
            f.write(f"Total Time:        {elapsed_time:.2f}s\n")
            f.write(f"Efficiency:        {avg_per_gen:.2f}s per generation\n")

            f.write("\n--- [4] BEST CANDIDATE RESULTS ---\n")
            f.write(f"Selection Metric:  Euclidean Distance to Origin (Knee Point)\n")
            f.write(f"Generation Found:  {getattr(candidate, 'generation', 'Unknown')}\n")
            f.write("-" * 30 + "\n")

            # Detailed Fitness Scores
            for obj, score in zip(self.config.active_objectives, candidate.fitness):
                f.write(f"  {obj.name:<15}: {float(score):.8f}\n")

            f.write("-" * 30 + "\n")
            f.write(f"Final Transcription: \"{best_mixed.text}\"\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write(" END OF REPORT\n")
            f.write("=" * 50 + "\n")

