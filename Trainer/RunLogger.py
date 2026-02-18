import os
import json
import datetime
import platform
import torch
import numpy as np
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import whisper

# Local Imports
from Datastructures.dataclass import BestMixedAudio
from helper import save_audio


def get_pareto_mask(fitness_matrix):
    """
    Returns a boolean mask of shape (N,) where True indicates the row is
    non-dominated (Pareto efficient).
    """
    population_size = fitness_matrix.shape[0]
    is_efficient = np.ones(population_size, dtype=bool)

    for i in range(population_size):
        if is_efficient[i]:
            current_candidate = fitness_matrix[i]
            all_better_or_equal = np.all(fitness_matrix <= current_candidate, axis=1)
            any_strictly_better = np.any(fitness_matrix < current_candidate, axis=1)

            dominators = all_better_or_equal & any_strictly_better

            if np.any(dominators):
                is_efficient[i] = False

    return is_efficient

class RunLogger:
    def __init__(self, active_objectives, tts_model, asr_model, vector_manipulator, device: str):
        """
        Initializes the logger with specific run results.
        """
        self.active_objectives = active_objectives
        self.tts_model = tts_model
        self.asr_model = asr_model
        self.vector_manipulator = vector_manipulator
        self.device = device

        # Initialize Directory
        self.folder_path = None

    def save_audios(self, audio_gt, audio_target, audio_best_mixed):

        save_audio(audio_gt, os.path.join(self.folder_path, "ground_truth.wav"))

        if audio_target is not None:
            save_audio(audio_target, os.path.join(self.folder_path, "target.wav"))

        save_audio(audio_best_mixed, os.path.join(self.folder_path, "best_mixed.wav"))

    def save_spectrograms(self, audio_gt, audio_target, audio_best_mixed):
        """
        Generates and saves mel spectrograms using Whisper's exact configuration.
        This shows what the ASR model actually "sees" during inference.

        Args:
            audio_gt: Ground truth audio tensor
            audio_target: Target audio tensor (can be None)
            audio_best_mixed: Best mixed/adversarial audio tensor
        """
        def generate_whisper_spectrogram(audio_tensor, title, filename, return_spec=False):
            """
            Helper function to generate and save a spectrogram using Whisper's parameters.
            Uses Whisper's exact mel spectrogram settings but skips padding for cleaner visualization.
            """
            # Ensure tensor format
            if not isinstance(audio_tensor, torch.Tensor):
                audio_tensor = torch.from_numpy(audio_tensor)

            # Move to CPU and ensure correct shape (batch_size, samples)
            audio_tensor = audio_tensor.detach().cpu()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 3:
                audio_tensor = audio_tensor.squeeze(1)

            # Resample to 16kHz (Whisper's sample rate)
            import torchaudio.functional as F
            audio_16k = F.resample(audio_tensor, 24000, 16000)

            # Generate mel spectrogram using Whisper's function WITHOUT padding
            # This uses Whisper's parameters: n_fft=400, hop_length=160, n_mels=80
            mel_spec = whisper.log_mel_spectrogram(audio_16k, n_mels=80).squeeze(0).numpy()

            # Create figure
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                mel_spec,
                sr=16000,
                hop_length=160,
                x_axis='time',
                y_axis='mel',
                cmap='viridis'
            )
            plt.colorbar(format='%+2.0f', label='Log Magnitude')
            plt.title(f'{title} (Whisper\'s Parameters)')
            plt.tight_layout()

            # Save figure
            save_path = os.path.join(self.folder_path, filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            if return_spec:
                return mel_spec

        # Generate spectrograms for each audio file
        mel_spec_gt = generate_whisper_spectrogram(
            audio_gt,
            'Ground Truth Spectrogram',
            'ground_truth_spectrogram.png',
            return_spec=True
        )

        if audio_target is not None:
            generate_whisper_spectrogram(
                audio_target,
                'Target Spectrogram',
                'target_spectrogram.png'
            )

        mel_spec_mixed = generate_whisper_spectrogram(
            audio_best_mixed,
            'Best Mixed Spectrogram',
            'best_mixed_spectrogram.png',
            return_spec=True
        )

        # Generate difference spectrogram (what changed)
        # Spectrograms should already be same shape due to pad_or_trim
        mel_spec_diff = mel_spec_mixed - mel_spec_gt

        # Plot difference with diverging colormap
        plt.figure(figsize=(10, 4))

        # Calculate symmetric color limits for proper diverging colormap
        vmax = np.max(np.abs(mel_spec_diff))
        vmin = -vmax

        librosa.display.specshow(
            mel_spec_diff,
            sr=16000,
            hop_length=160,
            x_axis='time',
            y_axis='mel',
            cmap='RdBu_r',  # Red = increase, Blue = decrease
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(format='%+2.0f', label='Log Magnitude Difference (Mixed - GT)')
        plt.title('Difference Spectrogram (Adversarial Perturbations)')
        plt.tight_layout()

        save_path = os.path.join(self.folder_path, 'difference_spectrogram.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print("[Log] Spectrograms saved successfully (using Whisper's configuration)")

    def save_fitness_history(self, fitness_history: list[np.ndarray] = None):
        """
        Saves the complete history of every individual to 'fitness_history.csv'.
        """
        obj_names = [obj.name for obj in self.active_objectives]

        # Collect all generations into a single list of DataFrames
        raw_list = []

        for gen_idx, gen_matrix in enumerate(fitness_history):
            # gen_matrix shape: (pop_size, num_objectives)

            # Create DataFrame for this generation
            df_gen = pd.DataFrame(gen_matrix, columns=obj_names)
            df_gen["Generation"] = gen_idx + 1
            df_gen["Individual_ID"] = df_gen.index  # ID 0 to 99

            raw_list.append(df_gen)

        # Concatenate everything into one massive table
        df_raw = pd.concat(raw_list, ignore_index=True)

        # Reorder columns nicely: [Generation, ID, WER, PESQ, ...]
        cols = ["Generation", "Individual_ID"] + obj_names
        df_raw = df_raw[cols]

        # Save Single Source of Truth
        csv_path = os.path.join(self.folder_path, "fitness_history.csv")
        df_raw.to_csv(csv_path, index=False)

        print("[Log] Full fitness history saved as fitness_history.csv")

    def select_best_candidate(self, candidates, thresholds=None):
        if not candidates:
            raise ValueError("Candidate list is empty.")

        f = np.array([c.fitness for c in candidates])

        print(f"\n[Log] Candidates on Pareto Front: {len(candidates)}")

        # 1. Threshold filtering
        satisfied_mask = np.ones(len(candidates), dtype=bool)

        if thresholds:
            for i, obj in enumerate(self.active_objectives):
                if obj in thresholds:
                    limit = thresholds[obj]
                    satisfied_mask &= (f[:, i] <= limit)

        # 2. Logic: If some satisfy thresholds, only pick from those.
        # Otherwise, pick from everyone (fallback).
        if np.any(satisfied_mask):
            final_indices = np.where(satisfied_mask)[0]
            final_fitness = f[satisfied_mask]
            print(f"[Log] Using {len(final_indices)} candidate(s) that meet all thresholds")
        else:
            final_indices = np.arange(len(candidates))
            final_fitness = f
            print(f"[Log] No candidate met thresholds. Preceding with all candidates.")

        # --- 3. Knee Point Selection (Balance) ---
        # Normalize to [0,1] range so large metrics (like PESQ) don't dominate small ones (like WER)
        mins = final_fitness.min(axis=0)
        maxs = final_fitness.max(axis=0)
        ranges = maxs - mins

        ranges[ranges == 0] = 1.0  # Avoid divide by zero

        normalized_fitness = (final_fitness - mins) / ranges

        # Calculate Euclidean distance to the ideal point (0,0,0)
        distances = np.linalg.norm(normalized_fitness, axis=1)

        # Retrieve the original candidate object
        best_local_idx = np.argmin(distances)
        best_global_idx = final_indices[best_local_idx]
        selected = candidates[best_global_idx]

        print(f"[Log] Selected Candidate Fitness: {selected.fitness.tolist()}")

        return selected

    def run_final_inference(self, best_candidate):
        # Reshape solution to original optimizer shape [1, input_length, size_per_phoneme]
        input_length = int(self.vector_manipulator.audio_embedding_gt.input_length.detach().cpu().item())
        size_per_phoneme = self.vector_manipulator.config_data.size_per_phoneme

        interpolation_vector = torch.as_tensor(
            best_candidate.solution, dtype=torch.float32
        ).view(1, input_length, size_per_phoneme).to(self.device)
        current_batch_size, interpolation_vectors, audio_embedding_data = self.vector_manipulator.interpolate(interpolation_vector)
        audio_best = self.tts_model.inference_on_embedding(audio_embedding_data)

        if isinstance(self.asr_model, torch.nn.DataParallel):
            asr_model = self.asr_model.module
        else:
            asr_model = self.asr_model

        # 2. Now call transcribe on the unwrapped model (keep batch dim for consistency with training)
        asr_texts, _ = asr_model.inference(audio_best)
        asr_text = asr_texts[0] if isinstance(asr_texts, list) else asr_texts

        return audio_best, asr_text, audio_embedding_data

    def save_torch_state(self, text_best, audio_embedding_best, candidate, config_data):
        """
        Saves all tensors required to reconstruct the adversarial audio
        without re-running the optimization.
        """
        state_dict = {
            # 1. Metadata for reference
            "metadata": {
                "attack_mode": config_data.mode.name,
                "text_gt": config_data.text_gt,
                "text_target": config_data.text_target,
                "asr_transcription": text_best,
                "fitness_scores": dict(zip([obj.name for obj in self.active_objectives], candidate.fitness.tolist()))
            },

            # 2. The Solution (Raw Optimization Result)
            "solution_vector": torch.from_numpy(candidate.solution).float().cpu(),

            # 3. Structural requirements for reconstruction
            "random_matrix": config_data.random_matrix.cpu(),
            "size_per_phoneme": config_data.size_per_phoneme,
            "iv_scalar": config_data.iv_scalar,

            # 4. Model Inputs (AudioData)
            # These are required by the TTS decoder to generate the audio
            "input_length": audio_embedding_best.input_length.cpu(),
            "text_mask": audio_embedding_best.text_mask.cpu(),
            "style_vector_acoustic": audio_embedding_best.style_vector_acoustic.cpu(),
            "style_vector_prosodic": audio_embedding_best.style_vector_prosodic.cpu(),

            # 5. The Mixed Embeddings (The final "Adversarial" embeddings)
            "h_text_gt": self.vector_manipulator.audio_embedding_gt.h_text.cpu(),
            "h_text_target": self.vector_manipulator.h_text_target.cpu(),
            "h_text_mixed": audio_embedding_best.h_text.cpu(),
        }

        save_path = os.path.join(self.folder_path, "reconstruction_pack.pt")
        torch.save(state_dict, save_path)
        print("[Log] Torch state saved as reconstruction_pack.pt")

    def save_run_summary(self, text_best, candidate, config_data, generation_count, elapsed_time_total):
        gpu_info = "CPU Only"
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_info = f"{torch.cuda.get_device_name(0)} ({vram:.2f} GB VRAM)"

        avg_per_gen = elapsed_time_total / generation_count if generation_count > 0 else 0

        summary = {
            "attack_mode": config_data.mode.name,
            "text_gt": config_data.text_gt,
            "text_target": config_data.text_target,
            "asr_transcription": text_best,
            "objectives": [obj.name for obj in self.active_objectives],
            "fitness_scores": dict(zip([obj.name for obj in self.active_objectives], candidate.fitness.tolist())),
            "best_candidate_generation": getattr(candidate, 'generation', None),
            "generation_count": generation_count,
            "elapsed_time_seconds": round(elapsed_time_total, 2),
            "avg_time_per_generation": round(avg_per_gen, 2),
            "pop_size": config_data.pop_size,
            "size_per_phoneme": config_data.size_per_phoneme,
            "iv_scalar": config_data.iv_scalar,
            "subspace_optimization": config_data.subspace_optimization,
            "thresholds": {k.name: v for k, v in config_data.thresholds.items()} if config_data.thresholds else None,
            "hardware": gpu_info,
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor(),
            "success": all(
                candidate.fitness[i] <= config_data.thresholds.get(obj, float('inf'))
                for i, obj in enumerate(self.active_objectives)
                if obj in config_data.thresholds
            ),
        }

        save_path = os.path.join(self.folder_path, "run_summary.json")
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print("[Log] Run summary saved as run_summary.json")

    def write_run_summary(self, text_best, candidate, config_data, generation_count, elapsed_time_total):
        gpu_info = "CPU Only"
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_info = f"{torch.cuda.get_device_name(0)} ({vram:.2f} GB VRAM)"

        os_info = f"{platform.system()} {platform.release()}"
        avg_per_gen = elapsed_time_total / generation_count if generation_count > 0 else 0

        summary_path = os.path.join(self.folder_path, "run_summary.txt")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write(" ADVERSARIAL TTS OPTIMIZATION REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write("--- [1] INPUT DATA ---\n")
            f.write(f"GT Text:      {config_data.text_gt}\n")
            f.write(f"Target Text:  {config_data.text_target if config_data.text_target else '[NONE]'}\n")

            f.write("\n--- [2] CLI ARGUMENTS & CONFIG ---\n")
            f.write(f"Attack Mode:       {config_data.mode.name}\n")
            f.write(f"Objectives:        {', '.join([obj.name for obj in self.active_objectives])}\n")
            f.write(f"Population Size:   {config_data.pop_size}\n")
            f.write(f"Size Per Phoneme:  {config_data.size_per_phoneme}\n")
            f.write(f"IV Scalar:         {config_data.iv_scalar}\n")
            f.write(f"Subspace Opt:      {config_data.subspace_optimization}\n")

            if config_data.thresholds:
                t_str = ", ".join([f"{k.name} <= {v}" for k, v in config_data.thresholds.items()])
                f.write(f"Early Stopping:    {t_str}\n")
            else:
                f.write(f"Early Stopping:    Off (Ran full duration)\n")

            f.write("\n--- [3] PERFORMANCE & HARDWARE ---\n")
            f.write(f"Hardware:          {gpu_info}\n")
            f.write(f"OS/CPU:            {os_info} | {platform.processor()}\n")
            f.write(f"Gens Completed:    {generation_count}\n")
            f.write(f"Total Time:        {elapsed_time_total:.2f}s\n")
            f.write(f"Efficiency:        {avg_per_gen:.2f}s per generation\n")

            f.write("\n--- [4] BEST CANDIDATE RESULTS ---\n")
            f.write(f"Selection Metric:  Euclidean Distance to Origin (Knee Point)\n")
            f.write(f"Generation Found:  {getattr(candidate, 'generation', 'Unknown')}\n")
            f.write("-" * 30 + "\n")

            for obj, score in zip(self.active_objectives, candidate.fitness):
                f.write(f"  {obj.name:<15}: {float(score):.8f}\n")

            f.write("-" * 30 + "\n")
            f.write(f"Final Transcription: \"{text_best}\"\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write(" END OF REPORT\n")
            f.write("=" * 50 + "\n")

        print("[Log] Run summary saved as run_summary.txt")

    def setup_output_directory(self):
        """Creates the timestamped output folder and returns its path."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        objectives_str = "_".join([obj.name for obj in self.active_objectives])

        folder_path = os.path.join("outputs/", objectives_str, timestamp)
        os.makedirs(folder_path, exist_ok=True)

        print(f"[Log] Output directory initialized: {folder_path}")
        self.folder_path = folder_path
        return folder_path

    def save_all_results(self, optimizer, fitness_data, generation_count, elapsed_time_total,
                        audio_gt, audio_target, config_data):
        """
        Handles all logging, saving, and graph generation for a completed optimization run.

        This is the main entry point for saving results. It orchestrates:
        - Output directory setup
        - Fitness history saving
        - Best candidate selection
        - Final inference
        - Audio and spectrogram saving
        - State saving
        - Graph generation

        Args:
            optimizer: The optimizer with best_candidates
            fitness_data: List of fitness arrays from each generation
            generation_count: Number of generations completed
            elapsed_time_total: Total time elapsed
            audio_gt: Ground truth audio
            audio_target: Target audio (can be None)
            config_data: Configuration data

        Returns:
            tuple: (folder_path, text_best, best_candidate, audio_best)
        """
        # 1. Setup output directory
        folder_path = self.setup_output_directory()

        # 2. Save fitness history
        self.save_fitness_history(fitness_data)

        # 3. Select best candidate
        best_candidate = self.select_best_candidate(optimizer.best_candidates, config_data.thresholds)

        # 4. Run final inference
        audio_best, text_best, audio_embedding_best = self.run_final_inference(best_candidate)

        # 5. Save audio files
        self.save_audios(audio_gt, audio_target, audio_best)

        # 6. Save spectrograms
        self.save_spectrograms(audio_gt, audio_target, audio_best)

        # 7. Save torch state
        self.save_torch_state(text_best, audio_embedding_best, best_candidate, config_data)

        # 8. Save run summary JSON and TXT
        self.save_run_summary(text_best, best_candidate, config_data, generation_count, elapsed_time_total)
        self.write_run_summary(text_best, best_candidate, config_data, generation_count, elapsed_time_total)

        # 9. Generate graphs
        from Trainer.GraphPlotter import GraphPlotter
        graph_plotter = GraphPlotter(self.active_objectives, generation_count, folder_path, fitness_data)
        graph_plotter.generate_hypervolume_graph()
        graph_plotter.generate_pareto_population_graph()
        graph_plotter.generate_mean_population_graph()
        graph_plotter.generate_minimal_population_graph()
        plt.close('all')



