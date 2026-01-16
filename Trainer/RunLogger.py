import os
import datetime
import torch
import numpy as np
import soundfile as sf
import pandas as pd

# Local Imports
from Datastructures.dataclass import BestMixedAudio


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
        if isinstance(audio_gt, torch.Tensor): audio_gt = audio_gt.detach().cpu().numpy().squeeze()
        sf.write(os.path.join(self.folder_path, "ground_truth.wav"), audio_gt, samplerate=24000)

        if audio_target is not None:
            if isinstance(audio_target, torch.Tensor): audio_target = audio_target.detach().cpu().numpy().squeeze()
            sf.write(os.path.join(self.folder_path, "target.wav"), audio_target, samplerate=24000)

        if isinstance(audio_best_mixed, torch.Tensor): audio_best_mixed = audio_best_mixed.detach().cpu().numpy().squeeze()
        sf.write(os.path.join(self.folder_path, "best_candidate.wav"), audio_best_mixed, samplerate=24000)

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

        interpolation_vector = torch.as_tensor(best_candidate.solution, dtype=torch.float32).view(1, -1, 1).to(self.device)
        current_batch_size, interpolation_vectors, audio_embedding_data = self.vector_manipulator.interpolate(interpolation_vector)
        audio_best = self.tts_model.inference_on_embedding(audio_embedding_data)

        if isinstance(self.asr_model, torch.nn.DataParallel):
            asr_model = self.asr_model.module
        else:
            asr_model = self.asr_model

        # 2. Now call transcribe on the unwrapped model
        asr_text = asr_model.transcribe(audio_best.squeeze())["text"]

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
                "fitness_scores": dict(zip([obj.name for obj in self.active_objectives],
                                           candidate.fitness.tolist()))
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
            "h_text_mixed": audio_embedding_best.h_text.cpu(),
            "h_bert_mixed": audio_embedding_best.h_bert.cpu()
        }

        save_path = os.path.join(self.folder_path, "reconstruction_pack.pt")
        torch.save(state_dict, save_path)
        print("[Log] Torch state saved as reconstruction_pack.pt")

    def setup_output_directory(self):
        """Creates the timestamped output folder and returns its path."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        objectives_str = "_".join([obj.name for obj in self.active_objectives])

        folder_path = os.path.join("outputs/", objectives_str, timestamp)
        os.makedirs(folder_path, exist_ok=True)

        print(f"[Log] Output directory initialized: {folder_path}")
        self.folder_path = folder_path
        return folder_path


