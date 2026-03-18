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

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util as st_util
from huggingface_hub import hf_hub_download

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
_STOPWORDS  = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def _lemmatize_word(word: str) -> str:
    for pos in ("a", "v", "n", "r"):
        lemma = _LEMMATIZER.lemmatize(word, pos=pos)
        if lemma != word:
            return lemma
    return word


def _recompute_set_overlap(gt: str, asr: str) -> float:
    clean_gt  = re.sub(r"[^\w\s]", "", gt.lower())
    gt_words  = {_lemmatize_word(w) for w in set(clean_gt.split()) - _STOPWORDS}
    if not gt_words:
        return 1.0
    clean_asr = re.sub(r"[^\w\s]", "", (asr or "").lower())
    asr_words = {_lemmatize_word(w) for w in set(clean_asr.split()) - _STOPWORDS}
    return min(len(gt_words & asr_words) / len(gt_words), 1.0)

# Local Imports
from helper import save_audio
from Trainer.GraphPlotter import GraphPlotter

_SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"


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

        # Sentence embedding model for semantic distance metric
        self._sentence_model = SentenceTransformer(_SENTENCE_MODEL_NAME, device=device)

        # SQUIM_SUBJECTIVE naturalness model (predicts MOS [1,5] with clean reference)
        from torchaudio.pipelines import SQUIM_SUBJECTIVE
        self._squim_model = SQUIM_SUBJECTIVE.get_model().to(device)
        self._squim_model.eval()

    def _compute_mos(self, audio, ref_audio) -> float:
        """Returns predicted MOS [1, 5] using ref_audio as the clean reference.
        Audio is expected at 16 kHz (StyleTTS2 resamples internally)."""

        def prepare(a):
            t = torch.as_tensor(a, dtype=torch.float32, device=self.device)
            if t.dim() == 1:
                t = t.unsqueeze(0)
            if t.requires_grad:
                t = t.detach()
            return t

        with torch.no_grad():
            mos = self._squim_model(prepare(audio), prepare(ref_audio))
        return round(float(mos[0].item()), 4)

    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between sentence embeddings (0 = different, 1 = identical)."""
        emb = self._sentence_model.encode([text_a, text_b], convert_to_tensor=True)
        return round(float(st_util.cos_sim(emb[0], emb[1]).item()), 6)

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

            # Audio is already at 16 kHz (StyleTTS2 resamples internally)
            audio_16k = audio_tensor

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

    def save_fitness_history_per_individual(self, fitness_history: list[np.ndarray] = None):
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

    def save_fitness_history_per_generation(self, fitness_history: list, archive_history: list):
        """
        Saves a compact per-generation summary to 'fitness_history.csv'.
        Each row: generation, best/mean per objective, hypervolume (2D only), pareto_size.
        """
        from helper import calculate_2d_hypervolume

        obj_names = [obj.name for obj in self.active_objectives]
        num_objectives = len(obj_names)

        rows = []
        for gen_idx, (gen_matrix, archive_snapshot) in enumerate(zip(fitness_history, archive_history)):
            row = {"generation": gen_idx + 1}

            gen_min = gen_matrix.min(axis=0)
            gen_mean = gen_matrix.mean(axis=0)
            for i, name in enumerate(obj_names):
                row[f"best_{name}"] = float(gen_min[i])
                row[f"mean_{name}"] = float(gen_mean[i])

            if num_objectives == 2 and archive_snapshot.shape[1] >= 2:
                hv = calculate_2d_hypervolume(archive_snapshot[:, :2], [1.1, 1.1])
            else:
                hv = float("nan")
            row["hypervolume"] = hv
            row["pareto_size"] = int(len(archive_snapshot))

            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.folder_path, "fitness_history.csv")
        df.to_csv(csv_path, index=False)
        print("[Log] Compact fitness history saved as fitness_history.csv")

    def select_best_candidate(self, candidates, thresholds=None):
        if not candidates:
            raise ValueError("Candidate list is empty.")

        f = np.array([c.fitness for c in candidates])
        indices = np.arange(len(candidates))
        print(f"\n[Log] Candidates on Pareto Front: {len(candidates)}")

        # Step 1: Restrict to candidates that meet all thresholds (if any do).
        # Without this filter, a candidate with near-zero value on one objective but
        # violating the other threshold can win via [min,max] range distortion.
        if thresholds:
            meets_all = np.ones(len(candidates), dtype=bool)
            for i, obj in enumerate(self.active_objectives):
                if obj in thresholds:
                    meets_all &= (f[:, i] <= thresholds[obj])
            if np.any(meets_all):
                print(f"[Log] {np.sum(meets_all)} candidate(s) meet all thresholds — restricting selection.")
                indices = indices[meets_all]
                f = f[meets_all]
            else:
                print(f"[Log] No candidate meets all thresholds — using full Pareto front.")

        # Step 2: Scale each objective dimension.
        # Positive threshold → divide by threshold (threshold maps to 1.0, giving it
        # proportional weight: tighter threshold = higher weight).
        # Zero/missing threshold → divide by observed max (normalises to [0, 1]).
        working_f = f.copy().astype(np.float64)
        for i, obj in enumerate(self.active_objectives):
            t = thresholds.get(obj, 0.0) if thresholds else 0.0
            if t > 0.0:
                working_f[:, i] /= t
            else:
                col_max = working_f[:, i].max()
                working_f[:, i] /= col_max if col_max > 0 else 1.0

        # Step 3: Pick the candidate closest to the origin using the L3 norm.
        # L3 penalises large coordinates more than L2, rewarding both smallness and
        # balance: (0.3, 0.3) beats (0.1, 0.9) because 0.3³+0.3³ < 0.1³+0.9³.
        # No [min,max] re-normalisation here — that step would be distorted by
        # extreme Pareto corners (e.g. IV≈0 solutions) and undo the threshold scaling.
        distances = np.linalg.norm(working_f, ord=3, axis=1)
        best_local_idx = np.argmin(distances)
        selected = candidates[indices[best_local_idx]]
        print(f"[Log] Selected Candidate Fitness: {selected.fitness.tolist()}")
        return selected

    def run_final_inference(self, best_candidate, batch_size: int = 1):
        audio_best = best_candidate.data[0].unsqueeze(0).to(self.device)

        # Use the transcription stored during optimization (same batch context → deterministic).
        # Fall back to re-inference only for candidates created before this fix.
        if best_candidate.data is not None and len(best_candidate.data) > 1 and best_candidate.data[1]:
            return audio_best, best_candidate.data[1]

        asr_model = self.asr_model.module if isinstance(self.asr_model, torch.nn.DataParallel) else self.asr_model
        audio_batch = audio_best.expand(batch_size, -1)
        asr_texts, _ = asr_model.inference(audio_batch)
        asr_text = asr_texts[0] if isinstance(asr_texts, list) else asr_texts

        return audio_best, asr_text

    def save_torch_state(self, text_best, candidate, config_data):
        """
        Saves all tensors required to reconstruct the adversarial audio
        without re-running the optimization.
        """
        input_length = int(self.vector_manipulator.audio_embedding_gt.input_length.detach().cpu().item())
        size_per_phoneme = self.vector_manipulator.config_data.size_per_phoneme
        interpolation_vector = torch.as_tensor(
            candidate.solution, dtype=torch.float32
        ).view(1, input_length, size_per_phoneme).to(self.device)
        _, _, audio_embedding_best = self.vector_manipulator.interpolate(interpolation_vector)

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

    def setup_objective_directory(self):
        """Creates the timestamped output folder and returns its path."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        objectives_str = "_".join([obj.name for obj in self.active_objectives])

        folder_path = os.path.join("outputs/", objectives_str, timestamp)
        os.makedirs(folder_path, exist_ok=True)

        print(f"[Log] Output directory initialized: {folder_path}")
        return folder_path

    def setup_multi_sentence_directory(self, sentence_id: int, run_id: int, run_timestamp: str, base_path: str = "outputs/results"):
        """Creates the structured results folder for a Harvard sentences run."""
        folder_path = os.path.join(base_path, run_timestamp, f"sentence_{sentence_id:03d}", f"run_{run_id}")
        os.makedirs(folder_path, exist_ok=True)
        print(f"[Log] Results directory initialized: {folder_path}")
        return folder_path

    def save_json_summary(
        self,
        text_best: str,
        best_candidate,
        optimizer,
        config_data,
        generation_count: int,
        elapsed_time_total: float,
        num_generations: int = None,
        sentence_id: int = None,
        run_id: int = None,
        run_timestamp: str = None,
        generation_found: int = None,
        seed_target: bool = False,
        seed_gt: bool = False,
        target_asr_text: str = "",
        min_generations: int = 0,
        gt_rms: float = None,
        target_rms: float = None,
        gt_asr_text: str = "",
        utmos_best: float = None,
        utmos_gt: float = None,
    ) -> dict:
        gpu_info = "CPU Only"
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_info = f"{torch.cuda.get_device_name(0)} ({vram:.2f} GB VRAM)"

        avg_per_gen = elapsed_time_total / generation_count if generation_count > 0 else 0
        fitness_names = [obj.name for obj in self.active_objectives]
        fitness_dict = dict(zip(fitness_names, [float(v) for v in best_candidate.fitness]))
        thresholds_dict = {k.name: v for k, v in config_data.thresholds.items()} if config_data.thresholds else {}
        success = all(
            best_candidate.fitness[i] <= config_data.thresholds.get(obj, float("inf"))
            for i, obj in enumerate(self.active_objectives)
            if obj in config_data.thresholds
        )

        summary = {
            "metadata": {
                "run_timestamp": run_timestamp,
                "sentence_id": sentence_id,
                "run_id": run_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "hardware": gpu_info,
                "os": f"{platform.system()} {platform.release()}",
                "cpu": platform.processor(),
            },
            "text_data": {
                "ground_truth_text": config_data.text_gt,
                "gt_transcription": gt_asr_text,
                "target_text": config_data.text_target if config_data.mode.name == "TARGETED" else target_asr_text,
                "asr_transcription": text_best,
                "semantic_similarity": self._semantic_similarity(config_data.text_gt, text_best),
            },
            "success_metrics": {
                "success": bool(success),
                "fitness_scores": fitness_dict,
                "thresholds": thresholds_dict,
            },
            "efficiency_metrics": {
                "generation_count": generation_count,
                "elapsed_time_seconds": round(elapsed_time_total, 2),
                "avg_time_per_generation": round(avg_per_gen, 2),
            },
            "algorithm_parameters": {
                "attack_mode": config_data.mode.name,
                "objectives": fitness_names,
                "pop_size": config_data.pop_size,
                "num_generations": num_generations,
                "size_per_phoneme": config_data.size_per_phoneme,
                "iv_scalar": config_data.iv_scalar,
                "subspace_optimization": config_data.subspace_optimization,
                "num_rms_candidates": getattr(config_data, "num_rms_candidates", 1),
                "seed_target": seed_target,
                "seed_gt": seed_gt,
                "min_generations": min_generations,
                "gt_rms": round(gt_rms, 6) if gt_rms is not None else None,
                "target_rms": round(target_rms, 6) if target_rms is not None else None,
            },
            "final_solution": {
                "generation_found": generation_found,
                "fitness_scores": fitness_dict,
            },
            "pareto_front": [c.fitness.tolist() for c in optimizer.best_candidates],
            "file_paths": {
                "best_mixed_audio": "best_mixed.wav",
                "ground_truth_audio": "ground_truth.wav",
                "fitness_history_csv": "fitness_history.csv",
            },
            "naturalness_scores": {
                "utmos_best": utmos_best,
                "utmos_gt":   utmos_gt,
            },
        }

        save_path = os.path.join(self.folder_path, "run_summary.json")
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Log] Run summary saved (sentence={sentence_id}, run={run_id})")
        return summary

    def save_results_run(
        self,
        optimizer,
        fitness_data: list,
        archive_data: list,
        generation_count: int,
        elapsed_time_total: float,
        audio_gt,
        audio_target,
        config_data,
        folder_path: str,
        num_generations: int = None,
        sentence_id: int = None,
        run_id: int = None,
        run_timestamp: str = None,
        save_torch_state: bool = False,
        save_spectrograms: bool = False,
        save_graphs: bool = False,
        generation_found: int = None,
        seed_target: bool = False,
        seed_gt: bool = False,
        min_generations: int = 0,
        gt_rms: float = None,
        target_rms: float = None,
    ) -> dict:
        os.makedirs(folder_path, exist_ok=True)
        self.folder_path = folder_path

        best_candidate = self.select_best_candidate(optimizer.best_candidates, config_data.thresholds)
        audio_best, text_best = self.run_final_inference(best_candidate, batch_size=min(config_data.batch_size, config_data.pop_size))

        # Transcribe target audio for non-TARGETED modes (e.g. NOISE_UNTARGETED)
        target_asr_text = ""
        if config_data.mode.name != "TARGETED" and audio_target is not None:
            asr_model = self.asr_model.module if isinstance(self.asr_model, torch.nn.DataParallel) else self.asr_model
            target_audio_tensor = audio_target.detach().cpu().unsqueeze(0) if audio_target.dim() == 1 else audio_target.detach().cpu().unsqueeze(0)
            target_asr_texts, _ = asr_model.inference(target_audio_tensor.to(self.device))
            target_asr_text = target_asr_texts[0] if target_asr_texts else ""

        # Transcribe ground truth audio
        asr_model = self.asr_model.module if isinstance(self.asr_model, torch.nn.DataParallel) else self.asr_model
        gt_audio_tensor = audio_gt.detach().cpu().unsqueeze(0) if audio_gt.dim() == 1 else audio_gt.detach().cpu()
        gt_asr_texts, _ = asr_model.inference(gt_audio_tensor.to(self.device))
        gt_asr_text = gt_asr_texts[0] if gt_asr_texts else ""

        utmos_best = self._compute_mos(audio_best, audio_gt)
        utmos_gt   = self._compute_mos(audio_gt,   audio_gt)

        self.save_audios(audio_gt, audio_target, audio_best)
        self.save_fitness_history_per_generation(fitness_data, archive_data)
        summary = self.save_json_summary(text_best, best_candidate, optimizer, config_data, generation_count, elapsed_time_total, num_generations, sentence_id, run_id, run_timestamp, generation_found=generation_found, seed_target=seed_target, seed_gt=seed_gt, target_asr_text=target_asr_text, min_generations=min_generations, gt_rms=gt_rms, target_rms=target_rms, gt_asr_text=gt_asr_text, utmos_best=utmos_best, utmos_gt=utmos_gt)

        if save_torch_state:
            self.save_torch_state(text_best, best_candidate, config_data)

        if save_spectrograms:
            self.save_spectrograms(audio_gt, audio_target, audio_best)

        if save_graphs:
            graph_plotter = GraphPlotter(self.active_objectives, generation_count, self.folder_path, fitness_data, archive_data)
            graph_plotter.generate_hypervolume_graph()
            graph_plotter.generate_pareto_population_graph()
            graph_plotter.generate_mean_population_graph()
            graph_plotter.generate_minimal_population_graph()
            plt.close("all")

        return summary

    # =========================================================================
    # Aggregation
    # =========================================================================

    @staticmethod
    def _flatten_summary(summary: dict) -> dict:
        row = {}

        meta = summary.get("metadata", {})
        row["run_timestamp"] = meta.get("run_timestamp")
        row["sentence_id"] = meta.get("sentence_id")
        row["run_id"] = meta.get("run_id")
        row["timestamp"] = meta.get("timestamp")
        row["hardware"] = meta.get("hardware")

        text = summary.get("text_data", {})
        row["ground_truth_text"] = text.get("ground_truth_text")
        row["gt_transcription"] = text.get("gt_transcription")
        row["target_text"] = text.get("target_text")
        row["asr_transcription"] = text.get("asr_transcription")
        row["semantic_similarity"] = text.get("semantic_similarity")

        success = summary.get("success_metrics", {})
        row["success"] = success.get("success")
        for obj, score in success.get("fitness_scores", {}).items():
            row[f"score_{obj}"] = score
        for obj, threshold in success.get("thresholds", {}).items():
            row[f"threshold_{obj}"] = threshold

        stored_so = success.get("fitness_scores", {}).get("SET_OVERLAP")
        recomp_so = _recompute_set_overlap(
            text.get("ground_truth_text", ""),
            text.get("asr_transcription", ""),
        )
        row["set_overlap_recomputed"] = recomp_so
        row["set_overlap_mismatch"]   = (
            stored_so is not None and abs(stored_so - recomp_so) > 0.05
        )

        eff = summary.get("efficiency_metrics", {})
        row["generation_count"] = eff.get("generation_count")
        row["elapsed_time_seconds"] = eff.get("elapsed_time_seconds")
        row["avg_time_per_generation"] = eff.get("avg_time_per_generation")

        algo = summary.get("algorithm_parameters", {})
        row["attack_mode"] = algo.get("attack_mode")
        row["objectives"] = ",".join(algo.get("objectives", []))
        row["pop_size"] = algo.get("pop_size")
        row["num_generations"] = algo.get("num_generations")
        row["size_per_phoneme"] = algo.get("size_per_phoneme")
        row["iv_scalar"] = algo.get("iv_scalar")
        row["subspace_optimization"] = algo.get("subspace_optimization")

        sol = summary.get("final_solution", {})
        row["generation_found"] = sol.get("generation_found")
        row["pareto_front_size"] = len(summary.get("pareto_front", []))

        nat = summary.get("naturalness_scores", {})
        row["utmos_best"] = nat.get("utmos_best")
        row["utmos_gt"]   = nat.get("utmos_gt")

        return row

    @staticmethod
    def aggregate_results(summaries: list, output_dir: str = "outputs"):
        """
        Aggregate a list of run_summary dicts into all_results.json and all_results.csv.

        Args:
            summaries: List of dicts as returned by save_results_run.
            output_dir: Directory to write output files into.
        """
        import csv as _csv

        if not summaries:
            print("[Aggregate] No summaries to aggregate.")
            return []

        all_rows = [RunLogger._flatten_summary(s) for s in summaries]

        os.makedirs(output_dir, exist_ok=True)

        json_out = os.path.join(output_dir, "all_results.json")
        with open(json_out, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"[Aggregate] Saved {json_out}")

        fieldnames = list(dict.fromkeys(k for row in all_rows for k in row))
        csv_out = os.path.join(output_dir, "all_results.csv")
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[Aggregate] Saved {csv_out}")

        total = len(all_rows)
        successes = sum(1 for r in all_rows if r.get("success"))
        sentences = sorted(set(r["sentence_id"] for r in all_rows if r["sentence_id"] is not None))
        print(f"[Aggregate] Total runs: {total} | Successful: {successes} ({100 * successes / total:.1f}%)")
        print(f"[Aggregate] Sentences: {len(sentences)} ({min(sentences)} – {max(sentences)})")

        return all_rows
