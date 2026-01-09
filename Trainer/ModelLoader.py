"""
Model loader and environment initialization for adversarial TTS.

This module handles:
1. Configuration parsing and validation
2. Required model loading (TTS, ASR)
3. Audio data generation (GT, target)
4. Optimizer initialization
5. ObjectiveManager creation (objectives lazy-load their own models)
"""

import numpy as np
import torch
import torch.nn as nn
import whisper

# Local imports
from Models.styletts2 import StyleTTS2
from helper import addNumbersPattern

# Import Pymoo components
from Optimizer.pymoo_optimizer import PymooOptimizer
from pymoo.algorithms.moo.nsga2 import NSGA2

# Import dataclasses and enums
from Datastructures.dataclass import ModelData, ConfigData, AudioData, EmbeddingData
from Datastructures.enum import FitnessObjective, AttackMode

# Import ObjectiveManager and registry
from Objectives.manager import ObjectiveManager
from Objectives.registry import ensure_all_registered, get_all_objective_enums


class EnvironmentLoader:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        # Ensure all objectives are registered, then get their order
        ensure_all_registered()
        self.objective_order: list[FitnessObjective] = get_all_objective_enums()

    def initialize(self):
        """
        Entry point to setup the full experimental environment.
        Returns: (config, model_data, audio_data, embedding_data, objective_manager)
        """
        # 1. Configuration
        config_data = self._load_configuration()
        if config_data is None:
            return None, None, None, None, None
        config_data.print_summary()

        # 2. Models
        tts_model, asr_model = self._load_required_models(config_data.multi_gpu)

        # 3. Audio Data
        audio_data = self._generate_audio_data(config_data, tts_model)

        # 4. Optimizer
        optimizer = self._load_optimizer(audio_data, config_data)

        # 5. Containers
        model_data = ModelData(tts_model=tts_model, asr_model=asr_model, optimizer=optimizer)
        embedding_data = EmbeddingData()

        # 6. Objective Manager
        print("\n[INFO] Initializing ObjectiveManager...")
        objective_manager = ObjectiveManager(
            config=config_data,
            model_data=model_data,
            device=self.device,
            embedding_data=embedding_data
        )

        # 7. Pre-compute
        self._precompute_audio_embeddings(objective_manager, audio_data, config_data)

        return config_data, model_data, audio_data, embedding_data, objective_manager

    def _load_configuration(self):
        """Parse and validate configuration."""
        # Subspace optimization setup
        random_matrix = torch.from_numpy(
            np.random.rand(self.args.size_per_phoneme, 512)
        ).to(self.device).float()

        try:
            mode = AttackMode[self.args.mode]
        except KeyError:
            print(f"Invalid mode '{self.args.mode}'. Available modes: {[m.name for m in AttackMode]}")
            return None

        # Parse active objectives
        active_objectives_raw = set()
        for obj_name in self.args.ACTIVE_OBJECTIVES:
            try:
                active_objectives_raw.add(FitnessObjective[obj_name])
            except KeyError:
                print(f"Warning: '{obj_name}' invalid objective.")

        if not active_objectives_raw:
            print("Error: No valid active_objectives selected.")
            return None

        # Maintain global order
        active_objectives = [obj for obj in self.objective_order if obj in active_objectives_raw]

        # Parse thresholds
        thresholds = {}
        if self.args.thresholds:
            for t in self.args.thresholds:
                try:
                    key_str, val_str = t.split("=")
                    obj_enum = FitnessObjective[key_str.strip()]
                    thresholds[obj_enum] = float(val_str.strip())
                except Exception as e:
                    print(f"Error parsing threshold '{t}': {e}")
                    return None

        # Correct batch size
        batch_size = min(self.args.batch_size, self.args.pop_size) if self.args.batch_size > 0 else self.args.pop_size

        # Multi-GPU: auto-enable if multiple GPUs available and not explicitly disabled
        multi_gpu = getattr(self.args, 'multi_gpu', False)
        if multi_gpu and torch.cuda.device_count() <= 1:
            print("[WARNING] multi_gpu enabled but only 1 GPU available. Disabling.")
            multi_gpu = False

        return ConfigData(
            text_gt=self.args.ground_truth_text,
            text_target=self.args.target_text,
            num_generations=self.args.num_generations,
            pop_size=self.args.pop_size,
            loop_count=self.args.loop_count,
            iv_scalar=self.args.iv_scalar,
            size_per_phoneme=self.args.size_per_phoneme,
            batch_size=batch_size,
            notify=self.args.notify,
            mode=mode,
            active_objectives=active_objectives,
            thresholds=thresholds,
            objective_order=self.objective_order,
            diffusion_steps=5,
            embedding_scale=1,
            subspace_optimization=self.args.subspace_optimization,
            random_matrix=random_matrix,
            multi_gpu=multi_gpu,
        )

    def _load_required_models(self, multi_gpu: bool = False):
        print("Loading StyleTTS2...")
        tts = StyleTTS2()
        tts.load_models()
        tts.load_checkpoints()
        tts.sample_diffusion()

        # Enable multi-GPU for TTS if requested
        if multi_gpu:
            tts.enable_multi_gpu()

        print("Loading ASR Model (Whisper)...")
        asr = whisper.load_model("tiny", device=self.device)

        # Enable multi-GPU for ASR if requested
        if multi_gpu:
            gpu_count = torch.cuda.device_count()
            print(f"[INFO] Wrapping Whisper ASR in DataParallel ({gpu_count} GPUs)")
            asr = nn.DataParallel(asr)

        return tts, asr

    def _generate_audio_data(self, config, tts):
        noise = torch.randn(1, 1, 256).to(self.device)

        if config.mode is AttackMode.TARGETED:
            # Text -> Tokens, while adding tokens if necessary
            tokens_gt, tokens_target = addNumbersPattern(
                tts.preprocess_text(config.text_gt),
                tts.preprocess_text(config.text_target),
                [16, 4]
            )
            h_text_gt, h_bert_raw_gt, h_bert_gt, input_lengths, text_mask = tts.extract_embeddings(tokens_gt)
            h_text_target, h_bert_raw_target, h_bert_target, _, _ = tts.extract_embeddings(tokens_target)
        else:
            tokens_gt = tts.preprocess_text(config.text_gt)
            h_text_gt, h_bert_raw_gt, h_bert_gt, input_lengths, text_mask = tts.extract_embeddings(tokens_gt)

            # Random embeddings for untargeted modes
            h_text_target = torch.randn_like(h_text_gt)
            h_text_target /= h_text_target.norm()

            h_bert_raw_target = torch.randn_like(h_bert_raw_gt)
            h_bert_raw_target /= h_bert_raw_target.norm()

            h_bert_target = torch.randn_like(h_bert_gt)
            h_bert_target /= h_bert_target.norm()

        # Generate style vectors
        style_ac_gt, style_pro_gt = tts.compute_style_vector(
            noise, h_bert_raw_gt, config.embedding_scale, config.diffusion_steps
        )
        style_ac_target, style_pro_target = tts.compute_style_vector(
            noise, h_bert_raw_target, config.embedding_scale, config.diffusion_steps
        )

        # Run inference for ground-truth and target
        audio_gt = tts.inference_on_embedding(
            input_lengths, text_mask, h_bert_gt, h_text_gt, style_ac_gt, style_pro_gt
        )
        audio_target = tts.inference_on_embedding(
            input_lengths, text_mask, h_bert_target, h_text_target, style_ac_target, style_pro_target
        )
        
        return AudioData(
            audio_gt, audio_target, h_text_gt, h_text_target,
            h_bert_raw_gt, h_bert_raw_target, h_bert_gt, h_bert_target,
            input_lengths, text_mask, style_ac_gt, style_pro_gt, noise
        )

    def _load_optimizer(self, audio_data, config_data):
        phoneme_count = audio_data.input_lengths.detach().cpu().item()
        return PymooOptimizer(
            bounds=(0, 1),
            algorithm=NSGA2,
            algo_params={"pop_size": config_data.pop_size},
            num_objectives=len(config_data.active_objectives),
            solution_shape=(phoneme_count, config_data.size_per_phoneme),
        )

    def _precompute_audio_embeddings(self, objective_manager, audio_data, config_data):
        wav2vec_objectives = [
            FitnessObjective.WAV2VEC_SIMILAR,
            FitnessObjective.WAV2VEC_DIFFERENT,
            FitnessObjective.WAV2VEC_ASR
        ]

        if not any(obj in config_data.active_objectives for obj in wav2vec_objectives):
            return

        wav2vec_obj = next((objective_manager.get_objective_instance(obj)
                            for obj in wav2vec_objectives if objective_manager.get_objective_instance(obj)), None)

        if wav2vec_obj is None: return

        print("[INFO] Pre-computing Wav2Vec embeddings...")
        with torch.no_grad():
            def get_emb(audio):
                return torch.mean(wav2vec_obj.wav2vec_model(
                    **wav2vec_obj.wav2vec_processor(audio, return_tensors="pt", sampling_rate=16000).to(self.device)
                ).last_hidden_state, dim=1)

            objective_manager.embedding_data.wav2vec_embedding_gt = get_emb(audio_data.audio_gt)

            if config_data.mode is AttackMode.TARGETED:
                objective_manager.embedding_data.wav2vec_embedding_target = get_emb(audio_data.audio_target)
            elif config_data.mode is AttackMode.NOISE_UNTARGETED:
                emb_gt = objective_manager.embedding_data.wav2vec_embedding_gt
                target = torch.randn_like(emb_gt)
                objective_manager.embedding_data.wav2vec_embedding_target = target / target.norm()


# Backward-compatible wrapper functions
def initialize_environment(args, device):
    """
    Initialize the complete environment for adversarial TTS optimization.

    Args:
        args: Parsed command-line arguments
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        Tuple of (config_data, model_data, audio_data, embedding_data, objective_manager)
    """
    loader = EnvironmentLoader(args, device)
    return loader.initialize()


def load_optimizer(audio_data, config_data):
    """Initialize the NSGA-II optimizer."""
    phoneme_count = audio_data.input_lengths.detach().cpu().item()

    return PymooOptimizer(
        bounds=(0, 1),
        algorithm=NSGA2,
        algo_params={"pop_size": config_data.pop_size},
        num_objectives=len(config_data.active_objectives),
        solution_shape=(phoneme_count, config_data.size_per_phoneme),
    )