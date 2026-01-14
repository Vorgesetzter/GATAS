"""
Model loader and environment initialization for adversarial TTS.

This module handles:
1. Argument parsing
2. Configuration parsing and validation
3. Required model loading (TTS, ASR)
4. Audio data generation (GT, target)
5. Optimizer initialization
6. ObjectiveManager creation (objectives lazy-load their own models)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import whisper
import warnings
import numpy as np

# Local imports
from Models.styletts2 import StyleTTS2
from helper import addNumbersPattern

# Import dataclasses and enums
from Datastructures.dataclass import ModelData, ConfigData, AudioData
from Datastructures.enum import FitnessObjective, AttackMode

# Import ObjectiveManager and registry
from Objectives.ObjectiveManager import ObjectiveManager
from Objectives.base import BaseObjective

class EnvironmentLoader:
    def __init__(self, device: str):
        self.device = device

        # Ensure all objectives are registered, then get their order
        self.objective_order: list[FitnessObjective] = BaseObjective.get_all_registered_enums()

    def initialize(self, args):
        """
        Entry point to setup the full experimental environment.
        Parses arguments and initializes all components.

        Returns: (config, model_data, audio_data, embedding_data, objective_manager)
        """

        # 1. Parse arguments and create ConfigData
        config_data = self._load_configuration(args)
        config_data.print_summary()

        # 2. Models
        model_data = self._load_required_models(config_data.multi_gpu)

        # 3. Audio Data
        audio_data = self._generate_audio_data(config_data, model_data.tts_model)

        # 4. Initialize Objective Manager
        objective_manager = ObjectiveManager(
            config=config_data,
            model_data=model_data,
            device=self.device,
            audio_data=audio_data
        )

        # 5. Load model for each objective and compute embeddings
        objective_manager.initialize_objectives()

        return config_data, model_data, audio_data, objective_manager

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _load_configuration(self, args) -> ConfigData:
        """Parse command-line arguments and create validated ConfigData.

        Args:
            args_override: Optional list of CLI args. Pass [] for defaults, None for sys.argv.
        """

        random_matrix = torch.from_numpy(
            np.random.rand(args.size_per_phoneme, 512)
        ).to(self.device).float()

        # Validate AttackMode Enum
        try:
            mode = AttackMode[args.mode]
        except KeyError:
            raise ValueError(f"Invalid mode '{args.mode}'. Available modes: {[m.name for m in AttackMode]}")

        # Validate Objective Enums
        active_objectives_raw = set()
        for obj_name in args.ACTIVE_OBJECTIVES:
            try:
                active_objectives_raw.add(FitnessObjective[obj_name])
            except KeyError:
                raise ValueError(f"'{obj_name}' invalid objective.")

        if not active_objectives_raw:
            raise ValueError("Error: No valid active_objectives selected.")

        # Set Objectives in correct order
        active_objectives = [obj for obj in self.objective_order if obj in active_objectives_raw]

        # Parse thresholds
        thresholds = {}
        if args.thresholds:
            for t in args.thresholds:
                try:
                    key_str, val_str = t.split("=")
                    obj_enum = FitnessObjective[key_str.strip()]
                    thresholds[obj_enum] = float(val_str.strip())
                except Exception as e:
                    raise ValueError(f"Error parsing threshold '{t}': {e}")

        # Set batch size (Set to pop_size if pop_size < batch_size or batch_size <= 0)
        batch_size = min(args.batch_size, args.pop_size) if args.batch_size > 0 else args.pop_size

        # Multi-GPU validation
        multi_gpu = args.multi_gpu
        if multi_gpu and torch.cuda.device_count() <= 1:
            warnings.warn("Multi-GPU enabled but only 1 GPU available. Disabling.")
            multi_gpu = False

        return ConfigData(
            text_gt=args.ground_truth_text,
            text_target=args.target_text,
            num_generations=args.num_generations,
            pop_size=args.pop_size,
            loop_count=args.loop_count,
            iv_scalar=args.iv_scalar,
            size_per_phoneme=args.size_per_phoneme,
            batch_size=batch_size,
            notify=args.notify,
            mode=mode,
            active_objectives=active_objectives,
            thresholds=thresholds,
            subspace_optimization=args.subspace_optimization,
            random_matrix=random_matrix,
            multi_gpu=multi_gpu,
        )

    def _load_required_models(self, multi_gpu: bool = False):
        print("Loading TTS Model (StyleTTS2)...")
        tts = StyleTTS2()
        tts.load_models()
        tts.load_checkpoints()
        tts.sample_diffusion()

        print("Loading ASR Model (Whisper)...")
        asr = whisper.load_model("tiny", device=self.device)

        # Enable multi-GPU for ASR if requested
        if multi_gpu:
            tts.enable_multi_gpu()
            asr = nn.DataParallel(asr)

        return ModelData(tts_model=tts, asr_model=asr)

    def _generate_audio_data(self, config, tts):
        noise = torch.randn(1, 1, 256).to(self.device)

        # Extract embeddings
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
        style_ac_gt, style_pro_gt = tts.compute_style_vector(noise, h_bert_raw_gt, embedding_scale=1, diffusion_steps=5)
        style_ac_target, style_pro_target = tts.compute_style_vector(noise, h_bert_raw_target, embedding_scale=1, diffusion_steps=5)

        # Run inference for ground-truth and target
        audio_gt = tts.inference_on_embedding(input_lengths, text_mask, h_bert_gt, h_text_gt, style_ac_gt, style_pro_gt)
        audio_target = tts.inference_on_embedding(input_lengths, text_mask, h_bert_target, h_text_target, style_ac_target, style_pro_target)
        
        return AudioData(
            audio_gt, audio_target,
            h_text_gt, h_text_target,
            h_bert_raw_gt, h_bert_raw_target,
            h_bert_gt, h_bert_target,
            input_lengths, text_mask,
            style_ac_gt, style_pro_gt,
        )