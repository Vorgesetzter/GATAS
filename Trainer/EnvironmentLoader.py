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
import whisper
import numpy as np

# Local imports
from Models.styletts2 import StyleTTS2
from Trainer.VectorManipulator import add_numbers_pattern, generate_similar_noise

# Import dataclasses and enums
from Datastructures.dataclass import ModelData, ConfigData, AudioEmbeddingData
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
        config_data = self.load_configuration(args)
        config_data.print_summary()

        # 2. Models
        tts_model, asr_model = self.load_required_models()

        # 3. Audio Data
        audio_gt, audio_target, audio_embedding_gt, audio_embedding_target = self.generate_audio_data(config_data.mode, config_data.text_gt, config_data.text_target, tts_model)

        # 4. Initialize Objective Manager
        objective_manager = ObjectiveManager(
            model_data=ModelData(tts_model=tts_model, asr_model=asr_model),
            device=self.device,
            text_gt=config_data.text_gt,
            text_target=config_data.text_target,
            mode=config_data.mode,
            audio_gt=audio_gt,
            style_vector_acoustic=audio_embedding_gt.style_vector_acoustic,
            style_vector_prosodic=audio_embedding_gt.style_vector_prosodic
        )

        # 5. Load model for each objective and compute embeddings
        objective_manager.initialize_objectives(config_data.active_objectives)

        return config_data, tts_model, asr_model, audio_gt, audio_target, audio_embedding_gt, audio_embedding_target, objective_manager

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def load_configuration(self, args) -> ConfigData:
        """Parse command-line arguments and create validated ConfigData.

        Args:
            args: Optional list of CLI args. Pass [] for defaults, None for sys.argv.
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
        )

    def load_required_models(self):
        print("Loading TTS Model (StyleTTS2)...")
        tts = StyleTTS2()

        print("Loading ASR Model (Whisper)...")
        asr = whisper.load_model("tiny", device=self.device)

        return tts, asr

    def generate_audio_data(self, mode: AttackMode, text_gt: str, text_target: str, tts: StyleTTS2):
        """Generate audio data for ground-truth and target texts."""
        noise = torch.randn(1, 1, 256).to(self.device)

        if mode is AttackMode.TARGETED:
            tokens_gt, tokens_target = add_numbers_pattern(
                tts.preprocess_text(text_gt),
                tts.preprocess_text(text_target),
                [16, 4]
            )
            audio_embedding_data_gt = tts.extract_embeddings(tokens_gt, noise)
            audio_embedding_data_target = tts.extract_embeddings(tokens_target, noise)
        else:
            audio_embedding_data_gt = tts.extract_embeddings(tts.preprocess_text(text_gt), noise)

            # Random normalized embeddings for untargeted modes
            h_bert_target = generate_similar_noise(audio_embedding_data_gt.h_bert)
            h_text_target = generate_similar_noise(audio_embedding_data_gt.h_text)
            style_ac_target = generate_similar_noise(audio_embedding_data_gt.style_vector_acoustic)
            style_pro_target = generate_similar_noise(audio_embedding_data_gt.style_vector_prosodic)

            audio_embedding_data_target = AudioEmbeddingData(audio_embedding_data_gt.input_length, audio_embedding_data_gt.text_mask, h_bert_target, h_text_target, style_ac_target, style_pro_target)

        # Run inference for ground-truth and target
        audio_gt = tts.inference_on_embedding(audio_embedding_data_gt).flatten()
        audio_target = tts.inference_on_embedding(audio_embedding_data_target).flatten()

        return audio_gt, audio_target, audio_embedding_data_gt, audio_embedding_data_target