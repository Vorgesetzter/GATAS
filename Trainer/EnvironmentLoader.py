"""
Model loader and environment initialization for adversarial TTS.

This module handles:
1. Argument parsing
2. Configuration parsing and validation
3. Required model loading (TTS, ASR)
4. Audio data generation (GT, target)
5. Objective initialization
"""
from __future__ import annotations

import torch
import numpy as np

# Local imports
from Models.styletts2 import StyleTTS2
from Models.whisper import Whisper
from Trainer.VectorManipulator import add_numbers_pattern, generate_similar_noise

# Import dataclasses and enums
from Datastructures.dataclass import ModelData, ConfigData, AudioEmbeddingData, ModelEmbeddingData
from Objectives.FitnessObjective import FitnessObjective
from Datastructures.enum import AttackMode
from Objectives.base import BaseObjective


class EnvironmentLoader:

    def __init__(self, device: str):
        self.device = device

    def initialize(self, args):
        """
        Entry point to setup the full experimental environment.
        Parses arguments and initializes all components.

        Returns: (config, tts_model, asr_model, audio_gt, audio_target,
                  audio_embedding_gt, audio_embedding_target, objectives)
        """

        # 1. Parse arguments and create ConfigData
        config_data = self.load_configuration(args)
        config_data.print_summary()

        # 2. Models
        tts_model, asr_model = self.load_required_models()

        # 3. Audio Data
        audio_gt, audio_target, audio_embedding_gt, audio_embedding_target = self.generate_audio_data(config_data.mode, config_data.text_gt, config_data.text_target, tts_model)

        # 4. Initialize Objectives
        objectives = self.initialize_objectives(
            active_objectives=config_data.active_objectives,
            model_data=ModelData(tts_model=tts_model, asr_model=asr_model),
            text_gt=config_data.text_gt,
            text_target=config_data.text_target,
            mode=config_data.mode,
            audio_gt=audio_gt,
        )

        return (config_data, tts_model, asr_model, audio_gt, audio_target,
                audio_embedding_gt, audio_embedding_target, objectives)

    def initialize_objectives(
        self,
        active_objectives: list[FitnessObjective],
        model_data: ModelData,
        text_gt: str,
        text_target: str,
        mode: AttackMode,
        audio_gt: torch.Tensor,
    ) -> dict[FitnessObjective, BaseObjective]:
        """
        Initialize objective instances from enum values.

        Returns dict mapping FitnessObjective enum -> objective instance.
        """
        embedding_data = ModelEmbeddingData()
        objectives = {}

        for obj_enum in active_objectives:
            try:
                # obj_enum.value is the objective class
                objective = obj_enum.value(
                    model_data=model_data,
                    device=self.device,
                    embedding_data=embedding_data,
                    text_gt=text_gt,
                    text_target=text_target,
                    mode=mode,
                    audio_gt=audio_gt,
                )
                objectives[obj_enum] = objective
                print(f"Initialized {obj_enum.name} (batching={objective.supports_batching})")
            except Exception as e:
                raise ValueError(f"Failed to initialize {obj_enum.name}: {e}") from e

        return objectives

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def load_configuration(self, args) -> ConfigData:
        """Parse command-line arguments and create validated ConfigData."""

        random_matrix = torch.from_numpy(
            np.random.rand(args.size_per_phoneme, 512)
        ).to(self.device).float()

        # Validate AttackMode Enum
        try:
            mode = AttackMode[args.mode]
        except KeyError:
            raise ValueError(f"Invalid mode '{args.mode}'. Available modes: {[m.name for m in AttackMode]}")

        # Parse combined objectives format (e.g., "PESQ=0.3, WER_GT=0.5")
        active_objectives_raw = set()
        thresholds = {}

        for entry in args.objectives.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" in entry:
                obj_name, val_str = entry.split("=")
                obj_name = obj_name.strip().upper()
                try:
                    obj_enum = FitnessObjective[obj_name]
                    active_objectives_raw.add(obj_enum)
                    thresholds[obj_enum] = float(val_str.strip())
                except KeyError:
                    raise ValueError(f"'{obj_name}' invalid objective.")
                except ValueError:
                    raise ValueError(f"Invalid threshold value '{val_str}' for {obj_name}")
            else:
                # Objective without threshold - just add to active objectives
                obj_name = entry.strip().upper()
                try:
                    obj_enum = FitnessObjective[obj_name]
                    active_objectives_raw.add(obj_enum)
                except KeyError:
                    raise ValueError(f"'{obj_name}' invalid objective.")

        if not active_objectives_raw:
            raise ValueError("Error: No valid objectives specified.")

        # Set Objectives in correct order (enum definition order)
        active_objectives = [obj for obj in FitnessObjective if obj in active_objectives_raw]

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
        tts = StyleTTS2(device=self.device)

        print("Loading ASR Model (Whisper)...")
        asr = Whisper(device=self.device)

        return tts, asr

    def generate_audio_data(self, mode: AttackMode, text_gt: str, text_target: str, tts: StyleTTS2):
        """Generate audio data for ground-truth and target texts."""
        noise = torch.randn(1, 1, 256).to(self.device)

        audio_embedding_data_gt = tts.extract_embeddings(tts.preprocess_text(text_gt), noise)

        if mode is AttackMode.TARGETED:
            tokens_gt, tokens_target = add_numbers_pattern(
                tts.preprocess_text(text_gt),
                tts.preprocess_text(text_target),
                [16, 4]
            )
            audio_embedding_data_gt = tts.extract_embeddings(tokens_gt, noise)
            audio_embedding_data_target = tts.extract_embeddings(tokens_target, noise)
        elif mode is AttackMode.ZERO_UNTARGETED:
            audio_embedding_data_target = AudioEmbeddingData(
                audio_embedding_data_gt.input_length,
                audio_embedding_data_gt.text_mask,
                torch.zeros_like(audio_embedding_data_gt.h_bert),
                torch.zeros_like(audio_embedding_data_gt.h_text),
                torch.zeros_like(audio_embedding_data_gt.style_vector_acoustic),
                torch.zeros_like(audio_embedding_data_gt.style_vector_prosodic),
            )

        elif mode is AttackMode.NEGATION_UNTARGETED:
            audio_embedding_data_target = AudioEmbeddingData(
                audio_embedding_data_gt.input_length,
                audio_embedding_data_gt.text_mask,
                -audio_embedding_data_gt.h_bert,
                -audio_embedding_data_gt.h_text,
                -audio_embedding_data_gt.style_vector_acoustic,
                -audio_embedding_data_gt.style_vector_prosodic,
            )

        else:
            gt_rms = tts.inference_on_embedding(audio_embedding_data_gt).flatten().pow(2).mean().sqrt().item()
            min_rms = 0.1 * gt_rms

            best_embedding = None
            best_rms = -1.0

            for attempt in range(10):
                candidate_embedding = AudioEmbeddingData(
                    audio_embedding_data_gt.input_length,
                    audio_embedding_data_gt.text_mask,
                    generate_similar_noise(audio_embedding_data_gt.h_bert),
                    generate_similar_noise(audio_embedding_data_gt.h_text),
                    generate_similar_noise(audio_embedding_data_gt.style_vector_acoustic),
                    generate_similar_noise(audio_embedding_data_gt.style_vector_prosodic),
                )
                candidate_audio = tts.inference_on_embedding(candidate_embedding).flatten()
                rms = candidate_audio.pow(2).mean().sqrt().item()

                if rms > best_rms:
                    best_rms = rms
                    best_embedding = candidate_embedding

                if rms >= min_rms:
                    print(f"[Log] Target sampled (attempt {attempt + 1}, RMS={rms:.4f}, GT RMS={gt_rms:.4f})")
                    break
            else:
                print(f"[Log] Using best target after 5 attempts (RMS={best_rms:.4f}, threshold={min_rms:.4f})")

            audio_embedding_data_target = best_embedding

        # Run inference for ground-truth and target
        audio_gt = tts.inference_on_embedding(audio_embedding_data_gt).flatten()
        audio_target = tts.inference_on_embedding(audio_embedding_data_target).flatten()

        return audio_gt, audio_target, audio_embedding_data_gt, audio_embedding_data_target
