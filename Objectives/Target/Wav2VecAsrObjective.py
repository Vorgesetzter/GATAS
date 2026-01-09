import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from Objectives.base import BaseObjective
from Datastructures.dataclass import ModelData, StepContext, AudioData
from Datastructures.enum import AttackMode, FitnessObjective


class Wav2VecAsrObjective(BaseObjective):
    """
    Wav2Vec2 audio embedding cosine similarity between mixed audio and
    audio synthesized from ASR text.

    This measures how well the mixed audio sounds like what the ASR heard.

    wav2vec_asr = cos_sim(emb_asr_audio, emb_mixed)
    Values: [-1, 1]
    -1 = Mixed audio very different to ASR-synthesized audio
    1 = Mixed audio same as ASR-synthesized audio

    We convert to fitness: 0 = same (good), 1 = different (bad).

    NOTE: This objective requires TARGETED mode.
    NOTE: This objective does NOT support batching because it needs to
          synthesize audio for each ASR text individually via the TTS model.
    """
    objective_type = FitnessObjective.WAV2VEC_ASR

    def __init__(self, config, model_data: ModelData, device: str = None, embedding_data=None):
        super().__init__(config, model_data)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Validate mode
        if config.mode is AttackMode.UNTARGETED:
            raise ValueError("AttackMode.UNTARGETED incompatible with Wav2VecAsrObjective")

        # Lazy load Wav2Vec2 model if not already loaded
        if self.model_data.wav2vec_model is None:
            print(f"[INFO] Loading Wav2Vec2 Model on {self.device}...")
            self.model_data.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
            model = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base-960h"
            ).to(self.device)
            model.eval()

            # Multi-GPU support
            if self.device == 'cuda' and torch.cuda.device_count() > 1:
                print(f"[INFO] Wav2Vec2 using {torch.cuda.device_count()} GPUs.")
                model = nn.DataParallel(model)

            self.model_data.wav2vec_model = model

        self.wav2vec_model = self.model_data.wav2vec_model
        self.wav2vec_processor = self.model_data.wav2vec_processor

    @property
    def supports_batching(self) -> bool:
        # Cannot batch because TTS synthesis is required per-sample
        return False

    def _calculate_logic(self, context: StepContext, audio_data: AudioData) -> float:
        """Process single sample (TTS synthesis required)."""
        audio_mixed = context.audio_mixed
        asr_text = context.clean_text

        # Ensure audio is numpy for processor
        if isinstance(audio_mixed, torch.Tensor):
            audio_mixed_np = audio_mixed.cpu().numpy()
        else:
            audio_mixed_np = audio_mixed

        # Synthesize audio from ASR text using TTS model
        audio_asr = self.model_data.tts_model.inference(asr_text, audio_data.noise)

        with torch.no_grad():
            # Get embedding for ASR-synthesized audio
            inputs_asr = self.wav2vec_processor(
                audio_asr,
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.device)
            outputs_asr = self.wav2vec_model(**inputs_asr)
            wav2vec_embedding_asr = outputs_asr.last_hidden_state.mean(dim=1)

            # Get embedding for mixed audio
            inputs_mixed = self.wav2vec_processor(
                audio_mixed_np,
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.device)
            outputs_mixed = self.wav2vec_model(**inputs_mixed)
            wav2vec_embedding_mixed = outputs_mixed.last_hidden_state.mean(dim=1)

        wav2vec_asr = F.cosine_similarity(
            wav2vec_embedding_asr,
            wav2vec_embedding_mixed,
            dim=1
        ).item()

        # Convert [-1, 1] to [0, 1] then invert
        val = (wav2vec_asr + 1) / 2.0
        val = -val + 1  # Invert: high similarity = low fitness (good)

        return float(val)
