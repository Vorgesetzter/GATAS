import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from Objectives.base import BaseObjective
from Datastructures.dataclass import ModelData, StepContext, AudioData
from Datastructures.enum import FitnessObjective


class Wav2VecSimilarObjective(BaseObjective):
    """
    Wav2Vec2 audio embedding cosine similarity between mixed audio and GT audio (batched).

    wav2vec_gt = cos_sim(emb_gt, emb_mixed)
    Values: [-1, 1]
    -1 = Mixed audio very different to GT, 1 = Mixed audio same as GT

    We convert to fitness: 0 = same as GT (good), 1 = different (bad).
    (We want to sound SIMILAR to ground-truth)
    """
    objective_type = FitnessObjective.WAV2VEC_SIMILAR

    def __init__(self, config, model_data: ModelData, device: str = None, embedding_data=None):
        super().__init__(config, model_data)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

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
        self.embedding_data = embedding_data

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: StepContext, audio_data: AudioData) -> list[float]:
        """Process entire batch at once."""
        audio_mixed = context.audio_mixed  # [Batch, Time] or [Time]

        # Ensure batch dimension
        if isinstance(audio_mixed, torch.Tensor):
            if audio_mixed.dim() == 1:
                audio_mixed = audio_mixed.unsqueeze(0)
            audio_list = [a.cpu().numpy() for a in audio_mixed]
        else:
            audio_list = [audio_mixed] if not isinstance(audio_mixed, list) else audio_mixed

        # Process batch through Wav2Vec2
        with torch.no_grad():
            # Processor handles padding for variable length audio
            inputs = self.wav2vec_processor(
                audio_list,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            outputs = self.wav2vec_model(**inputs)
            # Mean pooling over time dimension: [Batch, Time, Hidden] -> [Batch, Hidden]
            wav2vec_embeddings = outputs.last_hidden_state.mean(dim=1)

        # Get GT embedding (should already be computed)
        gt_emb = self.embedding_data.wav2vec_embedding_gt  # [1, Hidden]
        if gt_emb.dim() == 1:
            gt_emb = gt_emb.unsqueeze(0)

        # Batch cosine similarity
        similarities = F.cosine_similarity(
            gt_emb.expand(wav2vec_embeddings.size(0), -1),
            wav2vec_embeddings,
            dim=1
        )

        # Convert [-1, 1] to [0, 1] then invert
        val = (similarities + 1) / 2.0
        val = -val + 1  # Invert: high similarity = low fitness (good)

        return val.cpu().tolist()
