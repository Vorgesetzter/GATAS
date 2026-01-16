import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from Objectives.base import BaseObjective
from Datastructures.dataclass import ModelData, StepContext, ModelEmbeddingData
from Datastructures.enum import AttackMode, FitnessObjective


class Wav2VecDifferentObjective(BaseObjective):
    """
    Wav2Vec2 audio embedding cosine similarity between mixed audio and GT audio (batched).

    wav2vec_sim = cos_sim(emb_gt, emb_mixed)
    Values: [-1, 1]
    -1 = Mixed audio very different to GT, 1 = Mixed audio same as GT

    We convert to fitness: 0 = different from GT (good), 1 = same as GT (bad).
    (We want to sound DIFFERENT from ground-truth)

    NOTE: This objective requires TARGETED mode.
    """
    objective_type = FitnessObjective.WAV2VEC_DIFFERENT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Validate mode
        if self.mode is AttackMode.UNTARGETED:
            raise ValueError("AttackMode.UNTARGETED incompatible with Wav2VecDifferentObjective")

        # Load Wav2Vec2 model if not already loaded
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

        # Compute GT embedding if not already computed
        if self.embedding_data is not None and self.embedding_data.wav2vec_embedding_gt is None and self.audio_gt is not None:
            print("[INFO] Computing Wav2Vec GT embedding...")
            self.embedding_data.wav2vec_embedding_gt = self._compute_embedding(self.audio_gt)

    def _compute_embedding(self, audio) -> torch.Tensor:
        """Compute Wav2Vec embedding for audio."""
        with torch.no_grad():
            inputs = self.wav2vec_processor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).to(self.device)
            outputs = self.wav2vec_model(**inputs)
            return torch.mean(outputs.last_hidden_state, dim=1)

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: StepContext) -> list[float]:
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
            inputs = self.wav2vec_processor(
                audio_list,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            outputs = self.wav2vec_model(**inputs)
            wav2vec_embeddings = outputs.last_hidden_state.mean(dim=1)

        # Get GT embedding
        gt_emb = self.embedding_data.wav2vec_embedding_gt
        if gt_emb.dim() == 1:
            gt_emb = gt_emb.unsqueeze(0)

        # Batch cosine similarity
        similarities = F.cosine_similarity(
            gt_emb.expand(wav2vec_embeddings.size(0), -1),
            wav2vec_embeddings,
            dim=1
        )

        # Convert [-1, 1] to [0, 1]
        # High similarity to GT = high fitness (bad, we want to be different)
        val = (similarities + 1) / 2.0

        return val.cpu().tolist()
