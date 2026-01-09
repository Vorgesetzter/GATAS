import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
from Objectives.base import BaseObjective
from Datastructures.dataclass import ModelData, StepContext, AudioData
from Datastructures.enum import AttackMode, FitnessObjective


class SbertTargetObjective(BaseObjective):
    """
    SBERT cosine similarity between ASR text and target text (batched).

    sbert_target = cos_sim(emb_target, emb_asr)
    Values: [-1, 1]
    -1 = ASR very different to Target, 1 = ASR same as Target

    We convert to fitness: 0 = same as target (good), 1 = different (bad).

    NOTE: This objective requires TARGETED mode.
    """
    objective_type = FitnessObjective.SBERT_TARGET

    def __init__(self, config, model_data: ModelData, device: str = None, embedding_data=None):
        super().__init__(config, model_data)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Validate mode
        if config.mode is AttackMode.UNTARGETED:
            raise ValueError("AttackMode.UNTARGETED incompatible with SbertTargetObjective")

        # Lazy load SBERT model if not already loaded
        if self.model_data.sbert_model is None:
            print(f"[INFO] Loading SBERT Model (all-MiniLM-L6-v2) on {self.device}...")
            model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

            # Multi-GPU support
            if self.device == 'cuda' and torch.cuda.device_count() > 1:
                print(f"[INFO] SBERT using {torch.cuda.device_count()} GPUs.")
                # SentenceTransformer handles multi-GPU via start_multi_process_pool
                # but for simplicity we use the model's built-in pooling
                pool = model.start_multi_process_pool()
                model._pool = pool

            self.model_data.sbert_model = model

        self.sbert_model = self.model_data.sbert_model

        # Store target embedding (computed once)
        self.embedding_data = embedding_data
        if embedding_data is not None and embedding_data.s_bert_embedding_target is None:
            embedding_data.s_bert_embedding_target = self.sbert_model.encode(
                config.text_target,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: StepContext, audio_data: AudioData) -> list[float]:
        """Process entire batch at once."""
        asr_texts = context.clean_text
        if isinstance(asr_texts, str):
            asr_texts = [asr_texts]

        # Batch encode all ASR texts
        asr_embeddings = self.sbert_model.encode(
            asr_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=len(asr_texts)
        )

        # Compute cosine similarity for all at once
        # target_emb: [dim], asr_emb: [batch, dim]
        target_emb = self.embedding_data.s_bert_embedding_target
        if target_emb.dim() == 1:
            target_emb = target_emb.unsqueeze(0)  # [1, dim]

        # cos_sim returns [1, batch] matrix
        similarities = util.cos_sim(target_emb, asr_embeddings).squeeze(0)  # [batch]

        # Convert [-1, 1] to [0, 1] then invert
        val = (similarities + 1) / 2.0
        val = -val + 1  # Invert: high similarity = low fitness (good)

        return val.cpu().tolist()
