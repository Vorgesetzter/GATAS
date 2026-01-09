import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from Objectives.base import BaseObjective
from Datastructures.dataclass import ModelData, StepContext, AudioData
from Datastructures.enum import AttackMode, FitnessObjective


class TextEmbTargetObjective(BaseObjective):
    """
    Text embedding (MPNet) cosine similarity between ASR text and target text (batched).

    text_dist_target = cos_sim(emb_target, emb_asr)
    Values: [-1, 1]
    -1 = ASR very different to Target, 1 = ASR same as Target

    We convert to fitness: 0 = same as target (good), 1 = different (bad).

    NOTE: This objective requires TARGETED mode.
    """
    objective_type = FitnessObjective.TEXT_EMB_TARGET

    def __init__(self, config, model_data: ModelData, device: str = None, embedding_data=None):
        super().__init__(config, model_data)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Validate mode
        if config.mode is AttackMode.UNTARGETED:
            raise ValueError("AttackMode.UNTARGETED incompatible with TextEmbTargetObjective")

        # Lazy load embedding model if not already loaded
        if self.model_data.embedding_model is None:
            print(f"[INFO] Loading SentenceTransformer (all-mpnet-base-v2) on {self.device}...")
            model = SentenceTransformer('all-mpnet-base-v2', device=self.device)

            # Multi-GPU support
            if self.device == 'cuda' and torch.cuda.device_count() > 1:
                print(f"[INFO] MPNet using {torch.cuda.device_count()} GPUs.")
                pool = model.start_multi_process_pool()
                model._pool = pool

            self.model_data.embedding_model = model

        self.embedding_model = self.model_data.embedding_model

        # Store target embedding (computed once)
        self.embedding_data = embedding_data
        if embedding_data is not None and embedding_data.text_embedding_target is None:
            embedding_data.text_embedding_target = self.embedding_model.encode(
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
        asr_embeddings = self.embedding_model.encode(
            asr_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=len(asr_texts)
        )

        # Compute cosine similarity for all at once
        # target_emb: [dim], asr_emb: [batch, dim]
        target_emb = self.embedding_data.text_embedding_target
        if target_emb.dim() == 1:
            target_emb = target_emb.unsqueeze(0)  # [1, dim]

        # Batch cosine similarity: [1, dim] @ [dim, batch] -> [1, batch]
        similarities = F.cosine_similarity(
            target_emb.unsqueeze(1),  # [1, 1, dim]
            asr_embeddings.unsqueeze(0),  # [1, batch, dim]
            dim=2
        ).squeeze(0)  # [batch]

        # Convert [-1, 1] to [0, 1] then invert
        val = (similarities + 1) / 2.0
        val = -val + 1  # Invert: high similarity = low fitness (good)

        return val.cpu().tolist()
