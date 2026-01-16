import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
from Objectives.base import BaseObjective
from Datastructures.dataclass import ModelData, StepContext, ModelEmbeddingData
from Datastructures.enum import FitnessObjective


class SbertGtObjective(BaseObjective):
    """
    SBERT cosine similarity between ASR text and ground-truth text (batched).

    sbert_gt = cos_sim(emb_gt, emb_asr)
    Values: [-1, 1]
    -1 = ASR very different to GT, 1 = ASR same as GT

    We convert to fitness: 0 = different from GT (good), 1 = same as GT (bad).
    (We want to move AWAY from ground-truth)
    """
    objective_type = FitnessObjective.SBERT_GT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Load SBERT model if not already loaded
        if self.model_data.sbert_model is None:
            print(f"[INFO] Loading SBERT Model (all-MiniLM-L6-v2) on {self.device}...")
            model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

            # Multi-GPU support
            if self.device == 'cuda' and torch.cuda.device_count() > 1:
                print(f"[INFO] SBERT using {torch.cuda.device_count()} GPUs.")
                pool = model.start_multi_process_pool()
                model._pool = pool

            self.model_data.sbert_model = model

        self.sbert_model = self.model_data.sbert_model

        # Compute GT embedding if not already computed
        if self.embedding_data is not None and self.embedding_data.s_bert_embedding_gt is None:
            self.embedding_data.s_bert_embedding_gt = self.sbert_model.encode(
                self.text_gt,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: StepContext) -> list[float]:
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
        gt_emb = self.embedding_data.s_bert_embedding_gt
        if gt_emb.dim() == 1:
            gt_emb = gt_emb.unsqueeze(0)  # [1, dim]

        # cos_sim returns [1, batch] matrix
        similarities = util.cos_sim(gt_emb, asr_embeddings).squeeze(0)  # [batch]

        # Convert [-1, 1] to [0, 1]
        # High similarity to GT = high fitness (bad, we want to avoid GT)
        val = (similarities + 1) / 2.0

        return val.cpu().tolist()
