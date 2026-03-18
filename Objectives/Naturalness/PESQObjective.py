import numpy as np
import torch
from pesq import pesq
from Objectives.base.BaseObjective import BaseObjective
from Datastructures.dataclass import ModelData, ModelEmbeddingData, ObjectiveContext


class PesqObjective(BaseObjective):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Cache for resampled GT audio (computed on first use)
        self.cached_gt_16k = None

    @property
    def supports_batching(self):
        """
        PESQ is a CPU library (C++ binding) that accepts single numpy arrays.
        It cannot handle GPU tensors or batches natively.
        We return False so BaseObjective loops for us.
        """
        return False

    def _calculate_logic(self, context: ObjectiveContext) -> float:
        """
        Calculates PESQ for a SINGLE candidate (Not Batched).
        """
        audio_mixed = context.audio_mixed_batch

        # --- 1. Lazy Cache Ground Truth (Run Only Once) ---
        if self.cached_gt_16k is None:
            # Audio is already at 16 kHz (StyleTTS2 resamples internally)
            self.cached_gt_16k = self.audio_gt.squeeze().cpu().numpy()

        # --- 2. Process Candidate Audio ---
        # Audio is already at 16 kHz
        audio_np = audio_mixed.squeeze().cpu().numpy()

        # --- 3. Run PESQ ---
        try:
            # Mode 'wb' = Wideband (16kHz)
            # Returns float between -0.5 and 4.5
            score = pesq(16000, self.cached_gt_16k, audio_np, 'wb')
        except Exception as e:
            # PESQ can crash on silent/short audio. Return worst score.
            # print(f"[Warning] PESQ Failed: {e}")
            score = -0.5

        # --- 4. Normalization ---
        # Raw PESQ: -0.5 (Bad) -> 4.5 (Perfect)
        # Goal:      1.0 (Bad) -> 0.0 (Perfect)

        # Shift to [0.0, 5.0]
        val = score + 0.5
        # Scale to [0.0, 1.0] (where 1.0 is Best)
        val /= 5.0
        # Invert (where 0.0 is Best)
        fitness = 1.0 - val

        # Clamp for safety
        return float(max(0.0, min(1.0, fitness)))