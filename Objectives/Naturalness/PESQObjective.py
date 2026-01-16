import numpy as np
import torch
import torchaudio.functional as F
from pesq import pesq
from Objectives.base.BaseObjective import BaseObjective
from Datastructures.dataclass import ModelData, StepContext, ModelEmbeddingData
from Datastructures.enum import FitnessObjective


class PesqObjective(BaseObjective):
    objective_type = FitnessObjective.PESQ

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

    def _calculate_logic(self, context: StepContext) -> float:
        """
        Calculates PESQ for a SINGLE candidate (Not Batched).
        """

        # --- 1. Lazy Cache Ground Truth (Run Only Once) ---
        if self.cached_gt_16k is None:
            # Move GT to CPU and Resample 24k -> 16k
            gt_tensor = torch.as_tensor(self.audio_gt, device='cpu', dtype=torch.float32)

            # Ensure shape [1, Time] for torchaudio
            if gt_tensor.dim() == 1:
                gt_tensor = gt_tensor.unsqueeze(0)

            # Resample
            resampler = F.resample(gt_tensor, orig_freq=24000, new_freq=16000)
            self.cached_gt_16k = resampler.squeeze().numpy()

        # --- 2. Process Candidate Audio ---
        # Input is likely on GPU [Time]
        audio_tensor = torch.as_tensor(context.audio_mixed, device=self.device, dtype=torch.float32)

        # Resample on GPU (Faster than CPU resampling)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # F.resample handles gradients gracefully, but we don't need them for PESQ
        with torch.no_grad():
            resampled_tensor = F.resample(audio_tensor, orig_freq=24000, new_freq=16000)

        # Move to CPU / Numpy for the library
        audio_np = resampled_tensor.squeeze().cpu().numpy()

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