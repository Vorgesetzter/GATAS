import os
import numpy as np
import torch
from Objectives.base.BaseObjective import BaseObjective
from Datastructures.dataclass import ObjectiveContext


class VisqolObjective(BaseObjective):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Cache for resampled GT audio (computed on first use)
        self.cached_gt_16k = None

        # ViSQOL API instance — initialised once at construction
        self._api = self._init_api()

    @staticmethod
    def _init_api():
        from visqol import visqol_lib_py
        from visqol.pb2 import visqol_config_pb2

        config = visqol_config_pb2.VisqolConfig()
        config.audio.sample_rate = 16000
        config.options.use_speech_scoring = True  # speech mode (vs audio mode)

        # The SVR model is bundled with the visqol package
        model_path = os.path.join(
            os.path.dirname(visqol_lib_py.__file__),
            "model",
            "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_finetune_ldavg10_ldlayr.2_hyper1700_ep2400_linear1.zip",
        )
        config.options.svr_model_path = model_path

        api = visqol_lib_py.VisqolApi()
        api.Create(config)
        return api

    @property
    def supports_batching(self):
        """
        ViSQOL is a C++ binding that accepts single numpy arrays.
        BaseObjective will loop over the batch for us.
        """
        return False

    def _calculate_logic(self, context: ObjectiveContext) -> float:
        """
        Calculates ViSQOL MOS-LQO for a SINGLE candidate (not batched).
        """
        audio_mixed = context.audio_mixed_batch

        # --- 1. Lazy-cache ground truth (audio already at 16 kHz) ---
        if self.cached_gt_16k is None:
            # ViSQOL requires float64
            self.cached_gt_16k = self.audio_gt.squeeze().cpu().numpy().astype(np.float64)

        # --- 2. Candidate audio (already at 16 kHz) ---
        audio_np = audio_mixed.squeeze().cpu().numpy().astype(np.float64)

        # --- 3. Run ViSQOL ---
        try:
            result = self._api.Measure(self.cached_gt_16k, audio_np)
            score = result.moslqo  # MOS-LQO: 1.0 (worst) → 5.0 (perfect)
        except Exception:
            score = 1.0  # return worst score on failure

        # --- 4. Normalise to [0, 1] where 0.0 is best ---
        # MOS-LQO 1.0 → 5.0  becomes  fitness 1.0 → 0.0
        val = (score - 1.0) / 4.0       # [0, 1] where 1.0 = perfect
        fitness = 1.0 - val             # invert so 0.0 = perfect

        return float(max(0.0, min(1.0, fitness)))
