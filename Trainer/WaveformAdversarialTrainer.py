"""
WaveformAdversarialTrainer - Waveform-space variant of AdversarialTrainer.

Supports two attack modes:
  - TARGETED / NOISE_UNTARGETED: interpolates between audio_gt and audio_target waveforms
      audio_mixed = (1 - alpha) * audio_gt + alpha * audio_target
      optimizer bounds: (0, 1)
      IV=0 → pure GT, IV=1 → pure target (SET_OVERLAP=0)

All other optimization logic (NSGA-II, scoring, early stopping, logging) is inherited.
"""

import time
import torch
import torch.nn.functional as F
import numpy as np

from Datastructures.dataclass import ObjectiveContext
from Datastructures.enum import AttackMode
from Objectives.FitnessObjective import FitnessObjective
from Objectives.base import BaseObjective
from Trainer.AdversarialTrainer import AdversarialTrainer


class WaveformAdversarialTrainer(AdversarialTrainer):

    def __init__(
        self,
        tts_model,
        asr_model,
        thresholds: dict[FitnessObjective, float],
        objectives: dict[FitnessObjective, BaseObjective],
        original_audio: torch.Tensor,
        device: str,
        mode: AttackMode = AttackMode.NOISE_UNTARGETED,
        target_audio: torch.Tensor = None,
    ):
        super().__init__(tts_model, asr_model, thresholds, objectives, vector_manipulator=None, device=device)
        self.mode = mode

        self.original_audio = original_audio.to(device)
        if self.original_audio.dim() == 1:
            self.original_audio = self.original_audio.unsqueeze(0)

        if target_audio is not None:
            target_audio = target_audio.to(device)
            if target_audio.dim() == 1:
                target_audio = target_audio.unsqueeze(0)
            # Align target length to original
            T = self.original_audio.shape[-1]
            if target_audio.shape[-1] < T:
                target_audio = F.pad(target_audio, (0, T - target_audio.shape[-1]))
            else:
                target_audio = target_audio[..., :T]
            # Normalize target RMS to match GT so waveform interpolation is not dominated by amplitude
            gt_rms = self.original_audio.pow(2).mean().sqrt().clamp(min=1e-8)
            target_rms = target_audio.pow(2).mean().sqrt().clamp(min=1e-8)
            target_audio = target_audio * (gt_rms / target_rms)
        self.target_audio = target_audio

    def _process_batch(self, batch_idx: int, batch_size: int, interpolation_vectors_full: torch.Tensor):
        start_time = time.time()

        batch = interpolation_vectors_full[batch_idx: batch_idx + batch_size]
        current_batch_size = batch.shape[0]

        # Interpolate between original and target waveforms for all modes
        # IV=0 → pure GT, IV=1 → pure target (SET_OVERLAP=0)
        audio_mixed_batch = (
            (1.0 - batch) * self.original_audio.expand(current_batch_size, -1)
            + batch * self.target_audio.expand(current_batch_size, -1)
        )

        audio_mixed_batch = audio_mixed_batch.clamp(-1.0, 1.0)

        asr_texts, mel_batch = self.asr_model.inference(audio_mixed_batch)

        context = ObjectiveContext(
            audio_mixed_batch=audio_mixed_batch,
            asr_texts=asr_texts,
            interpolation_vectors=batch,
            mel_batch=mel_batch,
        )
        batch_scores_dict = self.evaluate_batch(context)

        end_time = time.time()

        batch_scores_list = []
        score_matrix = np.zeros((current_batch_size, len(self.objectives)), dtype=np.float64)

        for obj_idx, obj in enumerate(self.objectives):
            scores = np.array(batch_scores_dict[obj], dtype=np.float64)
            score_matrix[:, obj_idx] = scores
            batch_scores_list.append(scores.tolist())

        stop_optimization = self._check_early_stopping_batch(score_matrix)
        elapsed_time = end_time - start_time

        audio_list = [audio_mixed_batch[i].detach().cpu() for i in range(current_batch_size)]

        return stop_optimization, batch_scores_list, elapsed_time, audio_list, asr_texts
