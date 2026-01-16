import torch
from Objectives.base import BaseObjective
from Datastructures.dataclass import ModelData, StepContext, ModelEmbeddingData
from Datastructures.enum import FitnessObjective


class L2Objective(BaseObjective):
    """
    L2 Norm of the interpolation vector.

    L2 = sqrt(mean(IV^2)) [RMS - punishes larger values more]
    Values: (0, 1)
    0 = only GT, 1 = only Target

    Lower is better (we want to stay close to GT).
    """
    objective_type = FitnessObjective.L2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: StepContext) -> list[float]:
        iv = context.interpolation_vector

        if iv.dim() == 1:
            iv = iv.unsqueeze(0)

        # L2 = sqrt(mean(IV^2)) per sample
        l2_values = (iv ** 2).mean(dim=1).sqrt()

        return l2_values.cpu().tolist()
