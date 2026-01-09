"""
ObjectiveManager - Centralized management of fitness objectives.

This class handles:
1. Initialization of active objectives (with lazy model loading)
2. Batch evaluation of all objectives
3. Score collection and organization
"""

import torch
from typing import Optional
from Datastructures.dataclass import ModelData, ConfigData, AudioData, EmbeddingData, StepContext
from Datastructures.enum import FitnessObjective
from Objectives.base import BaseObjective
from Objectives.registry import get_objective, ensure_all_registered


class ObjectiveManager:
    """
    Manages initialization and evaluation of all fitness objectives.

    Usage:
        # Initialize once at start
        manager = ObjectiveManager(config_data, model_data, device, embedding_data)

        # Evaluate on each batch
        scores = manager.evaluate_batch(context, audio_data)
    """

    def __init__(
        self,
        config: ConfigData,
        model_data: ModelData,
        device: str,
        embedding_data: Optional[EmbeddingData] = None
    ):
        # Ensure all objective classes are registered before lookup
        ensure_all_registered()

        self.config = config
        self.model_data = model_data
        self.device = device
        self.embedding_data = embedding_data or EmbeddingData()

        # Initialize objectives dict
        self.objectives: dict[FitnessObjective, BaseObjective] = {}

        # Initialize all active objectives
        self._initialize_objectives()

    def _initialize_objectives(self):
        """
        Initialize only the active objectives.
        Models are lazy-loaded within each objective's __init__.
        """
        print(f"[ObjectiveManager] Initializing {len(self.config.active_objectives)} objectives...")

        for obj_enum in self.config.active_objectives:
            try:
                # Use the lazy-loading get_objective function from registry
                objective = get_objective(
                    obj_enum,
                    self.config,
                    self.model_data,
                    device=self.device,
                    embedding_data=self.embedding_data
                )

                self.objectives[obj_enum] = objective
                print(f"  [OK] {obj_enum.name} (batching={objective.supports_batching})")

            except ValueError as e:
                print(f"  [WARNING] Objective {obj_enum.name} not found in registry: {e}")
                continue
            except Exception as e:
                print(f"  [ERROR] Failed to initialize {obj_enum.name}: {e}")
                raise

        print(f"[ObjectiveManager] All objectives initialized.")

    def evaluate_batch(
        self,
        context: StepContext,
        audio_data: AudioData
    ) -> dict[FitnessObjective, list[float]]:
        """
        Evaluate all objectives on a batch of samples.

        Args:
            context: StepContext containing batch data (audio_mixed, asr_text, etc.)
            audio_data: AudioData with reference audio (GT, target)

        Returns:
            Dictionary mapping FitnessObjective -> list of scores (one per sample in batch)
        """
        scores: dict[FitnessObjective, list[float]] = {}

        for obj_enum, objective in self.objectives.items():
            try:
                batch_scores = objective.calculate_score(context, audio_data)
                scores[obj_enum] = batch_scores
            except Exception as e:
                print(f"[ERROR] {obj_enum.name} evaluation failed: {e}")
                # Return worst score (1.0) for all samples in batch
                batch_size = len(context)
                scores[obj_enum] = [1.0] * batch_size

        return scores

    def evaluate_single(
        self,
        context: StepContext,
        audio_data: AudioData,
        index: int = 0
    ) -> dict[FitnessObjective, float]:
        """
        Evaluate all objectives on a single sample.

        This is a convenience method that extracts a single sample from batch results.

        Args:
            context: StepContext (can be batch or single)
            audio_data: AudioData with reference audio
            index: Index of sample to extract (default 0)

        Returns:
            Dictionary mapping FitnessObjective -> single score
        """
        batch_scores = self.evaluate_batch(context, audio_data)
        return {obj: scores[index] for obj, scores in batch_scores.items()}

    @property
    def active_objectives(self) -> list[FitnessObjective]:
        """Returns list of active objectives in order."""
        return list(self.objectives.keys())

    def get_objective_instance(self, obj_enum: FitnessObjective) -> Optional[BaseObjective]:
        """Get a specific objective instance."""
        return self.objectives.get(obj_enum)
