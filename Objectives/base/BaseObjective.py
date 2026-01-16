from abc import ABC, abstractmethod
from typing import ClassVar, Type, Optional, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from Datastructures.enum import FitnessObjective, AttackMode

from Datastructures.dataclass import ModelData, StepContext, ModelEmbeddingData


class BaseObjective(ABC):
    """
    Abstract base class for all fitness objectives.

    Subclasses must define `objective_type` as a class variable to auto-register:
        class MyObjective(BaseObjective):
            objective_type = FitnessObjective.MY_OBJECTIVE
    """

    # Class-level registry: maps FitnessObjective enum -> objective class
    _registry: ClassVar[dict["FitnessObjective", Type["BaseObjective"]]] = {}

    # Each subclass must define this (set to None in base to allow abstract usage)
    objective_type: ClassVar["FitnessObjective"] = None

    def __init_subclass__(cls, **kwargs):
        """Auto-register subclasses that define objective_type."""
        super().__init_subclass__(**kwargs)
        if cls.objective_type is not None:
            BaseObjective._registry[cls.objective_type] = cls

    @classmethod
    def get_class(cls, objective_enum: "FitnessObjective") -> Type["BaseObjective"]:
        """Get the objective class for a given enum value."""
        if objective_enum not in cls._registry:
            raise ValueError(f"No objective registered for {objective_enum}")
        return cls._registry[objective_enum]

    @classmethod
    def get_all_registered_enums(cls) -> list["FitnessObjective"]:
        """Returns all registered FitnessObjective enums."""
        return list(cls._registry.keys())

    def __init__(
        self,
        model_data: ModelData,
        device: str = None,
        embedding_data: ModelEmbeddingData = None,
        text_gt: Optional[str] = None,
        text_target: Optional[str] = None,
        mode: Optional["AttackMode"] = None,
        audio_gt: Optional[torch.Tensor] = None,
        style_vector_acoustic: Optional[torch.Tensor] = None,
        style_vector_prosodic: Optional[torch.Tensor] = None,
    ):
        self.model_data = model_data
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_data = embedding_data
        self.text_gt = text_gt
        self.text_target = text_target
        self.mode = mode
        self.audio_gt = audio_gt
        self.style_vector_acoustic = style_vector_acoustic
        self.style_vector_prosodic = style_vector_prosodic

    @property
    def name(self):
        """Returns the class name (e.g., 'PesqObjective') for logging."""
        return self.__class__.__name__

    @property
    def supports_batching(self) -> bool:
        """
        Override this to True if your _calculate_logic can handle a batch.
        Default is False (Safe Mode).
        """
        return False

    def calculate_score(self, context: StepContext) -> list[float]:
        """
        Public API. ALWAYS returns a list of floats, even if batch_size=1.
        """
        # --- PATH A: Batch Optimized (GPU Models) ---
        if self.supports_batching:
            try:
                # We trust the child class to handle the whole batch at once
                return self._calculate_logic(context)
            except Exception as e:
                print(f"Error in {self.name} (Batch Mode): {e}")
                # Return a list of 1.0s equal to the batch size as fail-safe
                return [1.0] * len(context.asr_text)

        # --- PATH B: Single Item Loop (CPU/Legacy Models) ---
        else:
            scores = []
            # We assume context has a __len__ and get_item method
            for i in range(len(context)):
                try:
                    single_ctx = context.get_item(i)

                    # Run safety checks on single item
                    if not single_ctx.clean_text or len(
                            single_ctx.clean_text) < 2 or single_ctx.audio_mixed.numel() == 0:
                        scores.append(1.0)
                        continue

                    # Call logic on ONE item
                    val = self._calculate_logic(single_ctx)
                    scores.append(float(val))

                except Exception as e:
                    print(f"Error in {self.name} index {i}: {e}")
                    scores.append(1.0)

            return scores

    @abstractmethod
    def _calculate_logic(self, context: StepContext):
        """
        The specific math for this objective.
        """
        pass