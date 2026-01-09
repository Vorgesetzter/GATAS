from Objectives.base import BaseObjective
from Datastructures.dataclass import ModelData, StepContext, AudioData
from Datastructures.enum import FitnessObjective


class WhisperProbObjective(BaseObjective):
    """
    Whisper probability objective - measures how likely Whisper thinks
    the audio matches the target text.

    This objective uses pre-computed values from the batch ASR step.
    The whisper_prob values are computed in the main loop using:
    - Cross-entropy loss between predicted logits and target tokens
    - Converted to probability via exp(-loss)
    - Inverted to fitness: 1 - prob

    Values: [0, 1]
    0 = ASR output matches target perfectly (good)
    1 = ASR output very different from target (bad)
    """
    objective_type = FitnessObjective.WHISPER_PROB

    def __init__(self, config, model_data: ModelData, device: str = None):
        super().__init__(config, model_data)

    @property
    def supports_batching(self) -> bool:
        return True  # Uses pre-computed batch values

    def _calculate_logic(self, context: StepContext, audio_data: AudioData) -> list[float]:
        """
        Returns the pre-computed whisper probability fitness values.

        These values are computed externally in the batch processing loop
        and stored in context.whisper_prob.
        """
        if context.whisper_prob is None:
            raise ValueError(
                "WhisperProbObjective requires pre-computed whisper_prob values in StepContext. "
                "Ensure the main loop computes these before calling this objective."
            )

        # Return the pre-computed values directly
        return list(context.whisper_prob)
