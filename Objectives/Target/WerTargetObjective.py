import jiwer
from Objectives.base import BaseObjective
from Datastructures.dataclass import ModelData, StepContext, AudioData
from Datastructures.enum import FitnessObjective


class WerTargetObjective(BaseObjective):
    """
    Word Error Rate between ASR text and target text.

    WER = (Substitutions + Deletions + Insertions) / Number_of_reference_words
    Values: usually (0, 1), rarely > 1
    0 = perfect match, 1 = 100% of words wrong

    Lower is better (we want ASR output to match target).
    """
    objective_type = FitnessObjective.WER_TARGET

    def __init__(self, config, model_data: ModelData, device: str = None):
        super().__init__(config, model_data)
        self.text_target = config.text_target

        # Lazy load WER transformations if not already loaded
        if self.model_data.wer_transformations is None:
            self.model_data.wer_transformations = jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.RemovePunctuation(),
                jiwer.Strip(),
            ])

        self.wer_transformations = self.model_data.wer_transformations

    @property
    def supports_batching(self) -> bool:
        return False  # JIWER doesn't support batching natively

    def _calculate_logic(self, context: StepContext, audio_data: AudioData) -> float:
        asr_text = context.clean_text

        wer = jiwer.wer(
            self.text_target,
            asr_text,
            reference_transform=self.wer_transformations,
            hypothesis_transform=self.wer_transformations,
        )

        val = float(wer)
        val = min(val, 1.0)  # Clamp to (0, 1)

        return val
