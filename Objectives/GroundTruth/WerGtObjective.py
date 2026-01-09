import jiwer
from Objectives.base import BaseObjective
from Datastructures.dataclass import ModelData, StepContext, AudioData
from Datastructures.enum import FitnessObjective


class WerGtObjective(BaseObjective):
    """
    Word Error Rate between ASR text and ground-truth text (inverted).

    WER = (Substitutions + Deletions + Insertions) / Number_of_reference_words
    Values: usually (0, 1), rarely > 1
    0 = perfect match, 1 = 100% of words wrong

    We INVERT this: higher WER (more different from GT) is better.
    Output is normalized to (0, 1) where 0 = same as GT (bad), 1 = very different (good).
    But since we minimize fitness, we flip it: 0 = very different (good), 1 = same as GT (bad).
    """
    objective_type = FitnessObjective.WER_GT

    def __init__(self, config, model_data: ModelData, device: str = None):
        super().__init__(config, model_data)
        self.text_gt = config.text_gt

        # Lazy load WER transformations if not already loaded
        if self.model_data.wer_transformations is None:
            self.model_data.wer_transformations = jiwer.Compose([
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ])

        self.wer_transformations = self.model_data.wer_transformations

    @property
    def supports_batching(self) -> bool:
        return False  # JIWER doesn't support batching natively

    def _calculate_logic(self, context: StepContext, audio_data: AudioData) -> float:
        asr_text = context.clean_text

        wer = jiwer.wer(
            self.text_gt,
            asr_text,
            reference_transform=self.wer_transformations,
            hypothesis_transform=self.wer_transformations,
        )

        val = float(wer)
        val = min(val, 2.0)
        # Invert: high WER = good (different from GT)
        val = -val + 2.0
        val /= 2.0

        return val
