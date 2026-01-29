import jiwer
from Objectives.base import BaseObjective
from Datastructures.dataclass import ObjectiveContext


class MerTargetObjective(BaseObjective):
    """
    Match Error Rate (MER) optimization objective.

    Formula: (S + D + I) / (Length_Reference + I)

    This is preferred over WER for fitness optimization because it is strictly
    bounded between [0, 1], preventing the optimizer from exploiting unbounded
    values by spamming insertions.

    Output:
         0.0 = 0% similarity (Attack Succeeded / Distinct content)
         1.0 = 100% similarity (Attack Failed / Identical content)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        self.transformations = self.model_data.wer_transformations

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: ObjectiveContext) -> list[float]:
        """
        Calculate MER for each ASR text against target text.
        Returns list of scores in range (0, 1) where 0 = match (good), 1 = different (bad).
        """
        asr_texts = context.asr_texts

        scores = []
        for asr_text in asr_texts:
            # Skip empty/invalid texts
            if not asr_text or len(asr_text) < 2:
                scores.append(1.0)  # Penalize invalid
                continue

            raw_mer = jiwer.mer(
                self.text_target,
                asr_text,
                reference_transform=self.transformations,
                hypothesis_transform=self.transformations,
            )

            # MER 0.0 (Match to target) -> 0.0 (good for attack)
            # MER 1.0 (Different from target) -> 1.0 (bad for attack)
            scores.append(float(raw_mer))

        return scores