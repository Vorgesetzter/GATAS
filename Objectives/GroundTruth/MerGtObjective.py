import jiwer
from Objectives.base import BaseObjective
from Datastructures.dataclass import ObjectiveContext


class MerGtObjective(BaseObjective):
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
        Calculate MER for each ASR text against ground truth.
        Returns list of scores in range (0, 1) where 0 = different (good), 1 = same (bad).
        """
        asr_texts = context.asr_texts

        scores = []
        for asr_text in asr_texts:
            # Skip empty/invalid texts
            if not asr_text or len(asr_text) < 2:
                scores.append(1.0)  # Penalize invalid
                continue

            raw_mer = jiwer.mer(
                self.text_gt,
                asr_text,
                reference_transform=self.transformations,
                hypothesis_transform=self.transformations,
            )

            # Invert: MER 0.0 (Match) -> 1.0 (High Similarity, bad for attack)
            #         MER 1.0 (Diff)  -> 0.0 (Low Similarity, good for attack)
            scores.append(1.0 - float(raw_mer))

        return scores