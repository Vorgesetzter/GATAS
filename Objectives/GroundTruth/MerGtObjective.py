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
        asr_texts = context.asr_texts

        # 1. Filter empty/invalid texts to avoid crashes
        # We assign them a score of 1.0 (High Similarity/Bad for attack)
        # to force the optimizer to produce valid text.
        valid_indices = [i for i, t in enumerate(asr_texts) if t and len(t) >= 2]
        valid_texts = [asr_texts[i] for i in valid_indices]

        scores = [1.0] * len(asr_texts)  # Default bad score

        if not valid_texts:
            return scores

        # 2. Prepare Reference batch
        refs = [self.text_gt] * len(valid_texts)

        # 3. Calculate MER (Match Error Rate)
        # mer is 0.0 for perfect match, 1.0 for total mismatch
        mer_vals = jiwer.mer(
            reference=refs,
            hypothesis=valid_texts,
            reference_transform=self.transformations,
            hypothesis_transform=self.transformations
        )

        if isinstance(mer_vals, float):
            mer_vals = [mer_vals]

        # 4. Invert Logic: We want Similarity
        # MER 0.0 (Match) -> Score 1.0 (High Similarity)
        # MER 1.0 (Diff)  -> Score 0.0 (Low Similarity)
        for idx, val in zip(valid_indices, mer_vals):
            scores[idx] = 1.0 - val

        return scores