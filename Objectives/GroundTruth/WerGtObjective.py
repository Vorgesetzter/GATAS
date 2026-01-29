import jiwer
from Objectives.base import BaseObjective
from Datastructures.dataclass import ObjectiveContext
from tqdm.auto import tqdm


class WerGtObjective(BaseObjective):
    """
    Word Error Rate between ASR text and ground-truth text (inverted).

    WER = (Substitutions + Deletions + Insertions) / Number_of_reference_words
    Values: usually (0, 1), rarely > 1
    0 = perfect match, 1 = 100% of words wrong

    We INVERT this: higher WER (more different from GT) is better.
    Output is normalized to (0, 1) where:
         0 = 0% similarity / very different from GT (good for attack)
         1 = 100% similarity / same as GT (bad for attack)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        return True

    def _calculate_logic(self, context: ObjectiveContext) -> list[float]:
        """
        Batched WER calculation.
        Returns list of scores in range (0, 1) where 0 = different (good), 1 = same (bad).
        """
        asr_texts = context.asr_texts

        scores = []
        for asr_text in asr_texts:
            # Skip empty/invalid texts
            if not asr_text or len(asr_text) < 2:
                scores.append(1.0)  # Penalize invalid
                continue

            raw_wer = jiwer.wer(
                self.text_gt,
                asr_text,
                reference_transform=self.wer_transformations,
                hypothesis_transform=self.wer_transformations,
            )

            raw_wer = min(float(raw_wer), 2.0)
            raw_wer /= 2.0

            if raw_wer > 1:
                tqdm.write(f"WER > 1 for \"{asr_text}\"")
            val = -raw_wer + 1
            scores.append(val)

        return scores
