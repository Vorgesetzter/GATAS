import jiwer
import phonemizer
from phonemizer.separator import Separator
from Objectives.base import BaseObjective
from Datastructures.dataclass import ObjectiveContext


class PerGtObjective(BaseObjective):
    """
    Phoneme Error Rate (PER) optimization objective away from ground truth.

    Uses MER logic on phonemes to ensure bounded [0, 1] output.

    Values: [0, 1]
    0 = phoneme sequence different from GT (good for attack)
    1 = phoneme sequence matches GT (bad for attack)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize phonemizer backend (same as StyleTTS2)
        self._phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us',
            preserve_punctuation=False,
            with_stress=True,
            language_switch='remove-flags'
        )

        # Separator: space between phonemes, space between words
        # Keeps multi-char phonemes like 'dʒ' intact as single units
        self._separator = Separator(phone=" ", word=" ")

        # Pre-compute ground truth phonemes
        self._gt_phonemes = self._batch_to_phonemes([self.text_gt])[0]

        # Minimal jiwer transformations
        self._transformations = jiwer.Compose([
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ])

    def _batch_to_phonemes(self, texts: list[str]) -> list[str]:
        """Batch convert texts to space-separated phoneme strings."""
        if not texts:
            return []

        # Vectorized phonemization (much faster than looping)
        phoneme_list = self._phonemizer.phonemize(
            texts,
            strip=True,
            separator=self._separator
        )

        return [p.strip() if p else "" for p in phoneme_list]

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: ObjectiveContext) -> list[float]:
        """
        Calculate PER for each ASR text against ground truth.
        Returns list of scores in range [0, 1] where 0 = different (good), 1 = same (bad).
        """
        asr_texts = context.asr_texts

        # Filter valid inputs
        valid_indices = [i for i, t in enumerate(asr_texts) if t and len(t.strip()) > 0]
        valid_texts = [asr_texts[i] for i in valid_indices]

        scores = [1.0] * len(asr_texts)  # Default bad score (matches GT)

        if not valid_texts:
            return scores

        # Batch phonemization (performance optimization)
        asr_phonemes_list = self._batch_to_phonemes(valid_texts)

        # Calculate MER for each item individually
        # (batched jiwer.mer returns aggregate, not per-item)
        for idx, asr_phonemes in zip(valid_indices, asr_phonemes_list):
            if not asr_phonemes:
                continue

            raw_mer = jiwer.mer(
                self._gt_phonemes,
                asr_phonemes,
                reference_transform=self._transformations,
                hypothesis_transform=self._transformations,
            )

            # Invert: high MER (different from GT) = low score (good for attack)
            # MER 0.0 (Match) -> 1.0 (bad)
            # MER 1.0 (Diff)  -> 0.0 (good)
            scores[idx] = 1.0 - float(raw_mer)

        return scores
