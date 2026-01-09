import torch
from Objectives.base.BaseObjective import BaseObjective
from Datastructures.dataclass import ModelData, StepContext, AudioData
from Datastructures.enum import FitnessObjective


class PhonemeCountObjective(BaseObjective):
    objective_type = FitnessObjective.PHONEME_COUNT

    def __init__(self, config, model_data: ModelData, device: str = None):
        super().__init__(config, model_data)
        # This objective relies on the main TTS model which should already be loaded.
        if self.model_data.tts_model is None:
            raise RuntimeError("PhonemeCountObjective requires 'tts_model' to be loaded in ModelData.")

    @property
    def supports_batching(self):
        """
        We handle the list loop internally for cleaner code,
        even though the underlying preprocess_text is sequential.
        """
        return True

    def _calculate_logic(self, context: StepContext, audio_data: AudioData):
        """
        Calculates the length difference between Generated ASR Text and Ground Truth Text.
        Returns a LIST of floats [0.0, 1.0].
        """
        scores = []

        # 1. Get Ground Truth Length (n_gt)
        # audio_data.input_lengths contains the phoneme count of the original audio
        # We assume this is constant for the batch (attacking one file at a time).
        # We take the first item if it's a batch tensor.
        if isinstance(audio_data.input_lengths, torch.Tensor):
            n_gt = int(audio_data.input_lengths.view(-1)[0].item())
        else:
            n_gt = int(audio_data.input_lengths)

        # 2. Get TTS Model for preprocessing
        tts = self.model_data.tts_model

        # 3. Process Batch
        texts = context.asr_text
        if isinstance(texts, str): texts = [texts]

        for text in texts:
            try:
                # Preprocess: Text -> Phonemes -> Tokens
                # This uses the same phonemizer as the TTS generation
                tokens_asr = tts.preprocess_text(text)
                n_asr = int(tokens_asr.shape[-1])

                # 4. Calculate Fitness
                # Failure Case: Empty or massively too long (hallucination)
                if n_asr == 0 or n_asr > n_gt * 2:
                    val = 1.0
                else:
                    # Squared Error Ratio
                    # If n_gt=50, n_asr=50 -> error=0 -> val=0.0 (Perfect)
                    # If n_gt=50, n_asr=40 -> error=0.2 -> val=0.04
                    # If n_gt=50, n_asr=100 -> error=1.0 -> val=1.0 (Bad)
                    error = abs(n_asr - n_gt) / max(1, n_gt)
                    val = float(min(1.0, error * error))

                scores.append(val)

            except Exception as e:
                print(f"[Warning] PhonemeCount failed for '{text}': {e}")
                scores.append(1.0)

        return scores