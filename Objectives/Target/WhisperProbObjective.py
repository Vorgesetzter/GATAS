import torch
from whisper.tokenizer import get_tokenizer
from Objectives.base import BaseObjective
from Datastructures.dataclass import ModelData, StepContext, ModelEmbeddingData
from Datastructures.enum import FitnessObjective


class WhisperProbObjective(BaseObjective):
    """
    Whisper probability objective - measures how likely Whisper thinks
    the audio matches the target text.

    This objective computes:
    - Cross-entropy loss between predicted logits and target tokens
    - Converted to probability via exp(-loss)
    - Inverted to fitness: 1 - prob

    Values: [0, 1]
    0 = ASR output matches target perfectly (good)
    1 = ASR output very different from target (bad)
    """
    objective_type = FitnessObjective.WHISPER_PROB

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Get the actual ASR model (unwrap DataParallel if needed)
        self._real_asr_model = self._get_real_asr_model(self.model_data.asr_model)

        # Prepare tokenized target text for probability computation
        self._target_tokens_template = self._prepare_whisper_tokens()

    def _get_real_asr_model(self, asr_model):
        """Extract actual model from DataParallel wrapper if needed."""
        if isinstance(asr_model, torch.nn.DataParallel):
            return asr_model.module
        return asr_model

    def _prepare_whisper_tokens(self) -> torch.Tensor:
        """Prepare tokenized target text for WHISPER_PROB computation."""
        tokenizer = get_tokenizer(self._real_asr_model.is_multilingual)
        target_ids = (
            list(tokenizer.sot_sequence) +
            tokenizer.encode(self.text_target) +
            [tokenizer.eot]
        )
        return torch.tensor([target_ids]).to(self.device)

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: StepContext) -> list[float]:
        """
        Computes the whisper probability fitness values from the mel spectrogram.

        Args:
            context: StepContext containing mel_batch

        Returns:
            List of fitness values (0 = best, 1 = worst)
        """
        if context.mel_batch is None:
            raise ValueError(
                "WhisperProbObjective requires mel_batch in StepContext. "
                "Ensure the main loop provides mel spectrogram before calling this objective."
            )

        mel_batch = context.mel_batch
        batch_size = mel_batch.shape[0]

        # Expand target tokens to batch size
        target_tokens_batch = self._target_tokens_template.expand(batch_size, -1)

        with torch.no_grad():
            logits = self.model_data.asr_model(mel_batch, target_tokens_batch)

        # Remove start and end token
        logits_shifted = logits[:, :-1, :]
        targets_shifted = target_tokens_batch[:, 1:]

        # Cross-entropy loss per token
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        raw_losses = loss_fct(
            logits_shifted.reshape(-1, logits_shifted.size(-1)),
            targets_shifted.reshape(-1)
        )

        # Average loss over sentence length
        sample_losses = raw_losses.reshape(batch_size, -1).mean(dim=1)

        # Convert to fitness score (0.0 = best, 1.0 = worst)
        probs = torch.exp(-sample_losses)
        vals = 1.0 - probs

        return vals.detach().cpu().tolist()
