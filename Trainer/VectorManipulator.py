import torch
from torch import Tensor

from Datastructures.dataclass import AudioData, ConfigData, AudioEmbeddingData
from Datastructures.enum import AttackMode


def _extend_to_size(x: Tensor, target_size: int) -> Tensor:
    """
    Extends the last dimension of x to `target_size` by repeating elements.
    Supports inputs of any dimension (e.g., [Batch, Dim] or [Batch, 1, Dim]).
    """

    a = x.shape[-1]

    if a >= target_size:
        return x[..., :target_size]

    base = target_size // a
    rem = target_size % a

    repeats = torch.full((a,), base, device=x.device, dtype=torch.long)
    if rem > 0:
        repeats[:rem] += 1

    idx = torch.arange(a, device=x.device).repeat_interleave(repeats)
    assert idx.numel() == target_size

    return x[..., idx]


def _pad_with_pattern(tensor: Tensor, amount: int, pattern: list[int]) -> Tensor:
    """Pads tensor along last dimension with a repeating pattern."""
    padding = torch.as_tensor(pattern, device=tensor.device, dtype=tensor.dtype)[
        torch.arange(amount, device=tensor.device) % len(pattern)
    ]

    for _ in range(tensor.dim() - 1):
        padding = padding.unsqueeze(0)
    padding = padding.expand(*tensor.shape[:-1], amount)

    return torch.cat([tensor, padding], dim=-1)


def add_numbers_pattern(a: Tensor, b: Tensor, pattern: list[int]) -> tuple[Tensor, Tensor]:
    """Pads the shorter tensor to match the longer one using a repeating pattern."""
    len_a = a.size(-1)
    len_b = b.size(-1)

    if len_a == len_b:
        return a, b

    if len_a < len_b:
        a = _pad_with_pattern(a, len_b - len_a, pattern)
    else:
        b = _pad_with_pattern(b, len_a - len_b, pattern)

    return a, b


def _adjust_interpolation_vector(interpolation_vector: Tensor, matrix: Tensor, subspace_optimization: bool) -> Tensor:
    """Adjusts interpolation vector dimensions for TTS inference."""
    if interpolation_vector.shape[-1] != 1:
        if subspace_optimization:
            interpolation_vector = interpolation_vector @ matrix
        else:
            interpolation_vector = _extend_to_size(interpolation_vector, 512)

    interpolation_vector = interpolation_vector.transpose(-1, -2)

    if interpolation_vector.dim() < 3:
        interpolation_vector = interpolation_vector.unsqueeze(0)

    return interpolation_vector


def generate_similar_noise(reference_tensor):
    """
    Generates random noise matching per-feature mean and std of the reference tensor.
    For 3D tensors (batch, seq, features): computes stats across batch and sequence dims.
    For lower-dim tensors: falls back to global stats.
    """

    """
    if reference_tensor.dim() >= 3:
        dims = tuple(range(reference_tensor.dim() - 1))
        mu = reference_tensor.mean(dim=dims, keepdim=True)
        std = reference_tensor.std(dim=dims, keepdim=True).clamp(min=1e-6)
    else:
        mu = reference_tensor.mean()
        std = reference_tensor.std().clamp(min=1e-6)
    """
    mu = reference_tensor.mean()
    std = reference_tensor.std()
    return torch.randn_like(reference_tensor) * std + mu

class VectorManipulator:
    """
    Handles vector manipulation operations for adversarial TTS.
    Stores audio_data and config_data to avoid repeated parsing.

    Also provides standalone utility functions:
        - add_numbers_pattern: Pads tensors to equal length using a pattern
    """

    def __init__(self, audio_embedding_gt: AudioEmbeddingData, h_text_target: Tensor, config_data: ConfigData):
        self.audio_embedding_gt = audio_embedding_gt
        self.h_text_target = h_text_target
        self.config_data = config_data

    def interpolate(self, interpolation_vectors_batch: Tensor):
        # 1. Get batch slice
        current_batch_size = interpolation_vectors_batch.shape[0]

        # 2. Adjust interpolation vectors
        interpolation_vectors = _adjust_interpolation_vector(
            interpolation_vectors_batch,
            self.config_data.random_matrix,
            self.config_data.subspace_optimization
        )

        # 3. Expand shared data for batch
        input_length = self.audio_embedding_gt.input_length.expand(current_batch_size)
        text_mask = self.audio_embedding_gt.text_mask.expand(current_batch_size, -1)
        h_bert = self.audio_embedding_gt.h_bert.expand(current_batch_size, -1, -1)
        style_vector_acoustic = self.audio_embedding_gt.style_vector_acoustic.expand(current_batch_size, -1)
        style_vector_prosodic = self.audio_embedding_gt.style_vector_prosodic.expand(current_batch_size, -1)

        # 4. Interpolate h_text based on attack mode
        if self.config_data.mode is AttackMode.NOISE_UNTARGETED or self.config_data.mode is AttackMode.TARGETED:
            h_text_mixed = (1.0 - interpolation_vectors) * self.audio_embedding_gt.h_text + interpolation_vectors * self.h_text_target
        else:
            h_text_mixed = self.audio_embedding_gt.h_text + self.config_data.iv_scalar * interpolation_vectors

        return (
            current_batch_size,
            interpolation_vectors,
            AudioEmbeddingData(input_length, text_mask, h_bert, h_text_mixed, style_vector_acoustic, style_vector_prosodic),
        )
