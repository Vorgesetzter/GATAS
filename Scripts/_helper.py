import torch
from torch import Tensor
import math

def length_to_mask(lengths: Tensor) -> Tensor:
    mask = torch.arange(lengths.max())  # Creates a Vector [0,1,2,3,...,x], where x = biggest value in lengths
    mask = mask.unsqueeze(0)  # Creates a Matrix [1,x] from Vector [x]
    mask = mask.expand(lengths.shape[0], -1)  # Expands the matrix from [1,x] to [y,x], where y = number of elements in lengths
    mask = mask.type_as(lengths)  # Assign mask the same type as lengths
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))  # gt = greater than, compares each value from lengths to a row of values in mask; unsqueeze = splits vector lengths into vectors of size 1
    return mask  # returns a mask of shape (batch_size, max_length) where mask[i, j] = 1 if j < lengths[i] and mask[i, j] = 0 otherwise.

def addNumbersPattern(to_change: Tensor, reference: Tensor, pattern: list[int]) -> Tensor:

    diff = reference.size(-1) - to_change.size(-1)

    if diff < 0:
        print("Reference is smaller then whats to be changed")
        return to_change

    padding = torch.as_tensor(pattern, device=to_change.device, dtype=to_change.dtype)[torch.arange(diff, device=to_change.device) % len(pattern)]

    # Match batch dimensions, same logic as torch.full
    for _ in range(to_change.dim() - 1):
        padding = padding.unsqueeze(0)
    padding = padding.expand(*to_change.shape[:-1], diff)

    # Concatenate along last dimension
    to_change = torch.cat([to_change, padding], dim=-1)

    return to_change

def adjustInterpolationVector(IV: Tensor, matrix: Tensor, size_per_phoneme: int) -> Tensor:

    # Matrix Multiplication, since IV not Scalar Value
    if size_per_phoneme != 1:
        IV = IV @ matrix

    IV = IV.unsqueeze(0)
    IV = IV.permute(0, 2, 1)

    return interpolation_vector

def interpolateWithScalar(a: Tensor, b: Tensor, alpha: float) -> Tensor:
    return (1 - alpha) * a + alpha * b

def text_naturalness_from_ppl(ppl, min_loss=1.0, max_loss=10.0):
    """
    ppl: perplexity from your LM
    Returns score in [0,1], where 1 = very natural/common.
    """
    loss = math.log(ppl)             # log PPL = cross-entropy-ish
    loss = max(min(loss, max_loss), min_loss)
    return 1.0 - (loss - min_loss) / (max_loss - min_loss)

