from torch import Tensor
import torch
import whisper

def _asr_batch_inference(asr_model, audio_batch, device):
    """
    Führt Whisper ASR auf einem Batch von Audio-Tensoren aus.
    """
    # 1. Padden/Trimmen & Log-Mel Spectrograms (Sequenziell ist ok, da sehr schnell)
    mels = []
    for audio in audio_batch:
        # Whisper erwartet 30s Audio (16k * 30 = 480,000 Samples)
        # Pad/Trim Logic
        audio = whisper.pad_or_trim(audio)

        # Log-Mel
        mel = whisper.log_mel_spectrogram(audio).to(device)
        mels.append(mel)

    # 2. Stacken zu [Batch_Size, 80, 3000]
    batch_mels = torch.stack(mels)

    # 3. Batch Inferenz (Das ist der Speed-Boost!)
    options = whisper.DecodingOptions(fp16=False, language='en')

    # decode() gibt eine Liste von Results zurück
    results = whisper.decode(asr_model, batch_mels, options)

    # 4. Text & LogProbs extrahieren
    texts = [r.text for r in results]
    avg_logprobs = [r.avg_logprob for r in results]  # Whisper liefert avg_logprob direkt mit

    return texts, avg_logprobs

def length_to_mask(lengths: Tensor) -> Tensor:
    mask = torch.arange(lengths.max())  # Creates a Vector [0,1,2,3,...,x], where x = biggest value in lengths
    mask = mask.unsqueeze(0)  # Creates a Matrix [1,x] from Vector [x]
    mask = mask.expand(lengths.shape[0], -1)  # Expands the matrix from [1,x] to [y,x], where y = number of elements in lengths
    mask = mask.type_as(lengths)  # Assign mask the same type as lengths
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))  # gt = greater than, compares each value from lengths to a row of values in mask; unsqueeze = splits vector lengths into vectors of size 1
    return mask  # returns a mask of shape (batch_size, max_length) where mask[i, j] = 1 if j < lengths[i] and mask[i, j] = 0 otherwise.

def _pad_with_pattern(tensor: Tensor, amount: int, pattern: list[int]) -> Tensor:

    padding = torch.as_tensor(pattern, device=tensor.device, dtype=tensor.dtype)[torch.arange(amount, device=tensor.device) % len(pattern)]

    for _ in range(tensor.dim() - 1):
        padding = padding.unsqueeze(0)
    padding = padding.expand(*tensor.shape[:-1], amount)

    return torch.cat([tensor, padding], dim=-1)

def addNumbersPattern(a: Tensor, b: Tensor, pattern: list[int]) -> tuple[Tensor, Tensor]:

    len_a = a.size(-1)
    len_b = b.size(-1)

    # If equal: nothing to do
    if len_a == len_b:
        return a, b

    # Determine which tensor needs padding
    if len_a < len_b:
        a = _pad_with_pattern(a, len_b - len_a, pattern)
    else:
        b = _pad_with_pattern(b, len_a - len_b, pattern)

    return a, b

def _extend_to_size(x: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Extends the last dimension of x to `target_size` by repeating elements.
    Supports inputs of any dimension (e.g., [Batch, Dim] or [Batch, 1, Dim]).
    """
    # FIX: Get only the last dimension 'a', ignore preceding dimensions
    a = x.shape[-1]

    # If already large enough, just crop
    if a >= target_size:
        return x[..., :target_size] # Use ... to keep all leading dimensions

    # How many repeats per original position?
    base = target_size // a
    rem = target_size % a

    repeats = torch.full((a,), base, device=x.device, dtype=torch.long)
    if rem > 0:
        repeats[:rem] += 1

    # Build index pattern
    idx = torch.arange(a, device=x.device).repeat_interleave(repeats)
    assert idx.numel() == target_size

    # Apply index pattern to the last dimension using Ellipsis (...)
    return x[..., idx]

def adjustInterpolationVector(IV: Tensor, matrix: Tensor, subspace_optimization: bool) -> Tensor:

    # Matrix Multiplication, since IV not Scalar Value
    if IV.shape[-1] != 1:
        if subspace_optimization:
            IV = IV @ matrix
        else:
            IV = _extend_to_size(IV, 512)

    # 1. Swap the last two dimensions (Works for [L, C] or [B, L, C])
    IV = IV.transpose(-1, -2)

    # 2. Add batch dimension only if it's missing (i.e., if it is 2D)
    if IV.dim() < 3:
        IV = IV.unsqueeze(0)

    return IV

