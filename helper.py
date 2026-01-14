import os

from torch import Tensor
import torch
import whisper
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm


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


# Add this to helper.py
import numpy as np


def get_pareto_mask(fitness_matrix):
    """
    Returns a boolean mask of shape (N,) where True indicates the row is
    non-dominated (Pareto efficient).
    """
    population_size = fitness_matrix.shape[0]
    is_efficient = np.ones(population_size, dtype=bool)

    for i in range(population_size):
        if is_efficient[i]:
            current_candidate = fitness_matrix[i]
            all_better_or_equal = np.all(fitness_matrix <= current_candidate, axis=1)
            any_strictly_better = np.any(fitness_matrix < current_candidate, axis=1)

            dominators = all_better_or_equal & any_strictly_better

            if np.any(dominators):
                is_efficient[i] = False

    return is_efficient

def get_local_pareto_front(fitness_matrix):
    """Returns only the non-dominated rows from a fitness matrix."""
    is_efficient = np.ones(fitness_matrix.shape[0], dtype=bool)
    for i, c in enumerate(fitness_matrix):
        if is_efficient[i]:
            # Keep only individuals not dominated by others
            # (Assuming minimization; if PESQ is maximized, multiply it by -1 first)
            is_efficient[is_efficient] = np.any(fitness_matrix[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return fitness_matrix[is_efficient]

def calculate_2d_hypervolume(pareto_front, ref_point):
    """
    Calculates the area (Hypervolume) for a 2D Pareto front.
    pareto_front: np.ndarray of shape (N, 2)
    ref_point: list or array [r1, r2] (the 'worst' possible values)
    """
    if pareto_front.size == 0:
        return 0.0

    # 1. Sort the front by the first objective
    front = pareto_front[pareto_front[:, 0].argsort()]

    # 2. Ensure all points are within the reference point bounds
    # (Ignore points worse than the reference point)
    mask = (front[:, 0] <= ref_point[0]) & (front[:, 1] <= ref_point[1])
    front = front[mask]

    if len(front) == 0:
        return 0.0

    # 3. Calculate the area of the rectangles
    area = 0.0
    last_y = ref_point[1]

    for x, y in front:
        # Area = Width (distance to ref_x) * Height (distance between steps)
        area += (ref_point[0] - x) * (last_y - y)
        last_y = y

    return area

def send_whatsapp_notification():
    load_dotenv()
    phone = os.getenv("WHATSAPP_PHONE_NUMBER")
    apikey = os.getenv("WHATSAPP_API_KEY")
    text = "Optimization finished! Check the results folder."

    if not phone or not apikey:
        tqdm.write("[!] Cannot send WhatsApp: Missing env variables.")
        return

    url = f"https://api.callmebot.com/whatsapp.php?phone={phone}&text={text}&apikey={apikey}"
    try:
        requests.get(url, timeout=10)
        tqdm.write("WhatsApp notification sent.")
    except Exception as e:
        tqdm.write(f"Error sending WhatsApp: {e}")

