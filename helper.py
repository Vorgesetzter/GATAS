import os

from torch import Tensor
import torch
import whisper
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

import numpy as np
import soundfile as sf

from Datastructures.enum import AttackMode

def length_to_mask(lengths: Tensor) -> Tensor:
    mask = torch.arange(lengths.max())  # Creates a Vector [0,1,2,3,...,x], where x = biggest value in lengths
    mask = mask.unsqueeze(0)  # Creates a Matrix [1,x] from Vector [x]
    mask = mask.expand(lengths.shape[0], -1)  # Expands the matrix from [1,x] to [y,x], where y = number of elements in lengths
    mask = mask.type_as(lengths)  # Assign mask the same type as lengths
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))  # gt = greater than, compares each value from lengths to a row of values in mask; unsqueeze = splits vector lengths into vectors of size 1
    return mask  # returns a mask of shape (batch_size, max_length) where mask[i, j] = 1 if j < lengths[i] and mask[i, j] = 0 otherwise.

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

def save_audio(audio, file_path):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy().squeeze()
    sf.write(file_path, audio, samplerate=16000)



