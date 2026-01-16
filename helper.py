import os

from torch import Tensor
import torch
import whisper
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm
import platform
import numpy as np

from Datastructures.enum import AttackMode


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

def write_run_summary(folder_path, text_best, candidate, gen_count, elapsed_time, config_data):

    # 1. System Metadata
    os_info = f"{platform.system()} {platform.release()}"
    gpu_info = "CPU Only"
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        gpu_info = f"{torch.cuda.get_device_name(0)} ({vram:.2f} GB VRAM)"

    avg_per_gen = elapsed_time / gen_count if gen_count > 0 else 0

    summary_path = os.path.join(folder_path, "run_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(" ADVERSARIAL TTS OPTIMIZATION REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("--- [1] INPUT DATA ---\n")
        # Assuming these paths are stored in your config or audio_data
        f.write(f"GT Text:      {config_data.text_gt}\n")
        f.write(f"Target Text:  {config_data.text_target if config_data.text_target else '[NONE]'}\n")

        f.write("\n--- [2] CLI ARGUMENTS & CONFIG ---\n")
        f.write(f"Attack Mode:       {config_data.mode.name}\n")
        f.write(f"Objectives:        {', '.join([obj.name for obj in config_data.active_objectives])}\n")
        f.write(f"Population Size:   {config_data.pop_size}\n")
        f.write(f"Size Per Phoneme:  {config_data.size_per_phoneme}\n")
        f.write(f"IV Scalar:         {config_data.iv_scalar}\n")
        f.write(f"Subspace Opt:      {config_data.subspace_optimization}\n")

        if config_data.thresholds:
            t_str = ", ".join([f"{k.name} <= {v}" for k, v in config_data.thresholds.items()])
            f.write(f"Early Stopping:    {t_str}\n")
        else:
            f.write(f"Early Stopping:    Off (Ran full duration)\n")

        f.write("\n--- [3] PERFORMANCE & HARDWARE ---\n")
        f.write(f"Hardware:          {gpu_info}\n")
        f.write(f"OS/CPU:            {os_info} | {platform.processor()}\n")
        f.write(f"Gens Completed:    {gen_count}\n")
        f.write(f"Total Time:        {elapsed_time:.2f}s\n")
        f.write(f"Efficiency:        {avg_per_gen:.2f}s per generation\n")

        f.write("\n--- [4] BEST CANDIDATE RESULTS ---\n")
        f.write(f"Selection Metric:  Euclidean Distance to Origin (Knee Point)\n")
        f.write(f"Generation Found:  {getattr(candidate, 'generation', 'Unknown')}\n")
        f.write("-" * 30 + "\n")

        # Detailed Fitness Scores
        for obj, score in zip(config_data.active_objectives, candidate.fitness):
            f.write(f"  {obj.name:<15}: {float(score):.8f}\n")

        f.write("-" * 30 + "\n")
        f.write(f"Final Transcription: \"{text_best}\"\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write(" END OF REPORT\n")
        f.write("=" * 50 + "\n")

