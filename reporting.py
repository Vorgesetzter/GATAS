import datetime
import platform
import pandas as pd
import soundfile as sf
import requests
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

from _dataclass import *
# Local imports
from _helper import adjustInterpolationVector
from _enum import AttackMode

from Scripts._optimizer_candidate import OptimizerCandidate

def finalize_run(config_data, fitness_data, model_data, audio_data, progress_bar, gen, device):

    # 1. Setup Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    objectives_str = "_".join([obj.name for obj in config_data.active_objectives]) if config_data.active_objectives else "NONE"

    folder_path = os.path.join("outputs", "h_text", objectives_str, timestamp)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Results saved to: {folder_path}")

    # 2. Plot Graphs
    _generate_all_visualizations(config_data, fitness_data, folder_path)

    # 3. Get Best Candidate & Run Inference
    best_candidate = _select_best_candidate(model_data.optimizer.best_candidates)

    print("Best candidate fitness values:")
    for obj, score in zip(config_data.active_objectives, best_candidate.fitness):
        print(f"  {obj.name}: {score:.6f}")
    print()

    best_mixed_audio = _run_final_inference(
        best_candidate, model_data.tts_model, model_data.asr_model, audio_data, config_data, device
    )

    # 4. Save Audio & Torch State
    _save_artifacts(folder_path, best_mixed_audio, audio_data, config_data, best_candidate)

    # 5. Write Text Summary
    _write_run_summary(folder_path, config_data, best_mixed_audio, best_candidate, progress_bar, gen)

    # 6. Notify (WhatsApp)
    if config_data.notify:
        _send_whatsapp_notification()

    print("Done.")


# ================= INTERNAL HELPERS =================

def _generate_all_visualizations(config_data, fitness_data, folder_path):

    active_objectives = config_data.active_objectives

    # Image 1: Pareto Evolution (If 2 objectives)
    if len(fitness_data.total_fitness) >= 4 and len(active_objectives) == 2:
        _generate_pareto_population_graph(
            fitness_data.total_fitness,
            active_objectives,
            folder_path
        )

    # Image 2: Mean Progress
    _generate_mean_population_graph(
        fitness_data.mean_fitness,
        active_objectives,
        folder_path,
    )

    # Image 3: Population Evolution
    _generate_total_population_graph(
        fitness_data.total_fitness,
        active_objectives,
        folder_path
    )

def _generate_pareto_population_graph(total_fitness, active_objectives, folder_path):

    total_gens = len(total_fitness)

    # 1. Determine which 4 generations to plot
    # We use linspace to find 4 equidistant indices
    indices = np.linspace(0, total_gens - 1, 4, dtype=int)

    # 2. Setup Single Plot
    obj_names = [obj.name for obj in active_objectives]
    fig, ax = plt.subplots(figsize=(12, 10))

    # Generate 4 distinct colors using a colormap (e.g., 'viridis', 'plasma', 'coolwarm')
    # 0.0 = Start (Purple/Blue), 1.0 = End (Yellow/Red) depending on map
    colors = cm.viridis(np.linspace(0, 1, len(indices)))

    fig.suptitle(f"Pareto Front Evolution: {obj_names[0]} vs {obj_names[1]}", fontsize=18)

    # 3. Plot the 4 snapshots on the SAME axis
    for i, (idx, color) in enumerate(zip(indices, colors)):

        # Extract data
        fitness = _get_local_pareto_front(total_fitness[idx])

        # Safety check
        if fitness.size == 0 or fitness.shape[1] < 2:
            continue

        # Sort by first objective so the connecting line is clean, not a web
        fitness = fitness[fitness[:, 0].argsort()]

        # Create Label (e.g., "Gen 1 (0%)" or "Gen 50 (33%)")
        label_text = f"Gen {idx + 1} ({(idx + 1) / total_gens:.0%})"

        # Plot Scatter (Dots)
        ax.scatter(fitness[:, 0], fitness[:, 1], color=color, s=60, alpha=0.8, edgecolors='k', label=label_text, zorder=i + 2)

        # Plot Line (Connection)
        # alpha=0.4 ensures lines don't distract too much from the points
        ax.plot(fitness[:, 0], fitness[:, 1], color=color, linestyle='--', alpha=0.5, linewidth=1.5, zorder=i + 1)

    # 4. Final Styling
    ax.set_xlabel(obj_names[0], fontsize=12)
    ax.set_ylabel(obj_names[1], fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)

    # Add a legend to explain the colors
    ax.legend(title="Evolution Progress", fontsize=10, title_fontsize=12, loc='best')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save
    save_path = os.path.join(folder_path, "pareto_evolution_single.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Pareto graph saved to: {save_path}")

def _get_local_pareto_front(fitness_matrix: np.ndarray) -> np.ndarray:
    """Returns only the non-dominated rows from a fitness matrix."""
    is_efficient = np.ones(fitness_matrix.shape[0], dtype=bool)
    for i, c in enumerate(fitness_matrix):
        if is_efficient[i]:
            # Keep only individuals not dominated by others
            # (Assuming minimization; if PESQ is maximized, multiply it by -1 first)
            is_efficient[is_efficient] = np.any(fitness_matrix[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return fitness_matrix[is_efficient]

def _generate_mean_population_graph(mean_history, active_objectives, folder_path):
    """
    Generates a vertical stack of graphs showing the Mean fitness for each active objective.
    """
    df = pd.DataFrame(mean_history)

    fig, axs = plt.subplots(len(active_objectives), 1, figsize=(12, 5 * len(active_objectives)), squeeze=False)
    fig.suptitle("Mean Fitness Evolution per Objective", fontsize=18)

    for i, obj in enumerate(active_objectives):
        ax = axs[i, 0]
        col_name = f"{obj.name}_Mean"

        if col_name in df.columns:
            ax.plot(df["Generation"], df[col_name], color='blue', linewidth=2, label="Population Mean")
            ax.set_title(f"Objective: {obj.name}", fontsize=14)
            ax.set_ylabel("Fitness Score")
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"Data for {obj.name} not found", ha='center')

    plt.xlabel("Generation")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path = os.path.join(folder_path, "mean_fitness_stack.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Mean population graph saved to: {save_path}")


def _generate_total_population_graph(history_pop_fitness, active_objectives, folder_path):
    """
    Creates a SINGLE graph overlaying the Full Population Cloud at Start, 33%, 66%, and End.
    Uses Jitter to reveal stacked/duplicate individuals.
    """
    if len(active_objectives) < 2:
        print("Skipping Population Cloud Graph: Requires at least 2 active objectives.")
        return

    total_gens = len(history_pop_fitness)
    if total_gens == 0:
        print("Skipping Population Cloud Graph: History is empty.")
        return

    # 1. Determine which 4 generations to plot (Start -> End)
    indices = np.linspace(0, total_gens - 1, 4, dtype=int)

    # 2. Setup Plot
    obj_names = [obj.name for obj in active_objectives]
    fig, ax = plt.subplots(figsize=(12, 10))

    # Generate colors (Purple=Early -> Yellow=Late)
    colors = cm.viridis(np.linspace(0, 1, len(indices)))

    fig.suptitle(f"Population Cloud Evolution: {obj_names[0]} vs {obj_names[1]}", fontsize=18)

    # 3. Plot the 4 snapshots
    for i, (idx, color) in enumerate(zip(indices, colors)):

        # Extract the Raw Matrix (Shape: [Pop_Size, 2])
        # This contains ALL individuals, including identical clones.
        F = history_pop_fitness[idx]

        if F.size == 0 or F.shape[1] < 2:
            continue

        # --- JITTER LOGIC ---
        # Add small random noise to separate stacked dots
        # Scale: 0.002 is usually good for metrics like WER/PESQ (range 0-1)
        # If your dots are still stacked, increase this to 0.005
        jitter_x = np.random.normal(0, 0.002, size=F.shape[0])
        jitter_y = np.random.normal(0, 0.002, size=F.shape[0])

        x_coords = F[:, 0] + jitter_x
        y_coords = F[:, 1] + jitter_y
        # --------------------

        label_text = f"Gen {idx + 1} ({(idx + 1) / total_gens:.0%})"

        # Plot Scatter
        # alpha=0.5 helps show density (darker areas = more individuals piled up)
        ax.scatter(x_coords, y_coords, color=color, s=40, alpha=0.5, label=label_text, zorder=i + 2)

    # 4. Final Styling
    ax.set_xlabel(obj_names[0], fontsize=12)
    ax.set_ylabel(obj_names[1], fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)

    # Add legend
    ax.legend(title="Evolution Progress", fontsize=10, title_fontsize=12, loc='best')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save
    save_path = os.path.join(folder_path, "population_cloud_evolution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Population cloud graph saved to: {save_path}")

def _select_best_candidate(candidates):
    """
    Identifies the 'Knee Point' (best trade-off) from a list of OptimizerCandidates.
    Returns a single OptimizerCandidate object.
    """

    if not candidates:
        raise ValueError("Candidate list is empty.")

    # 1. Extract fitness tuples into a NumPy array for math operations
    # Shape will be (num_candidates, num_objectives)
    f = np.array([c.fitness for c in candidates])

    # 3. Calculate Euclidean distance to the 'Ideal Point' (0, 0, ...)
    # Since we minimize, the origin is the theoretical perfect score.
    distances = np.linalg.norm(f, axis=1)

    # 4. Find the index of the candidate with the minimum distance
    knee_idx = np.argmin(distances)

    # Return the actual dataclass object at that index
    return candidates[knee_idx]

def _run_final_inference(best_candidate, tts_model, asr_model, audio_data, config_data, device):
    """
    Reconstructs the best audio and runs ASR.
    Fixed: Uses 'data' instead of 'emb' and 'input_lengths' instead of 'phoneme_count'.
    """
    # Extract phoneme count from input_lengths tensor
    phoneme_count = int(audio_data.input_lengths.item())

    # Extract & Adjust Vector
    best_vector = torch.from_numpy(best_candidate.solution).to(device).float()
    best_vector = best_vector.view(phoneme_count, config_data.size_per_phoneme)
    best_vector = adjustInterpolationVector(best_vector, config_data.random_matrix, config_data.size_per_phoneme)

    # Mix Embeddings
    if config_data.mode is AttackMode.NOISE_UNTARGETED or config_data.mode is AttackMode.TARGETED:
        h_text_mixed_best = (1.0 - best_vector) * audio_data.h_text_gt + best_vector * audio_data.h_text_target
    else:
        # Fixed: Mode-dependent logic for UNTARGETED
        h_text_mixed_best = audio_data.h_text_gt + config_data.iv_scalar * best_vector

    # h_bert logic: StyleTTS2 often uses GT BERT even for mixed text
    h_bert_mixed_best = audio_data.h_bert_gt

    # Inference
    with torch.no_grad():
        audio_best = tts_model.inference_after_interpolation(
            audio_data.input_lengths,
            audio_data.text_mask,
            h_bert_mixed_best,
            h_text_mixed_best,
            audio_data.style_vector_acoustic,
            audio_data.style_vector_prosodic
        )

    # ASR Analysis
    asr_result, asr_logprob = asr_model.analyzeAudio(audio_best)

    return BestMixedAudio(
        audio=audio_best,
        text=asr_result["text"].strip(),
        logprob=float(asr_logprob),
        h_text=h_text_mixed_best,
        h_bert=h_bert_mixed_best
    )

def _save_artifacts(folder_path, best_mixed_audio, audio_data, config_data, best_candidate):

    """Fixed: Uses 'data' dictionary keys."""
    # Save Audio
    sf.write(os.path.join(folder_path, "ground_truth.wav"), audio_data.audio_gt, samplerate=24000)
    sf.write(os.path.join(folder_path, "target.wav"), audio_data.audio_target, samplerate=24000)
    sf.write(os.path.join(folder_path, "interpolated.wav"), best_mixed_audio.audio, samplerate=24000)

    # Save Torch State
    state_dict = {
        # 1. Variables from Configuration (Static)
        "AttackMode": config_data.mode.name,
        "Active Objectives": [obj.name for obj in config_data.active_objectives],
        "size_per_phoneme": config_data.size_per_phoneme,
        "IV_scalar": config_data.iv_scalar,
        "num_generations": config_data.num_generations,
        "population_size": config_data.pop_size,
        "text_gt": config_data.text_gt,
        "text_target": config_data.text_target,
        "random_matrix": config_data.random_matrix,

        # 2. Variables from Optimizer (The "Best" Candidate)
        "interpolation_vector": torch.from_numpy(best_candidate.solution).float().cpu(),
        "fitness_values": best_candidate.fitness,
        "generation_found": getattr(best_candidate, "generation", "Unknown"),

        # 3. Results from Final Inference (BestMixedAudio)
        "asr_text": best_mixed_audio.text,
        "asr_logprob": best_mixed_audio.logprob,
        "h_text_mixed_best": best_mixed_audio.h_text,
        "h_bert_mixed_best": best_mixed_audio.h_bert,

        # 4. Audio Constants for Reproducibility (AudioData)
        "input_lengths": audio_data.input_lengths,
        "text_mask": audio_data.text_mask,
        "style_vector_acoustic": audio_data.style_vector_acoustic,
        "style_vector_prosodic": audio_data.style_vector_prosodic,
        "noise": audio_data.noise
    }

    torch.save(state_dict, os.path.join(folder_path, "best_vector.pt"))


def _write_run_summary(folder_path: str, config_data: ConfigData, best_mixed_audio: BestMixedAudio, best_candidate: OptimizerCandidate, progress_bar, gen):

    # 1. Hardware Detection
    os_info = f"{platform.system()} {platform.release()}"
    cpu_info = platform.processor()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        hardware_str = f"GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)\n  CPU: {cpu_info}\n  OS:  {os_info}"
    else:
        hardware_str = f"GPU: None (CPU Only)\n  CPU: {cpu_info}\n  OS:  {os_info}"

    # 2. Timing Logic (Extracted from the progress_bar dict)
    rate = progress_bar.format_dict['rate']
    elapsed = progress_bar.format_dict['elapsed']
    time_per_gen = (1.0 / rate) if rate and rate > 0 else 0.0

    summary_path = os.path.join(folder_path, "run_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Adversarial TTS Optimization Summary ===\n")

        f.write(f"\n--- Texts ---\n")
        f.write(f"Ground Truth Text: {config_data.text_gt}\n")
        f.write(f"Target Text:       {config_data.text_target}\n")

        f.write(f"\n--- Configuration ---\n")
        f.write(f"AttackMode:        {config_data.mode.name}\n")
        f.write(f"Active Objectives: {', '.join([obj.name for obj in config_data.active_objectives])}\n")
        f.write(f"Population Size:   {config_data.pop_size}\n")
        f.write(f"Size Per Phoneme:  {config_data.size_per_phoneme}\n")
        f.write(f"IV Scalar:         {config_data.iv_scalar}\n")
        f.write(f"Subspace Opt:      {config_data.subspace_optimization}\n")

        if config_data.thresholds:
            t_str = ", ".join([f"{k.name}<={v}" for k, v in config_data.thresholds.items()])
            f.write(f"Thresholds Set:    {t_str}\n")
        else:
            f.write(f"Thresholds Set:    None (Ran full {config_data.num_generations} gens)\n")

        f.write(f"Generations Run:   {gen + 1}/{config_data.num_generations}\n")

        f.write(f"\n--- Performance ---\n")
        f.write(f"{hardware_str}\n")
        f.write(f"Total Duration:    {elapsed:.2f}s\n")
        f.write(f"Avg per Gen:       {time_per_gen:.2f}s\n")

        f.write(f"\n--- Best Candidate Fitness ---\n")
        f.write(f"Generation Found:  {getattr(best_candidate, 'generation', 'Unknown')}\n\n")

        for obj, score in zip(config_data.active_objectives, best_candidate.fitness):
            f.write(f"  {obj.name}: {float(score):.6f}\n")

        f.write(f"\n--- ASR Final Result ---\n")
        f.write(f"Transcription:     \"{best_mixed_audio.text}\"\n")
        f.write(f"Confidence/Logprob: {best_mixed_audio.logprob:.6f}\n")

def _send_whatsapp_notification():
    load_dotenv()
    phone = os.getenv("WHATSAPP_PHONE_NUMBER")
    apikey = os.getenv("WHATSAPP_API_KEY")
    text = "Optimization finished! Check the results folder."

    if not phone or not apikey:
        print("[!] Cannot send WhatsApp: Missing env variables.")
        return

    url = f"https://api.callmebot.com/whatsapp.php?phone={phone}&text={text}&apikey={apikey}"
    try:
        requests.get(url, timeout=10)
        print("WhatsApp notification sent.")
    except Exception as e:
        print(f"Error sending WhatsApp: {e}")