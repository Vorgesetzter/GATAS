import os
import datetime
import platform
import torch
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import requests
from dotenv import load_dotenv

# Local imports
from _helper import adjustInterpolationVector, AttackMode

def finalize_run(optimizer, models, args, run_context, data, device):
    """
    Main entry point to finalize the optimization run.
    Fixed: Uses 'data' instead of 'embedding_data' and 'models' dictionary.
    """
    # 1. Setup Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    objective_tags = [obj.name for obj in run_context['objective_order'] if obj in run_context['active_objectives']]
    objectives_str = "_".join(objective_tags) if objective_tags else "NONE"

    folder_path = os.path.join("outputs", "h_text", objectives_str, timestamp)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Results saved to: {folder_path}")

    # 2. Plot Graphs
    _plot_fitness_history(run_context['fitness_history'], run_context['mean_model'], folder_path)

    # 3. Get Best Candidate & Run Inference
    best_candidate = optimizer.best_candidates[0]

    # Extract specific models from the dictionary
    tts_model = models['tts_model']
    asr_model = models['asr_model']

    results = _run_final_inference(
        best_candidate, tts_model, asr_model, args, data, device
    )

    # 4. Save Audio & Torch State
    _save_artifacts(folder_path, results, best_candidate, data, args, run_context)

    # 5. Write Text Summary
    _write_run_summary(folder_path, args, run_context, results, best_candidate)

    # 6. Notify (WhatsApp)
    if args.notify:
        _send_whatsapp_notification()

    print("Done.")


# ================= INTERNAL HELPERS =================

def _plot_fitness_history(fitness_history, mean_model, folder_path):
    df_means = pd.DataFrame(mean_model)
    fitness_cols = [col for col in df_means.columns if col.endswith("_Mean") and col != "Generation"]

    fig, axs = plt.subplots(len(fitness_cols), 1, figsize=(14, 5 * len(fitness_cols)))
    if len(fitness_cols) == 1:
        axs = [axs]

    fig.suptitle("Evolution of Fitness Objectives (Mean)", fontsize=16)

    for j, col_name in enumerate(sorted(fitness_cols)):
        x_data = df_means["Generation"]
        y_data = df_means[col_name]
        color = "blue" if j % 2 == 0 else "green"
        axs[j].plot(x_data, y_data, color=color, linestyle="--", alpha=0.6, label="Population Mean")
        axs[j].set_title(col_name)
        axs[j].set_xlabel("Generation")
        axs[j].set_ylabel("Fitness")
        axs[j].grid(True, alpha=0.3)
        axs[j].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(folder_path, "graph.png"), dpi=300, bbox_inches="tight")
    plt.close()


def _run_final_inference(best, tts_model, asr_model, args, data, device):
    """
    Reconstructs the best audio and runs ASR.
    Fixed: Uses 'data' instead of 'emb' and 'input_lengths' instead of 'phoneme_count'.
    """
    # Extract phoneme count from input_lengths tensor
    phoneme_count = int(data['input_lengths'].item())

    # Extract & Adjust Vector
    best_vector = torch.from_numpy(best.solution).to(device).float()
    best_vector = best_vector.view(phoneme_count, args.size_per_phoneme)
    best_vector = adjustInterpolationVector(best_vector, data['random_matrix'], args.size_per_phoneme)

    # Mix Embeddings
    mode = data['mode']

    if mode is AttackMode.NOISE_UNTARGETED or mode is AttackMode.TARGETED:
        h_text_mixed_best = (1.0 - best_vector) * data['h_text_gt'] + best_vector * data['h_text_target']
    else:
        # Fixed: Mode-dependent logic for UNTARGETED
        h_text_mixed_best = data['h_text_gt'] + args.iv_scalar * best_vector

    # h_bert logic: StyleTTS2 often uses GT BERT even for mixed text
    h_bert_mixed_best = data['h_bert_gt']

    # Inference
    with torch.no_grad():
        audio_gt = tts_model.inference_after_interpolation(
            data['input_lengths'], data['text_mask'], data['h_bert_gt'], data['h_text_gt'],
            data['style_vector_acoustic'], data['style_vector_prosodic']
        )
        audio_target = tts_model.inference_after_interpolation(
            data['input_lengths'], data['text_mask'],
            data.get('h_bert_target', data['h_bert_gt']), data['h_text_target'],
            data['style_vector_acoustic'], data['style_vector_prosodic']
        )
        audio_best = tts_model.inference_after_interpolation(
            data['input_lengths'], data['text_mask'], h_bert_mixed_best, h_text_mixed_best,
            data['style_vector_acoustic'], data['style_vector_prosodic']
        )

    # ASR Analysis
    asr_final, conf_final = asr_model.analyzeAudio(audio_best)

    return {
        "audio_gt": audio_gt,
        "audio_target": audio_target,
        "audio_best": audio_best,
        "asr_text": asr_final["text"].strip(),
        "asr_conf": conf_final
    }


def _save_artifacts(folder_path, results, best, data, args, run_context):
    """Fixed: Uses 'data' dictionary keys."""
    # Save Audio
    sf.write(os.path.join(folder_path, "ground_truth.wav"), results['audio_gt'], samplerate=24000)
    sf.write(os.path.join(folder_path, "target.wav"), results['audio_target'], samplerate=24000)
    sf.write(os.path.join(folder_path, "interpolated.wav"), results['audio_best'], samplerate=24000)

    # Save Torch State
    state_dict = {
        "interpolation_vector": torch.tensor(best.solution).float().cpu(),
        "random_matrix": data['random_matrix'],
        "AttackMode": data['mode'].name,
        "Active Objectives": [obj.name for obj in run_context['active_objectives']],
        "size_per_phoneme": args.size_per_phoneme,
        "IV_scalar": args.iv_scalar,
        "fitness_values": best.fitness,
        "text_gt": args.ground_truth_text,
        "text_target": args.target_text,
        "asr_text": results['asr_text'],
        "num_generations": args.num_generations,
        "population_size": args.pop_size,
        "generation_found": getattr(best, "generation", "Unknown"),
    }
    torch.save(state_dict, os.path.join(folder_path, "best_vector.pt"))


def _write_run_summary(folder_path, args, run_context, results, best):
    # Hardware & Timing Logic
    os_info = f"{platform.system()} {platform.release()}"
    cpu_info = platform.processor()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        hardware_str = f"GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)\n  CPU: {cpu_info}\n  OS:  {os_info}"
    else:
        hardware_str = f"GPU: None (CPU Only)\n  CPU: {cpu_info}\n  OS:  {os_info}"

    pbar = run_context['progress_bar']
    rate = pbar.format_dict['rate']
    elapsed = pbar.format_dict['elapsed']
    time_per_gen = (1.0 / rate) if rate and rate > 0 else 0.0

    active_in_order = [obj for obj in run_context['objective_order'] if obj in run_context['active_objectives']]

    summary_path = os.path.join(folder_path, "run_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Adversarial TTS Optimization Summary ===\n")
        f.write(f"\n--- Texts ---\n")
        f.write(f"Ground Truth Text: {args.ground_truth_text}\n")
        f.write(f"Target Text: {args.target_text}\n")

        f.write(f"\n--- Variable Values ---\n")
        f.write(f"AttackMode: {args.mode}\n")
        f.write(f"Active Objectives: {', '.join([obj.name for obj in active_in_order])}\n")
        f.write(f"Population size: {args.pop_size}\n")
        f.write(f"Size per phoneme: {args.size_per_phoneme}\n")
        f.write(f"IV_scalar: {args.iv_scalar}\n")

        if run_context.get('thresholds'):
            t_str = ", ".join([f"{k.name}<={v}" for k, v in run_context['thresholds'].items()])
            f.write(f"Thresholds Set:    {t_str}\n")
            f.write(f"Stopped Early:     {'Yes' if run_context['stop_optimization'] else 'No'}\n")
        else:
            f.write(f"Thresholds Set:    None (Ran full {args.num_generations} gens)\n")

        f.write(f"Generations Run:   {run_context['current_gen'] + 1}/{args.num_generations}\n")

        f.write(f"\n--- Performance ---\n")
        f.write(f"{hardware_str}\n")
        f.write(f"Total Time Duration: {elapsed:.2f}s\n")
        f.write(f"Time per Generation: {time_per_gen:.2f}s\n")

        f.write(f"\n--- Fitness Values ---\n")
        f.write(f"Generation best candidate found: {getattr(best, 'generation', 'Unknown')}\n\n")
        f.write("Final fitness values (best candidate):\n")

        for obj, score in zip(active_in_order, best.fitness):
            f.write(f"  {obj.name}: {float(score):.6f}\n")

        f.write(f"\nASR transcription: \"{results['asr_text']}\"\n")
        f.write(f"ASR confidence: {float(results['asr_conf']):.6f}\n")


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