"""
RQ3 Analysis — Efficiency: TTS vs Waveform

Compares computational cost of both adversarial methods:
  - Time per generation, time per run, time per successful run
  - Generations to convergence (successful runs only)
  - Convergence distribution bar chart (with mean / median lines)
  - Cumulative success rate over generations
  - rq3_efficiency_analysis.json with all statistics

Usage:
    python Scripts/Analysis/rq3_analysis.py
    python Scripts/Analysis/rq3_analysis.py \\
        --tts_csv  outputs/results/TTS/all_results.csv \\
        --wf_csv   outputs/results/Waveform/all_results.csv \\
        --output_dir outputs/thesis_analysis/RQ3
"""

import os
import sys
import json
import shutil
import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from plot_utils import C_TTS, C_WAVEFORM, boxplot_colors, save_fig

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

DPI = 150  # efficiency figures are large; 150 is sufficient

TTS_CSV = "outputs/results/TTS/all_results.csv"
WF_CSV  = "outputs/results/Waveform/all_results.csv"


# =============================================================================
# Section 1 — Efficiency Metrics
# =============================================================================

def compute_efficiency(df: pd.DataFrame, method_name: str) -> dict:
    """Compute efficiency metrics for one method."""
    successful = df[df["success"] == True]
    failed     = df[df["success"] == False]

    all_gens = (
        list(successful["generation_found"].values)
        + list(failed["num_generations"].values)
    )

    df = df.copy()
    df["time_per_generation"] = df["elapsed_time_seconds"] / df["num_generations"]
    df["generations_used"] = df.apply(
        lambda r: r["generation_found"] if r["success"] else r["num_generations"], axis=1
    )

    n_succ = len(successful)
    return {
        "method":                        method_name,
        "n_runs":                        len(df),
        "success_rate":                  round(float(df["success"].mean() * 100), 2),
        "total_successful":              int(n_succ),
        "total_time_hours":              round(float(df["elapsed_time_seconds"].sum() / 3600), 2),
        "avg_time_per_run_sec":          round(float(df["elapsed_time_seconds"].mean()), 1),
        "median_time_per_run_sec":       round(float(df["elapsed_time_seconds"].median()), 1),
        "std_time_per_run_sec":          round(float(df["elapsed_time_seconds"].std()), 1),
        "avg_generations_overall":       round(float(np.mean(all_gens)), 2),
        "median_generations_successful": round(float(successful["generation_found"].median()), 1) if n_succ else None,
        "std_generations_successful":    round(float(successful["generation_found"].std()), 2)    if n_succ else None,
        "max_generations_successful":    int(successful["generation_found"].max())                if n_succ else None,
        "total_queries":                 int(df["generation_count"].sum()),
        "avg_queries_per_run":           round(float(df["generation_count"].mean()), 1),
        "avg_time_per_generation_sec":   round(float(df["time_per_generation"].mean()), 2),
        "median_time_per_generation_sec": round(float(df["time_per_generation"].median()), 2),
        "cost_per_successful_run_sec":   round(float(
            successful["elapsed_time_seconds"].mean() if n_succ else float("nan")
        ), 1),
    }


def compute_comparison(tts: dict, wf: dict) -> dict:
    def _ratio(a, b):
        return round(float(a / b), 3) if b and b != 0 else None
    return {
        "time_per_run_ratio":   _ratio(wf["avg_time_per_run_sec"],         tts["avg_time_per_run_sec"]),
        "generations_ratio":    _ratio(wf["avg_generations_overall"],      tts["avg_generations_overall"]),
        "time_per_gen_ratio":   _ratio(wf["avg_time_per_generation_sec"],  tts["avg_time_per_generation_sec"]),
        "success_rate_ratio":   _ratio(wf["success_rate"],                 tts["success_rate"]),
        "total_time_ratio":     _ratio(wf["total_time_hours"],             tts["total_time_hours"]),
        "queries_ratio":        _ratio(wf["total_queries"],                tts["total_queries"]),
    }


# =============================================================================
# Section 2 — Visualizations
# =============================================================================

def plot_core_efficiency(tts_df: pd.DataFrame, wf_df: pd.DataFrame, out_dir: str):
    """4-panel IQR boxplot: time/gen, gens/run, time/run, time/successful run."""
    tts_df = tts_df.copy()
    wf_df  = wf_df.copy()
    for df in [tts_df, wf_df]:
        df["time_per_generation"] = df["elapsed_time_seconds"] / df["num_generations"]
        df["generations_used"]    = df.apply(
            lambda r: r["generation_found"] if r["success"] else r["num_generations"], axis=1
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    panels = [
        (axes[0, 0], "time_per_generation",    "Seconds",     "Time Per Generation"),
        (axes[0, 1], "generations_used",        "Generations", "Generations per Run"),
        (axes[1, 0], "elapsed_time_seconds",    "Seconds",     "Time per Run"),
    ]
    for ax, col, ylabel, title in panels:
        data = [tts_df[col].dropna(), wf_df[col].dropna()]
        bp = ax.boxplot(data, labels=["TTS", "Waveform"], patch_artist=True, widths=0.6)
        boxplot_colors(bp, [C_TTS, C_WAVEFORM])
        for el in ["whiskers", "fliers", "means", "medians", "caps"]:
            plt.setp(bp[el], color="black", linewidth=1.5)
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=12)
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    tts_succ = tts_df[tts_df["success"] == True]["elapsed_time_seconds"]
    wf_succ  = wf_df[wf_df["success"]  == True]["elapsed_time_seconds"]
    bp = ax.boxplot([tts_succ.dropna(), wf_succ.dropna()], labels=["TTS", "Waveform"],
                    patch_artist=True, widths=0.6)
    boxplot_colors(bp, [C_TTS, C_WAVEFORM])
    for el in ["whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(bp[el], color="black", linewidth=1.5)
    ax.set_ylabel("Seconds", fontweight="bold", fontsize=12)
    ax.set_title("Time per Successful Run", fontweight="bold", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Core Efficiency Metrics (IQR Distribution)", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    save_fig(os.path.join(out_dir, "core_efficiency_metrics.png"), dpi=DPI)


def _convergence_distribution(df: pd.DataFrame):
    successful = df[df["success"] == True]["generation_found"].values
    failed_count = int((df["success"] == False).sum())
    total = len(df)
    dist = {}
    for g in successful:
        dist[int(g)] = dist.get(int(g), 0) + 1
    dist[100] = dist.get(100, 0) + failed_count  # bucket ≥100 includes failures
    for k in dist:
        dist[k] = dist[k] / total * 100
    return dist, successful, failed_count, total


def plot_convergence_bar(tts_df: pd.DataFrame, wf_df: pd.DataFrame, out_dir: str):
    """Side-by-side bar charts of convergence distribution per generation."""
    tts_dist, tts_succ, tts_fail, tts_n = _convergence_distribution(tts_df)
    wf_dist,  wf_succ,  wf_fail,  wf_n  = _convergence_distribution(wf_df)

    all_gens = list(range(0, 101))
    tts_vals = [tts_dist.get(g, 0) for g in all_gens]
    wf_vals  = [wf_dist.get(g, 0)  for g in all_gens]
    y_max = max(max(tts_vals), max(wf_vals)) * 1.15

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    def _bar(ax, dist, vals, succ_vals, color, method, n_runs):
        x = np.arange(len(all_gens))
        ax.bar(x, vals, color=color, alpha=0.7, edgecolor="none", width=0.8, align="center")
        sr = len(succ_vals) / n_runs * 100
        ax.set_title(f"{method} (Success: {sr:.1f}%, n={n_runs})", fontweight="bold", fontsize=13)
        ax.set_xlabel("Generation", fontweight="bold", fontsize=12)
        ax.set_ylabel("Percentage of Runs (%)", fontweight="bold", fontsize=12)
        ax.set_xlim(0, 101)
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        if len(succ_vals) > 0:
            ax.axvline(np.mean(succ_vals),   color="#f59e0b", linestyle="--", linewidth=2,
                       label=f"Mean: {np.mean(succ_vals):.1f}")
            ax.axvline(np.median(succ_vals), color="#ef4444", linestyle="--", linewidth=2,
                       label=f"Median: {np.median(succ_vals):.0f}")
        ax.legend(fontsize=11)

    _bar(axes[0], tts_dist, tts_vals, tts_succ, C_TTS,      "TTS",      tts_n)
    _bar(axes[1], wf_dist,  wf_vals,  wf_succ,  C_WAVEFORM, "Waveform", wf_n)

    plt.suptitle("Convergence Distribution by Generation", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(os.path.join(out_dir, "convergence_bar.png"), dpi=DPI)


def plot_cumulative_success(tts_df: pd.DataFrame, wf_df: pd.DataFrame, out_dir: str):
    """Cumulative attack success rate over generations (line chart)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    gens = np.arange(1, 101)

    for df, color, label in [
        (tts_df, C_TTS, "TTS"),
        (wf_df, C_WAVEFORM, "Waveform"),
    ]:
        found = df[df["success"] == True]["generation_found"].dropna().astype(int)
        cumul = np.array([(found <= g).sum() / len(df) * 100 for g in gens])
        ax.plot(gens, cumul, color=color, linewidth=2.5, label=label,
                marker="o", markersize=3, alpha=0.7)
        ax.fill_between(gens, cumul, alpha=0.1, color=color)
        ax.axhline(cumul[-1], color=color, linestyle="--", linewidth=1.5, alpha=0.5)

    ax.set_xlabel("Generation", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Cumulative Attack Success Rate over Generations",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")

    save_fig(os.path.join(out_dir, "cumulative_success_comparison.png"), dpi=DPI)


# =============================================================================
# Section 3 — Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RQ3 Efficiency Analysis")
    parser.add_argument("--tts_csv",     default=TTS_CSV)
    parser.add_argument("--wf_csv",      default=WF_CSV)
    parser.add_argument("--output_dir",  default="outputs/thesis_analysis/RQ3/efficiency")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("RQ3: EFFICIENCY ANALYSIS — TTS vs Waveform")
    print(f"{'='*70}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    tts_df = pd.read_csv(args.tts_csv)
    wf_df  = pd.read_csv(args.wf_csv)
    print(f"    TTS:      {len(tts_df)} rows")
    print(f"    Waveform: {len(wf_df)} rows")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n[2] Computing efficiency metrics...")
    tts_metrics = compute_efficiency(tts_df, "TTS")
    wf_metrics  = compute_efficiency(wf_df,  "Waveform")
    comparison  = compute_comparison(tts_metrics, wf_metrics)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("EFFICIENCY SUMMARY")
    print(f"{'='*70}")
    for m in [tts_metrics, wf_metrics]:
        print(f"\n{m['method']}:")
        print(f"  Runs: {m['n_runs']}, Success: {m['success_rate']:.1f}%")
        print(f"  Avg time/run: {m['avg_time_per_run_sec']:.0f}s | "
              f"Avg gen/run: {m['avg_generations_overall']:.1f}")
        print(f"  Total queries: {m['total_queries']:,} | "
              f"Total time: {m['total_time_hours']:.1f} h")
    print(f"\nWaveform vs TTS ratios:")
    for k, v in comparison.items():
        if v is not None:
            print(f"  {k}: {v:.2f}×")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results = {"tts": tts_metrics, "waveform": wf_metrics, "comparison_ratios": comparison}
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, "rq3_efficiency_analysis.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {json_path}")

    # ── Visualizations ────────────────────────────────────────────────────────
    print("\n[3] Generating visualizations...")
    plot_convergence_bar(tts_df, wf_df, args.output_dir)
    plot_cumulative_success(tts_df, wf_df, args.output_dir)

    # ── Example run figures ───────────────────────────────────────────────────
    print("\n[4] Generating example run figures...")
    try:
        script_dir = Path(__file__).resolve().parent
        analyze_dir = str(script_dir.parent)
        if analyze_dir not in sys.path:
            sys.path.insert(0, analyze_dir)
        from analyze_results import plot_example_run
        for csv_path, out_name in [
            (args.tts_csv, "example_run_tts.png"),
            (args.wf_csv,  "example_run_waveform.png"),
        ]:
            method_df = pd.read_csv(csv_path)
            method_df.columns = method_df.columns.str.strip()
            method_df.rename(columns={"score_PESQ": "pesq", "score_SET_OVERLAP": "set_overlap"}, inplace=True)
            method_df["success"] = method_df["success"].astype(bool)
            results_dir = os.path.dirname(csv_path)
            method_df["json_path"] = method_df.apply(
                lambda r: os.path.join(
                    results_dir,
                    f"sentence_{int(r['sentence_id']):03d}",
                    f"run_{int(r['run_id'])}",
                    "run_summary.json",
                ), axis=1,
            )
            plot_example_run(method_df, args.output_dir)
            generated = os.path.join(args.output_dir, "example_run.png")
            target = os.path.join(args.output_dir, out_name)
            if os.path.exists(generated):
                shutil.copy2(generated, target)
                os.remove(generated)
                print(f"[Saved] {out_name}")
    except Exception as e:
        print(f"[!] Example run figures failed: {e}")

    print(f"\n{'='*70}")
    print(f"[✓] RQ3 analysis complete. Results saved to: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
