#!/usr/bin/env python3
"""
RQ1 Analysis — Adversarial TTS Attack Effectiveness

Input:
  outputs/results/TTS/all_results.csv
  outputs/results/Waveform/all_results.csv

Output:
  outputs/thesis_analysis/RQ1/
  ├── data/
  │   ├── combined_results.csv
  │   └── rq1_summary.json
  ├── rq1_all_metrics_whitney_u_test_kde.png
  ├── rq1_comparison_venn.png
  ├── rq1_comparison_per_sentence.png
  ├── rq1_upset_combined_pct.png       (via rq1/rq1_upset_combined.py)
  ├── rq1_comparison_tables.png        (via rq1/rq1_example_runs_table.py)
  ├── venn_PESQ-SetOverlap_TTS_Waveform.png
  ├── venn_SBERT-SetOverlap_TTS_Waveform.png
  └── venn_UTMOS-PESQ_TTS_Waveform.png

Run:
  python Scripts/Analysis/rq1_analysis.py
"""

import os
import sys
import json
import shutil
import warnings
import argparse
import subprocess
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
from pathlib import Path

from plot_utils import (
    C_TTS, C_WAVEFORM, DPI,
    THR_PESQ, THR_SO, THR_SBERT, THR_UTMOS,
    iqr_stats, format_p,
    method_colors, method_subplots, save_fig,
    threshold_shading, kde_by_group,
    annotate_venn2, venn2_by_method,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

TTS_CSV = "outputs/results/TTS/all_results.csv"
WF_CSV  = "outputs/results/Waveform/all_results.csv"


# ============================================================================
# SECTION 1: Data Preparation
# ============================================================================

def load_and_merge(tts_csv: str, wf_csv: str) -> pd.DataFrame:
    """Load TTS and Waveform CSVs, normalize columns, add method label."""
    frames = []
    for csv_path, method in [(tts_csv, "TTS"), (wf_csv, "Waveform")]:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        rename = {
            "score_SET_OVERLAP": "set_overlap",
            "score_PESQ":        "pesq",
            "SET_OVERLAP":       "set_overlap",
            "PESQ":              "pesq",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df["method"] = method
        if "success" in df.columns:
            df["success"] = df["success"].astype(bool)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No result CSVs found.")

    combined = pd.concat(frames, ignore_index=True)
    if "success" in combined.columns:
        combined["success"] = combined["success"].astype(bool)
    print(f"[*] Loaded {len(combined)} runs: "
          f"TTS={len(combined[combined['method']=='TTS'])}, "
          f"Waveform={len(combined[combined['method']=='Waveform'])}")
    return combined


# ============================================================================
# SECTION 2: Statistics
# ============================================================================

def cohens_d(g1: pd.Series, g2: pd.Series) -> float:
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1 - 1) * g1.var() + (n2 - 1) * g2.var()) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled if pooled > 0 else 0.0


def build_summary(df: pd.DataFrame) -> dict:
    n = len(df)
    success_rates = {
        "combined_success_rate": round(100 * ((df["set_overlap"] <= THR_SO) &
                                               (df["pesq"] <= THR_PESQ)).sum() / n, 2),
        "set_overlap_met_rate":  round(100 * (df["set_overlap"] <= THR_SO).sum() / n, 2),
        "pesq_met_rate":         round(100 * (df["pesq"] <= THR_PESQ).sum() / n, 2),
    }
    if "semantic_similarity" in df.columns and df["semantic_similarity"].notna().any():
        success_rates["sbert_met_rate"] = round(
            100 * (df["semantic_similarity"] < THR_SBERT).sum() / n, 2)

    methods_stats = {}
    for method in sorted(df["method"].unique()):
        sub  = df[df["method"] == method]
        m_n  = len(sub)
        entry = {
            "n_runs":               m_n,
            "n_sentences":          int(sub["sentence_id"].nunique()),
            "combined_success_rate": round(100 * ((sub["set_overlap"] <= THR_SO) &
                                                   (sub["pesq"] <= THR_PESQ)).sum() / m_n, 2),
            "pesq_met_rate":        round(100 * (sub["pesq"] <= THR_PESQ).sum() / m_n, 2),
            "set_overlap_met_rate": round(100 * (sub["set_overlap"] <= THR_SO).sum() / m_n, 2),
            "pesq_stats":           iqr_stats(sub["pesq"].dropna()),
            "set_overlap_stats":    iqr_stats(sub["set_overlap"].dropna()),
        }
        if "semantic_similarity" in sub.columns:
            entry["sbert_met_rate"] = round(
                100 * (sub["semantic_similarity"] < THR_SBERT).sum() / m_n, 2)
        methods_stats[method] = entry

    method_comparison = {}
    if df["method"].nunique() == 2:
        methods = sorted(df["method"].unique())
        for col in ["set_overlap", "pesq", "semantic_similarity"]:
            if col not in df.columns:
                continue
            g1 = df[df["method"] == methods[0]][col].dropna()
            g2 = df[df["method"] == methods[1]][col].dropna()
            if len(g1) > 0 and len(g2) > 0:
                u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                method_comparison[col] = {
                    "method_1":   methods[0], "method_2": methods[1],
                    "median_1":   round(float(g1.median()), 4),
                    "median_2":   round(float(g2.median()), 4),
                    "u_statistic": round(float(u_stat), 2),
                    "p_value":    round(float(u_p), 6),
                    "cohens_d":   round(cohens_d(g1, g2), 4),
                    "significant": bool(float(u_p) < 0.05),
                }

    return {
        "n_runs":            n,
        "n_sentences":       int(df["sentence_id"].nunique()),
        "success_rates":     success_rates,
        "methods_stats":     methods_stats,
        "method_comparison": method_comparison,
    }


def print_summary(summary: dict):
    sr = summary["success_rates"]
    print(f"\n{'='*60}")
    print(f"RQ1 SUMMARY")
    print(f"{'='*60}")
    print(f"Total runs: {summary['n_runs']}  |  Sentences: {summary['n_sentences']}")
    print(f"Overall combined success (PESQ≤{THR_PESQ} & SO≤{THR_SO}): "
          f"{sr['combined_success_rate']}%")
    for method, s in summary["methods_stats"].items():
        print(f"\n{method} (n={s['n_runs']}, {s['n_sentences']} sentences):")
        print(f"  Combined success: {s['combined_success_rate']}%")
        print(f"  PESQ met: {s['pesq_met_rate']}%  |  "
              f"SET_OVERLAP met: {s['set_overlap_met_rate']}%")
        print(f"  PESQ median={s['pesq_stats']['median']}  |  "
              f"SO median={s['set_overlap_stats']['median']}")
    for col, mc in summary.get("method_comparison", {}).items():
        print(f"\nMann-Whitney {col}: U={mc['u_statistic']}, "
              f"p={mc['p_value']:.2e}, d={mc['cohens_d']:.3f}")


# ============================================================================
# SECTION 3: Visualizations
# ============================================================================

def plot_kde_distributions(df: pd.DataFrame, out: str):
    """KDE distributions with Mann-Whitney U stats for each metric."""
    metrics = [("pesq", "PESQ (Imperceptibility, target ≤0.2)", THR_PESQ, "left")]
    if "utmos_best" in df.columns and df["utmos_best"].notna().any():
        metrics.append(("utmos_best", "UTMOS (Naturalness, target ≥3.5)", THR_UTMOS, "right"))
    metrics.append(("set_overlap", "SetOverlap (Distortion, target ≤0.5)", THR_SO, "left"))
    if "semantic_similarity" in df.columns and df["semantic_similarity"].notna().any():
        metrics.append(("semantic_similarity", "SBERT (Semantic Sim., target ≤0.6)",
                        THR_SBERT, "left"))

    ncols = 2
    nrows = (len(metrics) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes_flat = axes.flatten() if nrows > 1 else list(axes)
    colors_m = method_colors(df)
    methods = sorted(df["method"].unique())

    for idx, (metric, title, threshold, side) in enumerate(metrics):
        ax = axes_flat[idx]
        kde_by_group(ax, df, metric, "method", colors_m)
        ax.axvline(threshold, color="red", linestyle="--", linewidth=2)
        threshold_shading(ax, threshold, side)

        # ── Stats box (top-right): Mann-Whitney U + Cohen's d ─────────────────
        if len(methods) == 2:
            g1 = df[df["method"] == methods[0]][metric].dropna().values
            g2 = df[df["method"] == methods[1]][metric].dropna().values
            if len(g1) > 0 and len(g2) > 0:
                u, p = mannwhitneyu(g1, g2, alternative="two-sided")
                p_str = (f"p < 0.001***" if p < 0.001 else
                         f"p = {p:.4f}**" if p < 0.01 else
                         f"p = {p:.4f}*"  if p < 0.05 else
                         f"p = {p:.4f}ns")
                d = cohens_d(pd.Series(g1), pd.Series(g2))
                box = f"Mann-Whitney U\nU = {u:.0f}\n{p_str}\nd = {d:.2f}"
                ax.text(0.98, 0.97, box, transform=ax.transAxes, fontsize=9,
                        va="top", ha="right",
                        bbox=dict(boxstyle="round", facecolor="lightyellow",
                                  alpha=0.8, edgecolor="gray"),
                        fontfamily="monospace", fontweight="bold")

        # ── Legend (top-left): methods + threshold + success/failure zones ────
        legend_handles = []
        for method in methods:
            legend_handles.append(
                mpatches.Patch(color=colors_m.get(method, C_TTS), alpha=0.6, label=method))
        legend_handles.append(
            plt.Line2D([0], [0], color="red", linestyle="--", linewidth=2,
                       label=f"Threshold ({threshold})"))
        legend_handles.append(
            mpatches.Patch(facecolor="green", alpha=0.3, label="Success Zone"))
        legend_handles.append(
            mpatches.Patch(facecolor="none", hatch="///", edgecolor="red",
                           alpha=0.8, label="Failure Zone"))
        ax.legend(handles=legend_handles, fontsize=8, loc="upper left", frameon=True)

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel(metric.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.3)

    for i in range(len(metrics), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.suptitle("Optimization Metrics Distribution: TTS vs Waveform (KDE)",
                 fontsize=13, fontweight="bold")
    save_fig(out)



def plot_venn_comparison(df: pd.DataFrame, out: str):
    """Venn: Distortion vs Imperceptibility, side-by-side TTS | Waveform."""
    has_sbert = "semantic_similarity" in df.columns and df["semantic_similarity"].notna().any()
    has_utmos = "utmos_best" in df.columns and df["utmos_best"].notna().any()

    def set_a(sub):  # distortion
        so  = set(sub.index[sub["set_overlap"] <= THR_SO])
        sb  = set(sub.index[sub["semantic_similarity"] < THR_SBERT]) if has_sbert else set(sub.index)
        return so & sb

    def set_b(sub):  # imperceptibility
        pesq = set(sub.index[sub["pesq"] <= THR_PESQ])
        utm  = set(sub.index[sub["utmos_best"] >= THR_UTMOS]) if has_utmos else set(sub.index)
        return pesq & utm

    label_a = (f"Distortion\n(SET_OVL ≤ {THR_SO}"
               + (f"\n& Sem.Sim. < {THR_SBERT}" if has_sbert else "") + ")")
    label_b = (f"Imperceptibility\n(PESQ ≤ {THR_PESQ}"
               + (f"\n& UTMOS ≥ {THR_UTMOS}" if has_utmos else "") + ")")

    venn2_by_method(df, set_a, set_b, label_a, label_b,
                    "Condition Overlap: Distortion vs Imperceptibility", out)


def plot_venn_pesq_setoverlap(df: pd.DataFrame, out: str):
    """Venn: PESQ vs SetOverlap thresholds, side-by-side TTS | Waveform."""
    venn2_by_method(
        df,
        lambda sub: set(sub.index[sub["pesq"] <= THR_PESQ]),
        lambda sub: set(sub.index[sub["set_overlap"] <= THR_SO]),
        f"Imperceptibility\n(PESQ ≤ {THR_PESQ})",
        f"Distortion\n(SetOverlap ≤ {THR_SO})",
        "SetOverlap vs PESQ Thresholds: TTS vs Waveform",
        out,
    )


def plot_venn_sbert_setoverlap(df: pd.DataFrame, out: str):
    """Venn: SBERT vs SetOverlap thresholds, side-by-side TTS | Waveform."""
    if "semantic_similarity" not in df.columns or not df["semantic_similarity"].notna().any():
        print("[Skip] SBERT-SetOverlap Venn: no SBERT data")
        return
    venn2_by_method(
        df,
        lambda sub: set(sub.index[sub["semantic_similarity"] < THR_SBERT]),
        lambda sub: set(sub.index[sub["set_overlap"] <= THR_SO]),
        f"Semantic Divergence\n(SBERT < {THR_SBERT})",
        f"Distortion\n(SetOverlap ≤ {THR_SO})",
        "SBERT vs SetOverlap Thresholds: TTS vs Waveform",
        out,
    )


def plot_venn_utmos_pesq(df: pd.DataFrame, out: str):
    """Venn: UTMOS vs PESQ thresholds, side-by-side TTS | Waveform."""
    if "utmos_best" not in df.columns or not df["utmos_best"].notna().any():
        print("[Skip] UTMOS-PESQ Venn: no UTMOS data")
        return
    venn2_by_method(
        df,
        lambda sub: set(sub.index[sub["utmos_best"] >= THR_UTMOS]),
        lambda sub: set(sub.index[sub["pesq"] <= THR_PESQ]),
        f"Naturalness\n(UTMOS ≥ {THR_UTMOS})",
        f"Imperceptibility\n(PESQ ≤ {THR_PESQ})",
        "UTMOS vs PESQ Thresholds: TTS vs Waveform",
        out,
    )


def plot_per_sentence_outcomes(df: pd.DataFrame, out: str):
    """Pie charts: per-sentence success rate (out of 3 runs), side-by-side."""
    fig, axes_flat, methods, _ = method_subplots(df)
    colors_outcome = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]

    for idx, method in enumerate(methods):
        ax = axes_flat[idx]
        sub = df[df["method"] == method]
        buckets = {0: 0, 1: 0, 2: 0, 3: 0}
        for sid in sorted(sub["sentence_id"].unique()):
            n_success = int(sub.loc[sub["sentence_id"] == sid, "success"].sum())
            buckets[min(n_success, 3)] += 1

        total = sum(buckets.values())
        labels, sizes, pie_colors = [], [], []
        for k in [3, 2, 1, 0]:
            v = buckets[k]
            if v > 0:
                labels.append(f"{k}/3 runs\n({v} sent)")
                sizes.append(v)
                pie_colors.append(colors_outcome[k])

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%",
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight("bold")
        ax.set_title(f"{method} ({total} sentences)", fontsize=12, fontweight="bold")

    plt.suptitle("Per-Sentence Outcome Distribution", fontsize=13, fontweight="bold")
    save_fig(out)


def _scatter_side_by_side(df, x_col, y_col, xlabel, ylabel, title,
                          threshold_x=None, threshold_y=None,
                          diagonal=False, out=None):
    """Generic side-by-side scatter (TTS | Waveform) coloured by success/failure."""
    C_S = "#1f77b4"   # blue  — success  (matches original)
    C_F = "#ff7f0e"   # orange — failure  (matches original)
    methods = sorted(df["method"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6))
    if len(methods) == 1:
        axes = [axes]

    for idx, method in enumerate(methods):
        ax = axes[idx]
        sub = df[df["method"] == method].dropna(subset=[x_col, y_col])
        n = len(sub)
        for success, color, label in [(True, C_S, "Success"), (False, C_F, "Failure")]:
            mask = sub["success"] == success
            n_g = int(mask.sum())
            if n_g:
                ax.scatter(sub.loc[mask, x_col], sub.loc[mask, y_col],
                           color=color, label=f"{label} (n={n_g})",
                           alpha=0.7, s=30, edgecolors="none")
        if threshold_x is not None:
            ax.axvline(threshold_x, color="red", linestyle="-", linewidth=1.5, alpha=0.8)
        if threshold_y is not None:
            ax.axhline(threshold_y, color="red", linestyle="-", linewidth=1.5, alpha=0.8)
        if diagonal:
            all_vals = pd.concat([sub[x_col], sub[y_col]])
            lo, hi = all_vals.min() - 0.1, all_vals.max() + 0.1
            ax.plot([lo, hi], [lo, hi], color="green", linewidth=1.5,
                    label="Perfect preservation (y=x)")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{method} (n={n})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, frameon=True)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    save_fig(out)


def plot_scatter_pesq_setoverlap(df: pd.DataFrame, out: str):
    _scatter_side_by_side(df, "pesq", "set_overlap",
                          "PESQ", "SetOverlap", "PESQ vs SetOverlap",
                          threshold_x=THR_PESQ, threshold_y=THR_SO, out=out)


def plot_scatter_setoverlap_sbert(df: pd.DataFrame, out: str):
    if "semantic_similarity" not in df.columns or not df["semantic_similarity"].notna().any():
        print("[Skip] scatter SetOverlap-SBERT: no SBERT data")
        return
    _scatter_side_by_side(df, "set_overlap", "semantic_similarity",
                          "SetOverlap", "SBERT", "SetOverlap vs SBERT",
                          threshold_x=THR_SO, out=out)


def plot_scatter_utmos_gt(df: pd.DataFrame, out: str):
    if "utmos_best" not in df.columns or not df["utmos_best"].notna().any():
        print("[Skip] scatter UTMOS_GT-UTMOS: no UTMOS data")
        return
    if "utmos_gt" not in df.columns or not df["utmos_gt"].notna().any():
        print("[Skip] scatter UTMOS_GT-UTMOS: no utmos_gt data")
        return
    _scatter_side_by_side(df, "utmos_gt", "utmos_best",
                          "UTMOS_GT", "UTMOS", "UTMOS_GT vs UTMOS",
                          diagonal=True, out=out)


def plot_scatter_utmos_pesq(df: pd.DataFrame, out: str):
    if "utmos_best" not in df.columns or not df["utmos_best"].notna().any():
        print("[Skip] scatter UTMOS-PESQ: no UTMOS data")
        return
    _scatter_side_by_side(df, "utmos_best", "pesq",
                          "UTMOS", "PESQ", "UTMOS vs PESQ",
                          threshold_x=THR_UTMOS, threshold_y=THR_PESQ, out=out)



# ============================================================================
# SECTION 4: Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RQ1 Attack Effectiveness Analysis")
    parser.add_argument("--tts_csv",    default=TTS_CSV)
    parser.add_argument("--wf_csv",     default=WF_CSV)
    parser.add_argument("--output_dir", default="outputs/thesis_analysis/RQ1")
    args = parser.parse_args()

    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    if os.path.exists(project_root / "outputs"):
        os.chdir(project_root)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"RQ1 ANALYSIS: Adversarial TTS Attack Effectiveness")
    print(f"{'='*60}")
    print(f"Output: {args.output_dir}")

    # --- Data ---
    print("\n[1/4] Loading and merging data...")
    df = load_and_merge(args.tts_csv, args.wf_csv)

    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    csv_out = os.path.join(data_dir, "combined_results.csv")
    df.to_csv(csv_out, index=False)
    print(f"[Saved] data/combined_results.csv ({len(df)} rows)")

    # --- Statistics ---
    print("\n[2/4] Computing statistics...")
    summary = build_summary(df)
    json_out = os.path.join(data_dir, "rq1_summary.json")
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] data/rq1_summary.json")
    print_summary(summary)

    # --- Visualizations ---
    print("\n[3/4] Generating visualizations...")
    od = args.output_dir

    plot_kde_distributions(df,      os.path.join(od, "rq1_all_metrics_whitney_u_test_kde.png"))
    plot_venn_comparison(df,        os.path.join(od, "rq1_comparison_venn.png"))
    plot_per_sentence_outcomes(df,  os.path.join(od, "rq1_comparison_per_sentence.png"))
    plot_venn_pesq_setoverlap(df,   os.path.join(od, "venn_PESQ-SetOverlap_TTS_Waveform.png"))
    plot_venn_sbert_setoverlap(df,  os.path.join(od, "venn_SBERT-SetOverlap_TTS_Waveform.png"))
    plot_venn_utmos_pesq(df,        os.path.join(od, "venn_UTMOS-PESQ_TTS_Waveform.png"))
    plot_scatter_pesq_setoverlap(df,  os.path.join(od, "scatter_PESQ-SetOverlap_TTS_Waveform.png"))
    plot_scatter_setoverlap_sbert(df, os.path.join(od, "scatter_SetOverlap-SBERT_TTS_Waveform.png"))
    plot_scatter_utmos_gt(df,         os.path.join(od, "scatter_UTMOS_GT-UTMOS_TTS_Waveform.png"))
    plot_scatter_utmos_pesq(df,       os.path.join(od, "scatter_UTMOS-PESQ_TTS_Waveform.png"))

    # --- UpSet plot (rq1_upset_combined.py) ---
    print("\n[3d/4] Generating UpSet plot...")
    upset_src = str(script_dir / "rq1" / "rq1_upset_combined.py")
    if os.path.exists(upset_src):
        rc = subprocess.run([sys.executable, upset_src]).returncode
        upset_generated = os.path.join(od, "02_success_analysis", "rq1_upset_combined_pct.png")
        upset_target    = os.path.join(od, "rq1_upset_combined_pct.png")
        if os.path.exists(upset_generated):
            shutil.copy2(upset_generated, upset_target)
            print(f"[Saved] rq1_upset_combined_pct.png")
            shutil.rmtree(os.path.join(od, "02_success_analysis"), ignore_errors=True)
            shutil.rmtree(os.path.join(od, "data"), ignore_errors=True)
        elif rc != 0:
            print(f"[!] rq1_upset_combined.py failed (exit {rc})")

    # --- Comparison tables (rq1_example_runs_table.py) ---
    print("\n[3e/4] Generating comparison tables...")
    tables_src = str(script_dir / "rq1" / "rq1_example_runs_table.py")
    if os.path.exists(tables_src):
        rc = subprocess.run([sys.executable, tables_src]).returncode
        if rc != 0:
            print(f"[!] rq1_example_runs_table.py failed (exit {rc})")

    print(f"\n[4/4] Done. All outputs in: {od}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
