"""
RQ2 Analysis — Human Validation

Combined script merging all RQ2 analyses:
  - CER computation (alphabet-normalized Levenshtein)
  - Delta CER and quality threshold analysis
  - MOS ratings by condition and PESQ threshold
  - Correlation analysis: PESQ / UTMOS / MOS (Pearson + Spearman)
  - Spearman scatter: CER vs PESQ/UTMOS, MOS vs UTMOS/PESQ/CER (TTS only)
  - SBERT semantic recovery boxplots
  - 3-way Venn: PESQ / MOS / UTMOS thresholds
  - rq2_summary.json with all statistics

Usage:
    python Scripts/Analysis/rq2_analysis.py
    python Scripts/Analysis/rq2_analysis.py \\
        --survey_csv outputs/human_survey.csv \\
        --output_dir outputs/thesis_analysis/RQ2
"""

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr, pearsonr

from plot_utils import (
    C_GT, C_TTS, C_WAVEFORM, DPI,
    THR_PESQ, THR_SO, THR_UTMOS, THR_MOS,
    iqr_stats, format_p,
    save_fig, boxplot_colors, outlier_scatter,
    scatter_outcome, kde_by_group,
    mann_whitney_box, spearman_box,
    venn3_simple,
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

# Success/failure colors for RQ2 (distinct from RQ1 to match prior thesis figures)
C_SUCCESS = "#0173B2"
C_FAILURE = "#DE8F05"


# =============================================================================
# Section 1 — Data Loading & CER Computation
# =============================================================================

def normalize_text(text: str) -> str:
    """Lowercase + alphabet-only (removes punctuation, fillers, whitespace)."""
    return "".join(c for c in str(text).lower() if c.isalpha())


def levenshtein_distance(s1, s2) -> int:
    if isinstance(s1, float) or isinstance(s2, float):
        return np.nan
    s1 = normalize_text(s1)
    s2 = normalize_text(s2)
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(s1)][len(s2)]


def compute_cer(gt_text, participant_text) -> float:
    """CER = levenshtein_distance(gt, participant) / len(normalize(gt))."""
    if isinstance(gt_text, float) or isinstance(participant_text, float):
        return np.nan
    gt_norm = normalize_text(gt_text)
    if len(gt_norm) == 0:
        return np.nan
    dist = levenshtein_distance(gt_text, participant_text)
    if isinstance(dist, float) and np.isnan(dist):
        return np.nan
    return dist / len(gt_norm)


def load_survey(csv_path: str) -> pd.DataFrame:
    """Load survey CSV and normalise column names."""
    df = pd.read_csv(csv_path)
    column_mapping = {
        "Unique ID":            "participant_id",
        "Category":             "condition",
        "Sentence_ID":          "sentence_id",
        "Bucket":               "outcome",
        "MOS Score":            "mos_rating",
        "PESQ":                 "pesq",
        "Set Overlap":          "set_overlap",
        "Ground Truth":         "ground_truth",
        "Transcribed Sentence": "asr_transcription",
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    for col in ["sentence_id", "mos_rating", "pesq", "set_overlap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "condition" in df.columns:
        df["condition"] = df["condition"].str.upper()
    return df


def ensure_cer_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add CER and delta_cer columns if not already present."""
    df = df.copy()

    if "CER" not in df.columns:
        print("  [CER] Computing alphabet-normalised CER...")
        df["CER"] = df.apply(
            lambda r: compute_cer(r.get("ground_truth"), r.get("asr_transcription")),
            axis=1,
        )

    if "delta_cer" not in df.columns:
        print("  [CER] Computing delta_cer vs GT baseline per sentence...")
        gt_baseline = (
            df[df["condition"] == "GT"][["sentence_id", "CER"]]
            .dropna()
            .groupby("sentence_id")["CER"]
            .mean()
        )
        df["delta_cer"] = np.nan
        adv_mask = df["condition"].isin(["TTS", "WAVEFORM"])
        for idx, row in df[adv_mask].iterrows():
            sid = row["sentence_id"]
            if pd.notna(row["CER"]) and sid in gt_baseline.index:
                df.at[idx, "delta_cer"] = row["CER"] - gt_baseline[sid]

    return df


def compute_quality_threshold(df: pd.DataFrame) -> dict:
    """Compute Mean(GT CER) + 1σ quality threshold."""
    gt_cer   = df[df["condition"] == "GT"]["CER"].dropna()
    mean_gt  = float(gt_cer.mean())
    std_gt   = float(gt_cer.std())
    threshold = mean_gt + std_gt

    results = {
        "quality_threshold": threshold,
        "gt_stats": {"mean": mean_gt, "std": std_gt, "n": int(len(gt_cer))},
        "overall": {},
        "by_pesq_threshold": {},
    }

    for cond in ["TTS", "WAVEFORM"]:
        cond_data   = df[df["condition"] == cond]
        valid_delta = cond_data["delta_cer"].dropna()
        n    = len(valid_delta)
        good = int((valid_delta <= threshold).sum())
        results["overall"][cond] = {
            "total_responses":    n,
            "good_quality_count": good,
            "good_quality_pct":   round(100 * good / n, 2) if n > 0 else 0.0,
            "mean_delta_cer":     round(float(valid_delta.mean()), 4) if n > 0 else None,
            "std_delta_cer":      round(float(valid_delta.std()),  4) if n > 0 else None,
        }
        results["by_pesq_threshold"][cond] = {}
        for met, label in [(True, "pesq_met"), (False, "pesq_not_met")]:
            subset = (cond_data[cond_data["pesq"] <= THR_PESQ] if met
                      else cond_data[cond_data["pesq"] > THR_PESQ])
            sd = subset["delta_cer"].dropna()
            sn = len(sd)
            if sn > 0:
                sg = int((sd <= threshold).sum())
                results["by_pesq_threshold"][cond][label] = {
                    "count":              sn,
                    "good_quality_count": sg,
                    "good_quality_pct":   round(100 * sg / sn, 2),
                    "mean_delta_cer":     round(float(sd.mean()), 4),
                    "std_delta_cer":      round(float(sd.std()),  4),
                }
    return results


# =============================================================================
# Section 2 — Statistics
# =============================================================================

def _is_success(pesq, so) -> bool:
    if pd.isna(pesq) or pd.isna(so):
        return False
    return pesq <= THR_PESQ and so <= THR_SO


def compute_mos_stats(df: pd.DataFrame) -> dict:
    out = {}
    for cond in ["GT", "TTS", "WAVEFORM"]:
        subset = df[df["condition"] == cond]["mos_rating"]
        if len(subset.dropna()) > 0:
            out[f"mos_{cond}"] = iqr_stats(subset)
    for cond in ["TTS", "WAVEFORM"]:
        sub = df[df["condition"] == cond]
        for met, label in [(True, "met"), (False, "not_met")]:
            mask = sub["pesq"] <= THR_PESQ if met else sub["pesq"] > THR_PESQ
            s = sub[mask]["mos_rating"].dropna()
            if len(s) > 0:
                out.setdefault(f"mos_by_pesq_{cond}", {})[label] = iqr_stats(s)
    return out


def compute_correlation_stats(df: pd.DataFrame) -> dict:
    """Pearson and Spearman correlations for PESQ / UTMOS / MOS."""
    out = {}
    pesq_col  = "pesq"       if "pesq"      in df.columns else "PESQ"
    utmos_col = "UTMOS"      if "UTMOS"     in df.columns else "utmos"
    mos_col   = "mos_rating" if "mos_rating" in df.columns else "MOS"

    pairs = []
    if pesq_col  in df.columns and mos_col   in df.columns:
        pairs.append((pesq_col,  mos_col,   "PESQ_MOS"))
    if utmos_col in df.columns and mos_col   in df.columns:
        pairs.append((utmos_col, mos_col,   "UTMOS_MOS"))
    if pesq_col  in df.columns and utmos_col in df.columns:
        pairs.append((pesq_col,  utmos_col, "PESQ_UTMOS"))
    if "set_overlap" in df.columns and mos_col in df.columns:
        pairs.append(("set_overlap", mos_col, "SETOVERLAP_MOS"))

    for x_col, y_col, key in pairs:
        valid = df[[x_col, y_col]].dropna()
        if len(valid) > 2:
            r_p, p_p = pearsonr(valid[x_col],  valid[y_col])
            r_s, p_s = spearmanr(valid[x_col], valid[y_col])
            out[f"overall_{key}"] = {
                "pearson_r":  round(float(r_p), 4), "pearson_p":  round(float(p_p), 4),
                "spearman_r": round(float(r_s), 4), "spearman_p": round(float(p_s), 4),
                "n": len(valid),
            }
        for cond in ["GT", "TTS", "WAVEFORM"]:
            subset = df[df["condition"] == cond][[x_col, y_col]].dropna()
            if len(subset) > 2:
                r_p, p_p = pearsonr(subset[x_col],  subset[y_col])
                r_s, p_s = spearmanr(subset[x_col], subset[y_col])
                out[f"{cond}_{key}"] = {
                    "pearson_r":  round(float(r_p), 4), "pearson_p":  round(float(p_p), 4),
                    "spearman_r": round(float(r_s), 4), "spearman_p": round(float(p_s), 4),
                    "n": len(subset),
                }
    return out


def build_summary(df: pd.DataFrame, quality_data: dict, corr_data: dict) -> dict:
    summary = {
        "mos":         compute_mos_stats(df),
        "correlation": corr_data,
        "cer":         quality_data,
    }

    cer_stats = {}
    for cond in ["GT", "TTS", "WAVEFORM"]:
        s = df[df["condition"] == cond]["CER"].dropna()
        if len(s) > 0:
            cer_stats[cond] = iqr_stats(s)
    summary["cer_by_condition"] = cer_stats

    sbert_cols = ["sbert_gt_guess_normalized", "sbert_gt_asr_normalized"]
    if all(c in df.columns for c in sbert_cols):
        sbert_gain = {}
        for cond in ["GT", "TTS", "WAVEFORM"]:
            aligned = df[df["condition"] == cond][sbert_cols].dropna()
            if len(aligned) > 0:
                gain = aligned["sbert_gt_guess_normalized"] - aligned["sbert_gt_asr_normalized"]
                sbert_gain[cond] = iqr_stats(gain)
        summary["sbert_recovery_gain"] = sbert_gain

    return summary


def print_summary(summary: dict):
    print("\n" + "=" * 70)
    print("RQ2: HUMAN VALIDATION SUMMARY")
    print("=" * 70)

    mos = summary.get("mos", {})
    if mos:
        print("\nMOS Ratings:")
        for cond in ["GT", "TTS", "WAVEFORM"]:
            s = mos.get(f"mos_{cond}")
            if s:
                print(f"  {cond:10s}: mean={s['mean']:.2f}±{s['std']:.2f}, "
                      f"median={s['median']:.2f}, n={s['n']}")

    cer = summary.get("cer_by_condition", {})
    if cer:
        print("\nCER by condition:")
        for cond in ["GT", "TTS", "WAVEFORM"]:
            s = cer.get(cond)
            if s:
                print(f"  {cond:10s}: mean={s['mean']:.4f}±{s['std']:.4f}, n={s['n']}")

    qt  = summary.get("cer", {})
    thr = qt.get("quality_threshold")
    if thr:
        print(f"\nQuality Threshold: {thr:.4f} (GT mean+1σ)")
        for cond in ["TTS", "WAVEFORM"]:
            o = qt.get("overall", {}).get(cond)
            if o:
                print(f"  {cond:10s}: {o['good_quality_count']}/{o['total_responses']} "
                      f"({o['good_quality_pct']:.1f}%) good quality")

    corr = summary.get("correlation", {})
    for key in ["overall_PESQ_MOS", "overall_UTMOS_MOS", "overall_SETOVERLAP_MOS"]:
        s = corr.get(key)
        if s:
            label = key.replace("overall_", "").replace("_", " ↔ ")
            print(f"\n  {label}: r={s['pearson_r']:.4f} (Pearson), "
                  f"ρ={s['spearman_r']:.4f} (Spearman), n={s['n']}")


# =============================================================================
# Section 3 — Visualizations
# =============================================================================

def plot_mos_boxplots(df: pd.DataFrame, out_dir: str):
    """IQR boxplots: MOS by condition (left) and by PESQ threshold (right)."""
    if "mos_rating" not in df.columns:
        return

    conditions = ["GT", "TTS", "WAVEFORM"]
    labels     = ["GT", "TTS", "Waveform"]
    colors_bp  = [C_GT, C_TTS, C_WAVEFORM]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # — left: MOS by condition —
    ax   = axes[0]
    data = [df[df["condition"] == c]["mos_rating"].dropna().values for c in conditions]
    bp   = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6, showfliers=False)
    boxplot_colors(bp, colors_bp)
    outlier_scatter(ax, data, colors_bp)
    for i, (cond, _) in enumerate(zip(conditions, labels)):
        n = len(df[df["condition"] == cond]["mos_rating"].dropna())
        ax.text(i + 1, 5.2, f"n={n}", ha="center", fontsize=9)
    ax.axhline(y=THR_MOS, color="red", linestyle="--", alpha=0.5, linewidth=2,
               label="MOS threshold (3.5)")
    ax.set_ylabel("Mean Opinion Score (MOS)", fontsize=11, fontweight="bold")
    ax.set_title("MOS Ratings by Condition", fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 5.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    # — right: MOS by PESQ threshold —
    ax = axes[1]
    if "pesq" in df.columns:
        all_data, tick_labels = [], []
        adv_conds  = ["TTS", "WAVEFORM"]
        adv_labels = ["TTS", "Waveform"]
        adv_colors = [C_TTS, C_WAVEFORM]
        for met, thr_label in [(True, "PESQ ≤ 0.2\n(Met)"), (False, "PESQ > 0.2\n(Not Met)")]:
            for cond, lbl in zip(adv_conds, adv_labels):
                sub  = df[(df["condition"] == cond) &
                          ((df["pesq"] <= THR_PESQ) if met else (df["pesq"] > THR_PESQ))]
                vals = sub["mos_rating"].dropna().values
                if len(vals):
                    all_data.append(vals)
                    tick_labels.append(f"{thr_label}\n{lbl}")
        bp = ax.boxplot(all_data, labels=tick_labels, patch_artist=True,
                        widths=0.6, showfliers=False)
        tile_colors = adv_colors * 2
        boxplot_colors(bp, tile_colors)
        outlier_scatter(ax, all_data, tile_colors)
        ax.axhline(y=THR_MOS, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_ylabel("MOS", fontsize=11, fontweight="bold")
        ax.set_title("MOS by PESQ Threshold", fontsize=12, fontweight="bold")
        ax.set_ylim(0.5, 5.5)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=9)

    save_fig(os.path.join(out_dir, "rq2_mos_boxplot.png"))


def plot_cer_boxplot_mann_whitney(df: pd.DataFrame, out_dir: str,
                                  quality_threshold: float = None):
    """CER boxplot (GT / TTS / Waveform) with Mann-Whitney U annotations."""
    cer_col = "CER" if "CER" in df.columns else None
    if cer_col is None:
        print("[!] CER column missing, skipping CER boxplot")
        return

    listen_df  = df[df["Task"] == "listen+rate"].copy() if "Task" in df.columns else df.copy()
    conditions = ["GT", "TTS", "WAVEFORM"]
    labels     = ["GT", "TTS", "Waveform"]
    colors_bp  = [C_GT, C_TTS, C_WAVEFORM]

    data_dict = {c: listen_df[listen_df["condition"] == c][cer_col].dropna().values
                 for c in conditions}
    data_list = [data_dict[c] for c in conditions]

    fig, ax = plt.subplots(figsize=(10, 7))
    bp = ax.boxplot(data_list, tick_labels=labels, patch_artist=True,
                    widths=0.6, showfliers=True)
    boxplot_colors(bp, colors_bp)
    for whisker in bp["whiskers"]:
        whisker.set(linewidth=1.5, color="black", alpha=0.7)
    for cap in bp["caps"]:
        cap.set(linewidth=1.5, color="black", alpha=0.7)
    for median in bp["medians"]:
        median.set(linewidth=2, color="red")
    for flier in bp["fliers"]:
        flier.set(marker="o", markerfacecolor="gray", markersize=5, alpha=0.5)

    for i, n in enumerate([len(v) for v in data_list], 1):
        ax.text(i, ax.get_ylim()[1] * 0.95, f"n={n}", ha="center",
                fontsize=10, fontweight="bold")

    if quality_threshold is not None:
        ax.axhline(y=quality_threshold, color="red", linestyle="--", linewidth=2, alpha=0.7,
                   label=f"Quality Threshold (μ+1σ = {quality_threshold:.4f})")
        ax.legend(loc="upper left", fontsize=10)

    labels_map = {"GT": "GT", "TTS": "TTS", "WAVEFORM": "Waveform"}
    mann_whitney_box(ax, data_dict,
                     [("GT", "TTS"), ("GT", "WAVEFORM"), ("TTS", "WAVEFORM")],
                     labels_map=labels_map, position=(0.98, 0.02))

    ax.set_ylabel("Character Error Rate (CER)", fontsize=12, fontweight="bold")
    ax.set_title("CER Distribution: GT vs TTS vs Waveform",
                 fontsize=13, fontweight="bold", pad=20)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_axisbelow(True)
    save_fig(os.path.join(out_dir, "rq2_cer_boxplot.png"))


def plot_quality_threshold_analysis(df: pd.DataFrame, quality_data: dict, out_dir: str):
    """6-panel: TTS pie, Waveform pie, overall bar, TTS by PESQ, Waveform by PESQ, info."""
    threshold = quality_data.get("quality_threshold")
    if threshold is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"CER Quality Threshold Analysis  (threshold = {threshold:.4f})",
                 fontsize=14, fontweight="bold", y=1.01)

    def _pie(ax, good, total, title, color):
        bad = total - good
        ax.pie([good, bad],
               labels=[f"Good\n({good}/{total})", f"Poor\n({bad}/{total})"],
               colors=[color, "#e5e7eb"],
               autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax.set_title(title, fontweight="bold")

    for ax_idx, (cond, color, label) in enumerate(
        [("TTS", C_TTS, "TTS Quality"), ("WAVEFORM", C_WAVEFORM, "Waveform Quality")]
    ):
        o = quality_data.get("overall", {}).get(cond, {})
        _pie(axes[0, ax_idx], o.get("good_quality_count", 0),
             o.get("total_responses", 1), label, color)

    ax    = axes[0, 2]
    conds = ["TTS", "WAVEFORM"]
    pcts  = [quality_data["overall"].get(c, {}).get("good_quality_pct", 0) for c in conds]
    clrs  = [C_TTS, C_WAVEFORM]
    bars  = ax.bar(["TTS", "Waveform"], pcts, color=clrs, alpha=0.8,
                   edgecolor="white", linewidth=1.5)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", fontweight="bold", fontsize=10)
    ax.set_ylabel("Good Quality (%)", fontweight="bold")
    ax.set_title("Good Quality by Method", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    for ax_idx, (cond, label) in enumerate([("TTS", "TTS"), ("WAVEFORM", "Waveform")]):
        ax      = axes[1, ax_idx]
        bp_data = quality_data.get("by_pesq_threshold", {}).get(cond, {})
        sub_labels, sub_vals, sub_colors = [], [], []
        for pesq_label, display, color in [
            ("pesq_met",     "PESQ ≤ 0.2\n(Met)",    "#22c55e"),
            ("pesq_not_met", "PESQ > 0.2\n(Not Met)", "#ef4444"),
        ]:
            entry = bp_data.get(pesq_label)
            if entry:
                sub_labels.append(display)
                sub_vals.append(entry.get("good_quality_pct", 0))
                sub_colors.append(color)
        if sub_vals:
            bars = ax.bar(sub_labels, sub_vals, color=sub_colors,
                          alpha=0.8, edgecolor="white", linewidth=1.5)
            for bar, pct in zip(bars, sub_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{pct:.1f}%", ha="center", fontweight="bold", fontsize=10)
        ax.set_ylabel("Good Quality (%)", fontweight="bold")
        ax.set_title(f"{label}: Good Quality by PESQ", fontweight="bold")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 2]
    ax.axis("off")
    gt_s = quality_data.get("gt_stats", {})
    info = (
        f"Quality Threshold Definition\n"
        f"{'='*35}\n\n"
        f"Threshold = Mean(GT CER) + 1σ\n\n"
        f"GT CER Mean : {gt_s.get('mean', 0):.4f}\n"
        f"GT CER Std  : {gt_s.get('std',  0):.4f}\n"
        f"Threshold   : {threshold:.4f}\n\n"
        f"Interpretation:\n"
        f"  ΔCE ≤ {threshold:.4f} → good quality\n"
        f"  ΔCE  > {threshold:.4f} → poor quality"
    )
    ax.text(0.1, 0.9, info, transform=ax.transAxes, fontsize=10,
            va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

    os.makedirs(os.path.join(out_dir, "cer"), exist_ok=True)
    save_fig(os.path.join(out_dir, "cer", "cer_quality_analysis.png"))


def plot_sbert_recovery(df: pd.DataFrame, out_dir: str):
    """SBERT recovery gain boxplots (guess vs ASR baseline) by condition."""
    req = ["sbert_gt_guess_normalized", "sbert_gt_asr_normalized"]
    if not all(c in df.columns for c in req):
        print("[!] SBERT columns missing, skipping SBERT recovery plot")
        return

    conditions = ["GT", "TTS", "WAVEFORM"]
    labels     = ["GT", "TTS", "Waveform"]
    colors_bp  = [C_GT, C_TTS, C_WAVEFORM]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax   = axes[0]
    data = [df[df["condition"] == c]["sbert_gt_asr_normalized"].dropna().values
            for c in conditions]
    bp   = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6, showfliers=False)
    boxplot_colors(bp, colors_bp)
    ax.set_ylabel("SBERT(GT, ASR transcription)", fontsize=11, fontweight="bold")
    ax.set_title("ASR Semantic Similarity by Condition", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    ax        = axes[1]
    gain_data = []
    for cond in conditions:
        sub  = df[df["condition"] == cond][req].dropna()
        gain = (sub["sbert_gt_guess_normalized"] - sub["sbert_gt_asr_normalized"]).values
        gain_data.append(gain)
    bp = ax.boxplot(gain_data, labels=labels, patch_artist=True, widths=0.6, showfliers=False)
    boxplot_colors(bp, colors_bp)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label="No gain")
    ax.set_ylabel("SBERT Gain (Guess − ASR)", fontsize=11, fontweight="bold")
    ax.set_title("Semantic Recovery Gain by Condition", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    save_fig(os.path.join(out_dir, "rq2_sbert_recovery.png"))


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str):
    """2×2 correlation heatmap: Pearson / Spearman overall + per-condition + scatter."""
    pesq_col  = "pesq"       if "pesq"      in df.columns else "PESQ"
    utmos_col = "UTMOS"      if "UTMOS"     in df.columns else "utmos"
    mos_col   = "mos_rating" if "mos_rating" in df.columns else "MOS"

    available = [c for c in [pesq_col, utmos_col, mos_col] if c in df.columns]
    if len(available) < 2:
        print("[!] Not enough columns for correlation heatmap")
        return

    valid = df[available].dropna()
    if len(valid) < 3:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, method_name in [(axes[0, 0], "pearson"), (axes[0, 1], "spearman")]:
        corr = valid.corr(method=method_name)
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Correlation"})
        ax.set_title(f"{method_name.capitalize()} (n={len(valid)})",
                     fontweight="bold", fontsize=11)

    cond_rows = {}
    for cond in ["TTS", "WAVEFORM"]:
        sub = df[df["condition"] == cond][available].dropna()
        if len(sub) > 2:
            c   = sub.corr(method="pearson")
            row = {}
            if pesq_col  in available and mos_col   in available:
                row["PESQ↔MOS"]   = c.loc[pesq_col,  mos_col]
            if utmos_col in available and mos_col   in available:
                row["UTMOS↔MOS"]  = c.loc[utmos_col, mos_col]
            if pesq_col  in available and utmos_col in available:
                row["PESQ↔UTMOS"] = c.loc[pesq_col,  utmos_col]
            cond_rows[cond] = row
    if cond_rows:
        cond_df = pd.DataFrame(cond_rows).T
        sns.heatmap(cond_df, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    vmin=-1, vmax=1, ax=axes[1, 0], cbar_kws={"label": "Correlation"})
        axes[1, 0].set_title("Pearson by Condition (Adversarial)",
                              fontweight="bold", fontsize=11)
    else:
        axes[1, 0].axis("off")

    ax = axes[1, 1]
    if utmos_col in df.columns and mos_col in df.columns:
        colors_dict = {"GT": C_GT, "TTS": C_TTS, "WAVEFORM": C_WAVEFORM}
        for cond in ["GT", "TTS", "WAVEFORM"]:
            sub = df[df["condition"] == cond][[utmos_col, mos_col]].dropna()
            if len(sub) > 0:
                ax.scatter(sub[utmos_col], sub[mos_col], alpha=0.5, s=30,
                           label=cond, color=colors_dict.get(cond, "gray"))
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
        if utmos_col in valid and mos_col in valid:
            r_all, _ = pearsonr(valid[utmos_col], valid[mos_col])
            ax.text(0.05, 0.95, f"r={r_all:.3f}", transform=ax.transAxes,
                    fontsize=11, va="top", fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        ax.set_xlabel(utmos_col, fontweight="bold")
        ax.set_ylabel(mos_col, fontweight="bold")
        ax.set_title("UTMOS vs MOS (scatter)", fontweight="bold", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.axis("off")

    os.makedirs(os.path.join(out_dir, "correlation"), exist_ok=True)
    save_fig(os.path.join(out_dir, "correlation", "correlation_matrix_heatmap.png"))


def _spearman_scatter_panel(ax, df, x_col, y_col, title):
    """Helper: scatter with success/failure + Spearman annotation on one axes."""
    valid   = df[[x_col, y_col, "is_success"]].dropna()
    success = valid[valid["is_success"]]
    failed  = valid[~valid["is_success"]]
    scatter_outcome(ax, valid, x_col, y_col, success_col="is_success",
                    c_success=C_SUCCESS, c_fail=C_FAILURE)
    spearman_box(ax, valid[x_col], valid[y_col])
    ax.set_xlabel(x_col, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_col, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10)
    return {
        "spearman_r": round(float(spearmanr(valid[x_col], valid[y_col])[0]), 4),
        "p_value":    round(float(spearmanr(valid[x_col], valid[y_col])[1]), 4),
        "n_success":  len(success), "n_failure": len(failed), "n_total": len(valid),
    }


def plot_spearman_cer_correlations(df: pd.DataFrame, out_dir: str):
    """CER vs PESQ / UTMOS scatter (TTS only, success/failure separation)."""
    cer_col = "CER"
    if cer_col not in df.columns:
        print("[!] CER column missing, skipping CER Spearman plot")
        return {}

    uid_col  = "Unique ID.1" if "Unique ID.1" in df.columns else None
    task_col = "Task"        if "Task"        in df.columns else None

    tts_df = df[df["condition"] == "TTS"].copy()
    if task_col:
        tts_df = tts_df[tts_df[task_col] == "listen+rate"]

    agg_dict = {cer_col: "mean", "pesq": "first", "set_overlap": "first"}
    if "UTMOS" in tts_df.columns:
        agg_dict["UTMOS"] = "first"

    if uid_col and uid_col in tts_df.columns:
        grouped = tts_df.groupby(uid_col).agg(
            {k: v for k, v in agg_dict.items() if k in tts_df.columns}
        ).dropna(subset=[cer_col])
    else:
        grouped = tts_df

    grouped = grouped.copy()
    grouped["is_success"] = grouped.apply(
        lambda r: _is_success(r.get("pesq"), r.get("set_overlap")), axis=1)

    y_targets = [c for c in ["pesq", "UTMOS"] if c in grouped.columns]
    if not y_targets:
        return {}

    fig, axes = plt.subplots(1, len(y_targets), figsize=(8 * len(y_targets), 7))
    if len(y_targets) == 1:
        axes = [axes]

    all_results = {}
    for ax, y_col in zip(axes, y_targets):
        valid = grouped[[cer_col, y_col, "is_success"]].dropna()
        if len(valid) < 2:
            continue
        all_results[f"CER_vs_{y_col}"] = _spearman_scatter_panel(
            ax, valid, cer_col, y_col, f"CER vs {y_col}")

    plt.suptitle("TTS: CER Correlations (Success/Failure)",
                 fontweight="bold", fontsize=14, y=0.995)
    save_fig(os.path.join(out_dir, "rq2_cer_spearman_correlations.png"))
    return all_results


def plot_spearman_mos_correlations(df: pd.DataFrame, out_dir: str):
    """MOS vs UTMOS / PESQ / CER scatter (TTS only, success/failure separation)."""
    mos_col  = "mos_rating" if "mos_rating" in df.columns else None
    uid_col  = "Unique ID.1" if "Unique ID.1" in df.columns else None
    task_col = "Task"        if "Task"        in df.columns else None

    if mos_col is None:
        print("[!] MOS column missing, skipping MOS Spearman plot")
        return {}

    tts_df = df[df["condition"] == "TTS"].copy()
    if task_col:
        tts_df = tts_df[tts_df[task_col] == "listen+rate"]

    agg_dict = {mos_col: "mean", "pesq": "first", "set_overlap": "first"}
    for c in ["UTMOS", "CER"]:
        if c in tts_df.columns:
            agg_dict[c] = "first"

    if uid_col and uid_col in tts_df.columns:
        grouped = tts_df.groupby(uid_col).agg(
            {k: v for k, v in agg_dict.items() if k in tts_df.columns}
        ).dropna(subset=[mos_col])
    else:
        grouped = tts_df

    grouped = grouped.copy()
    grouped["is_success"] = grouped.apply(
        lambda r: _is_success(r.get("pesq"), r.get("set_overlap")), axis=1)

    x_targets = [c for c in ["UTMOS", "pesq", "CER"] if c in grouped.columns]
    if not x_targets:
        return {}

    fig, axes = plt.subplots(1, len(x_targets), figsize=(6 * len(x_targets), 6))
    if len(x_targets) == 1:
        axes = [axes]

    all_results = {}
    for ax, x_col in zip(axes, x_targets):
        valid = grouped[[x_col, mos_col, "is_success"]].dropna()
        if len(valid) < 2:
            continue
        all_results[f"{x_col}_vs_MOS"] = _spearman_scatter_panel(
            ax, valid, x_col, mos_col, f"{x_col} vs MOS")
        ax.legend(fontsize=10, loc="lower right")

    plt.suptitle("TTS: MOS Correlations (Success/Failure)",
                 fontweight="bold", fontsize=14, y=0.995)
    save_fig(os.path.join(out_dir, "rq2_spearman_UTMOS-MOS_PESQ-MOS_CER-MOS_TTS.png"))
    return all_results


def plot_venn_3way(df: pd.DataFrame, out_dir: str):
    """3-way Venn: adversarial samples meeting PESQ / MOS / UTMOS thresholds."""
    pesq_col  = "pesq"       if "pesq"      in df.columns else "PESQ"
    mos_col   = "mos_rating" if "mos_rating" in df.columns else "MOS"
    utmos_col = "UTMOS"      if "UTMOS"     in df.columns else "utmos"

    missing = [c for c in [pesq_col, mos_col, utmos_col] if c not in df.columns]
    if missing:
        print(f"[!] Venn 3-way skipped — missing columns: {missing}")
        return

    adv = df[df["condition"].isin(["TTS", "WAVEFORM"])][
        [pesq_col, mos_col, utmos_col]
    ].dropna()

    pesq_set  = set(adv[adv[pesq_col]  <= THR_PESQ].index)
    mos_set   = set(adv[adv[mos_col]   >= THR_MOS].index)
    utmos_set = set(adv[adv[utmos_col] >= THR_UTMOS].index)

    os.makedirs(os.path.join(out_dir, "correlation"), exist_ok=True)
    venn3_simple(
        [pesq_set, mos_set, utmos_set],
        labels=(
            f"PESQ ≤ {THR_PESQ}\n(Optimization Goal)",
            f"MOS ≥ {THR_MOS}\n(Acceptable Quality)",
            f"UTMOS ≥ {THR_UTMOS}\n(Predicted Quality)",
        ),
        title=(f"3-Way Venn: Adversarial Samples Meeting Quality Thresholds\n"
               f"(TTS & Waveform, n={len(adv)})"),
        out=os.path.join(out_dir, "correlation", "venn_3way_thresholds.png"),
    )


def _kde_with_mann_whitney(df, col, group_col, color_map, labels_map,
                           title, xlabel, out_path, threshold=None, threshold_label=None):
    """Shared KDE + Mann-Whitney box for a single metric grouped by condition."""
    listen_df  = df[df["Task"] == "listen+rate"].copy() if "Task" in df.columns else df.copy()
    conditions = sorted(listen_df[group_col].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    data_dict = {}
    for cond in conditions:
        vals = listen_df[listen_df[group_col] == cond][col].dropna().values
        if len(vals) > 1:
            lbl = labels_map.get(cond, cond)
            sns.kdeplot(data=vals, color=color_map.get(cond, "gray"), linewidth=2,
                        label=f"{lbl} (n={len(vals)})", fill=True, alpha=0.3, ax=ax)
            data_dict[cond] = vals

    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", linewidth=2, alpha=0.7,
                   label=threshold_label or f"threshold ({threshold})")

    conds = list(data_dict.keys())
    pairs = [(conds[i], conds[j]) for i in range(len(conds)) for j in range(i + 1, len(conds))]
    mann_whitney_box(ax, data_dict, pairs, labels_map=labels_map)

    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    save_fig(out_path)


def plot_cer_mann_whitney_kde(df: pd.DataFrame, out_dir: str):
    """KDE of CER distribution by condition with Mann-Whitney U annotations."""
    if "CER" not in df.columns:
        print("[!] CER column missing, skipping CER KDE")
        return
    color_map  = {"GT": C_GT, "TTS": C_TTS, "WAVEFORM": C_WAVEFORM}
    labels_map = {"GT": "GT", "TTS": "TTS", "WAVEFORM": "Waveform"}
    _kde_with_mann_whitney(
        df, "CER", "condition", color_map, labels_map,
        title="CER Distribution by Condition (KDE + Mann-Whitney U)",
        xlabel="Character Error Rate (CER)",
        out_path=os.path.join(out_dir, "rq2_cer_mann_whitney_kde.png"),
    )


def plot_mos_mann_whitney_kde(df: pd.DataFrame, out_dir: str):
    """KDE of MOS distribution by condition with Mann-Whitney U annotations."""
    if "mos_rating" not in df.columns:
        print("[!] MOS column missing, skipping MOS KDE")
        return
    color_map  = {"GT": C_GT, "TTS": C_TTS, "WAVEFORM": C_WAVEFORM}
    labels_map = {"GT": "GT", "TTS": "TTS", "WAVEFORM": "Waveform"}
    _kde_with_mann_whitney(
        df, "mos_rating", "condition", color_map, labels_map,
        title="MOS Distribution by Condition (KDE + Mann-Whitney U)",
        xlabel="Mean Opinion Score (MOS)",
        out_path=os.path.join(out_dir, "rq2_mos_mann_whitney_kde.png"),
        threshold=THR_MOS, threshold_label=f"MOS threshold ({THR_MOS})",
    )


# =============================================================================
# Section 4 — Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RQ2 Human Validation Analysis")
    parser.add_argument("--survey_csv",  default="outputs/human_survey.csv")
    parser.add_argument("--output_dir",  default="outputs/thesis_analysis/RQ2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("RQ2: HUMAN VALIDATION ANALYSIS")
    print(f"{'='*70}")
    print(f"Survey CSV : {args.survey_csv}")
    print(f"Output dir : {args.output_dir}")

    print("\n[1] Loading survey data...")
    df = load_survey(args.survey_csv)
    print(f"    Rows: {len(df)}")
    if "condition" in df.columns:
        print(f"    Conditions: {sorted(df['condition'].dropna().unique().tolist())}")

    print("\n[2] Ensuring CER columns...")
    df = ensure_cer_columns(df)

    print("\n[3] Computing quality threshold...")
    quality_data = compute_quality_threshold(df)
    thr = quality_data["quality_threshold"]
    print(f"    Threshold = {thr:.4f}  "
          f"(GT mean={quality_data['gt_stats']['mean']:.4f} "
          f"+ 1σ={quality_data['gt_stats']['std']:.4f})")
    for cond in ["TTS", "WAVEFORM"]:
        o = quality_data["overall"].get(cond, {})
        print(f"    {cond:10s}: {o.get('good_quality_count','?')}/{o.get('total_responses','?')} "
              f"({o.get('good_quality_pct','?'):.1f}%) good quality")

    print("\n[4] Computing correlation statistics...")
    corr_data = compute_correlation_stats(df)

    print("\n[5] Building summary...")
    summary = build_summary(df, quality_data, corr_data)
    print_summary(summary)

    print("\n[6] Generating visualizations...")
    plot_mos_boxplots(df, args.output_dir)
    plot_mos_mann_whitney_kde(df, args.output_dir)
    plot_cer_boxplot_mann_whitney(df, args.output_dir, quality_threshold=thr)
    plot_cer_mann_whitney_kde(df, args.output_dir)
    cer_spearman = plot_spearman_cer_correlations(df, args.output_dir)
    mos_spearman = plot_spearman_mos_correlations(df, args.output_dir)
    plot_quality_threshold_analysis(df, quality_data, args.output_dir)
    plot_sbert_recovery(df, args.output_dir)
    plot_correlation_heatmap(df, args.output_dir)
    plot_venn_3way(df, args.output_dir)

    if cer_spearman:
        summary["cer_spearman_correlations"] = cer_spearman
    if mos_spearman:
        summary["mos_spearman_correlations"] = mos_spearman
    summary["cer_quality_analysis"] = quality_data

    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    summary_path = os.path.join(data_dir, "rq2_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] {summary_path}")

    print(f"\n{'='*70}")
    print(f"[✓] RQ2 analysis complete. Results saved to: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
