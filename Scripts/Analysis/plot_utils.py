"""
Shared plotting utilities for RQ1, RQ2, RQ3 analysis scripts.

Provides:
  Constants  — colors, DPI, thresholds
  Data       — iqr_stats, format_p
  Layout     — method_colors, method_subplots, save_fig
  Primitives — boxplot_colors, outlier_scatter, threshold_shading,
               scatter_outcome, kde_by_group
  Stats      — spearman_box, mann_whitney_box
  Venn       — annotate_venn2, venn2_by_method, venn3_simple
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr
from matplotlib_venn import venn2, venn3

# ── Constants ──────────────────────────────────────────────────────────────────

C_GT       = "#9ca3af"   # Gray  (ground truth)
C_TTS      = "#3b82f6"   # Blue  (TTS method)
C_WAVEFORM = "#06b6d4"   # Teal  (Waveform method)
C_SUCCESS  = "#2ecc71"   # Green (attack succeeded)
C_FAIL     = "#ef4444"   # Red   (attack failed)

DPI = 300

THR_PESQ  = 0.2
THR_SO    = 0.5
THR_SBERT = 0.6
THR_UTMOS = 3.5
THR_MOS   = 3.5


# ── Data Helpers ───────────────────────────────────────────────────────────────

def iqr_stats(series, include_mean: bool = True, include_std: bool = True,
              include_iqr: bool = True) -> dict:
    """Descriptive statistics: median, q1, q3, n, plus optional mean/std/iqr."""
    s = series.dropna()
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    result = {
        "median": round(float(s.median()), 4),
        "q1":     round(q1, 4),
        "q3":     round(q3, 4),
        "n":      int(len(s)),
    }
    if include_mean:
        result["mean"] = round(float(s.mean()), 4)
    if include_std:
        result["std"] = round(float(s.std()), 4)
    if include_iqr:
        result["iqr"] = round(q3 - q1, 4)
    return result


def format_p(p: float) -> str:
    """Format p-value with significance stars."""
    if p < 0.001:
        return "< 0.001***"
    elif p < 0.01:
        return f"{p:.4f}**"
    elif p < 0.05:
        return f"{p:.4f}*"
    return f"{p:.4f}ns"


# ── Layout Helpers ─────────────────────────────────────────────────────────────

def method_colors(df) -> dict:
    """Map method names to standard colors: first→TTS blue, second→Waveform teal."""
    methods = sorted(df["method"].unique())
    palette = [C_TTS, C_WAVEFORM]
    return {m: palette[i] for i, m in enumerate(methods)}


def method_subplots(df, nrows: int = 1, figsize_base: tuple = (7, 6)):
    """Create one subplot column per method.

    Returns (fig, axes_flat, methods, colors_m).
    axes_flat is a flat list for nrows==1; a 2-D ndarray for nrows>1.
    """
    methods = sorted(df["method"].unique())
    colors_m = method_colors(df)
    ncols = len(methods)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_base[0] * ncols, figsize_base[1] * nrows))
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    elif nrows == 1 or ncols == 1:
        axes_flat = list(np.array(axes).flatten())
    else:
        axes_flat = axes  # 2-D ndarray, caller uses axes_flat[row, col]
    return fig, axes_flat, methods, colors_m


def save_fig(path: str, fig=None, dpi: int = DPI):
    """Tight-layout, save, close, and print confirmation."""
    if fig is None:
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    print(f"[Saved] {os.path.basename(path)}")


# ── Plot Primitives ────────────────────────────────────────────────────────────

def boxplot_colors(bp, colors, alpha: float = 0.7):
    """Apply facecolors to a patch_artist=True boxplot."""
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(alpha)


def outlier_scatter(ax, data_list, colors, positions=None, jitter: float = 0.02):
    """Overlay IQR outlier jitter on a showfliers=False boxplot."""
    for i, (vals, color) in enumerate(zip(data_list, colors)):
        pos = positions[i] if positions is not None else i + 1
        if len(vals) == 0:
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        out_vals = vals[(vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)]
        if len(out_vals):
            ax.scatter(np.random.normal(pos, jitter, len(out_vals)), out_vals,
                       color=color, s=60, alpha=0.8, edgecolors="black",
                       linewidth=0.5, zorder=10)


def threshold_shading(ax, threshold: float, side: str = "left"):
    """Shade the 'good' region of a KDE/distribution plot.

    side='left'  → good = x ≤ threshold  (PESQ, SetOverlap, SBERT)
    side='right' → good = x ≥ threshold  (UTMOS)
    """
    y_max = ax.get_ylim()[1]
    x_min, x_max = ax.get_xlim()
    kw = dict(facecolor="none", hatch="///", edgecolor="red", alpha=0.5, zorder=1)
    if side == "left":
        ax.fill_between([x_min, threshold], 0, y_max, color="green", alpha=0.07, zorder=1)
        ax.fill_between([threshold, x_max], 0, y_max, **kw)
    else:
        ax.fill_between([x_min, threshold], 0, y_max, **kw)
        ax.fill_between([threshold, x_max], 0, y_max, color="green", alpha=0.07, zorder=1)


def scatter_outcome(ax, df, x_col: str, y_col: str,
                    success_col: str = "success",
                    c_success=C_SUCCESS, c_fail=C_FAIL,
                    alpha: float = 0.85, s: int = 80):
    """Scatter plot with success/failure color split."""
    for outcome, color, label in [(True, c_success, "Success"),
                                   (False, c_fail,    "Failure")]:
        mask = df[success_col] == outcome
        n = int(mask.sum())
        if n > 0:
            ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                       color=color, label=f"{label} (n={n})",
                       alpha=alpha, s=s, edgecolors="black", linewidth=0.5)


def kde_by_group(ax, df, col: str, group_col: str, color_map: dict,
                 fill: bool = True, alpha: float = 0.3, linewidth: float = 2.0):
    """Plot one filled KDE per group on shared axes."""
    for group in sorted(df[group_col].unique()):
        data = df[df[group_col] == group][col].dropna().values
        if len(data) > 1:
            sns.kdeplot(data=data, color=color_map.get(group, "gray"),
                        linewidth=linewidth, label=f"{group} (n={len(data)})",
                        fill=fill, alpha=alpha, ax=ax)


# ── Statistical Annotations ────────────────────────────────────────────────────

def spearman_box(ax, x, y, position: tuple = (0.95, 0.95), fontsize: int = 11):
    """Add Spearman ρ / p / n annotation box to axes."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return
    r, p = spearmanr(x, y)
    ax.text(position[0], position[1],
            f"ρ = {r:.4f}\np = {format_p(p)}\nn = {len(x)}",
            transform=ax.transAxes, fontsize=fontsize, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
            fontfamily="monospace", fontweight="bold")


def mann_whitney_box(ax, data_dict: dict, pairs,
                     labels_map: dict = None,
                     position: tuple = (0.98, 0.97),
                     include_rbc: bool = False):
    """Add Mann-Whitney U annotation box.

    data_dict  : {label: 1-D array-like}
    pairs      : list of (label_a, label_b)
    labels_map : optional display rename {internal: display}
    include_rbc: add rank-biserial correlation per pair
    """
    labels_map = labels_map or {}
    lines = []
    for (a, b) in pairs:
        ga = np.asarray(data_dict.get(a, []), dtype=float)
        gb = np.asarray(data_dict.get(b, []), dtype=float)
        ga = ga[~np.isnan(ga)]
        gb = gb[~np.isnan(gb)]
        if len(ga) == 0 or len(gb) == 0:
            continue
        u, p = mannwhitneyu(ga, gb, alternative="two-sided")
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        la, lb = labels_map.get(a, a), labels_map.get(b, b)
        line = f"{la} vs {lb}: U={u:.0f}, p {sig}"
        if include_rbc:
            rbc = 1 - (2 * u) / (len(ga) * len(gb))
            line += f", rbc={rbc:.2f}"
        lines.append(line)
    if lines:
        ax.text(position[0], position[1], "\n".join(lines),
                transform=ax.transAxes, fontsize=9, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                fontfamily="monospace")


# ── Venn Diagrams ──────────────────────────────────────────────────────────────

def annotate_venn2(v, a10: int, a01: int, a11: int, n_total: int):
    """Set percentage + count labels on a venn2 object."""
    for sid, count in [("10", a10), ("01", a01), ("11", a11)]:
        lbl = v.get_label_by_id(sid)
        if lbl:
            lbl.set_text(f"{100 * count / n_total:.1f}%\n({count})")
            lbl.set_fontsize(10)
            lbl.set_fontweight("bold")


def venn2_by_method(df, set_a_fn, set_b_fn,
                    label_a: str, label_b: str,
                    suptitle: str, out: str,
                    color_b: str = "#f59e0b"):
    """Side-by-side venn2, one panel per method.

    set_a_fn(sub_df) -> set of row indices belonging to set A
    set_b_fn(sub_df) -> set of row indices belonging to set B
    """
    methods = sorted(df["method"].unique())
    colors_m = method_colors(df)
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6))
    if len(methods) == 1:
        axes = [axes]
    for idx, method in enumerate(methods):
        ax = axes[idx]
        sub = df[df["method"] == method]
        set_a = set_a_fn(sub)
        set_b = set_b_fn(sub)
        n = len(sub)
        a10 = len(set_a - set_b)
        a01 = len(set_b - set_a)
        a11 = len(set_a & set_b)
        neither = n - len(set_a | set_b)
        c = colors_m.get(method, C_TTS)
        v = venn2(subsets=(1, 1, 1), set_labels=(label_a, label_b),
                  set_colors=(c, color_b), alpha=0.6, ax=ax)
        annotate_venn2(v, a10, a01, a11, n)
        ax.text(0.97, 0.03, f"Neither: {100 * neither / n:.1f}%\n({neither})",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="dimgray")
        ax.set_title(f"{method} (n={n})", fontsize=12, fontweight="bold")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    save_fig(out)


def venn3_simple(sets, labels, title: str, out: str):
    """Single 3-way Venn diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))
    venn3(sets, set_labels=labels, ax=ax)
    plt.title(title, fontweight="bold", fontsize=12)
    save_fig(out)
