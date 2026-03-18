#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import warnings
from upsetplot import UpSet

# Suppress warnings for cleaner console output
warnings.simplefilter(action='ignore', category=FutureWarning)

# ============================================================================
# Configuration
# ============================================================================

COMBINED_CSV = "outputs/thesis_analysis/RQ1/data/combined_results.csv"
OUTPUT_DIR = "outputs/thesis_analysis/RQ1/02_success_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Success thresholds
THR_PESQ = 0.2
THR_SET_OVERLAP = 0.5
THR_SBERT = 0.6
THR_UTMOS = 3.5  

COLORS = {'TTS': '#3b82f6', 'Waveform': '#06b6d4'}

print("="*80)
print("UPSET PLOT: TTS vs Waveform Success Criteria (Percentages)")
print("="*80)

# ============================================================================
# Data Processing
# ============================================================================

def get_counts_by_method(path, method):
    """Loads combined CSV and returns counts for specific method"""
    if not os.path.exists(path):
        print(f"[!] Warning: File not found {path}")
        return pd.Series(dtype=float), 0

    df = pd.read_csv(path)
    # Filter by method
    df = df[df['method'] == method]

    # Create boolean indicators (Success = True)
    indicators = pd.DataFrame({
        'SET_OVERLAP': df['set_overlap'] <= THR_SET_OVERLAP,
        'PESQ': df['pesq'] <= THR_PESQ,
        'SBERT': (df['semantic_similarity'].notna()) & (df['semantic_similarity'] <= THR_SBERT),
        'UTMOS': df['utmos_best'] >= THR_UTMOS
    })

    # Aggregate counts for each unique intersection
    counts = indicators.groupby(['SET_OVERLAP', 'PESQ', 'SBERT', 'UTMOS']).size()
    return counts, len(df)

print(f"[*] Processing data...")
tts_counts, tts_total = get_counts_by_method(COMBINED_CSV, 'TTS')
wf_counts, wf_total = get_counts_by_method(COMBINED_CSV, 'Waveform')

# Align datasets
all_combos = tts_counts.index.union(wf_counts.index)
combined_df = pd.DataFrame(index=all_combos)
combined_df['TTS_count'] = tts_counts
combined_df['WF_count'] = wf_counts
combined_df = combined_df.fillna(0)

# Calculate percentages
combined_df['TTS_pct'] = (combined_df['TTS_count'] / tts_total) * 100 if tts_total > 0 else 0
combined_df['WF_pct'] = (combined_df['WF_count'] / wf_total) * 100 if wf_total > 0 else 0

# Sort the combinations based on the TTS percentage in decreasing order
combined_df = combined_df.sort_values(by='TTS_pct', ascending=False)
sorted_index = combined_df.index

# ============================================================================
# Plotting
# ============================================================================

print(f"[*] Generating UpSet plot...")

# totals_plot_elements=0 removes the horizontal bars on the left
upset = UpSet(pd.Series(1, index=sorted_index), sort_by=None, show_counts=False, totals_plot_elements=0)

fig = plt.figure(figsize=(16, 9))
axes = upset.plot(fig)

ax_inter = axes['intersections']
ax_inter.clear() 

x = np.arange(len(sorted_index))
width = 0.35

# Plot side-by-side grouped bars
rects1 = ax_inter.bar(x - width/2, combined_df['TTS_pct'], width, 
                     label='TTS', color=COLORS['TTS'], alpha=0.85)
rects2 = ax_inter.bar(x + width/2, combined_df['WF_pct'], width, 
                     label='Waveform', color=COLORS['Waveform'], alpha=0.85)

# Styling
ax_inter.set_ylabel("Percentage of Total Runs (%)")
max_y = max(combined_df['TTS_pct'].max(), combined_df['WF_pct'].max())
ax_inter.set_ylim(0, max_y * 1.3) 

# Legend: Outside top left
ax_inter.legend(title="Method", frameon=True, loc='lower left', 
                bbox_to_anchor=(0, 1.05), ncol=2)
ax_inter.grid(axis='y', linestyle=':', alpha=0.5)

# Add percentage labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax_inter.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=45)

autolabel(rects1)
autolabel(rects2)

output_path = os.path.join(OUTPUT_DIR, "rq1_upset_combined_pct.png")
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.close()

print(f"[✓] Saved Plot: {output_path}")

# ============================================================================
# Summary
# ============================================================================

intersections_json = {
    "TTS": {str(k): float(v) for k, v in combined_df['TTS_pct'].to_dict().items()},
    "Waveform": {str(k): float(v) for k, v in combined_df['WF_pct'].to_dict().items()}
}

summary = {
    "thresholds": {
        "PESQ": THR_PESQ, "SET_OVERLAP": THR_SET_OVERLAP, 
        "SBERT": THR_SBERT, "UTMOS": THR_UTMOS
    },
    "totals": {"TTS": tts_total, "Waveform": wf_total},
    "intersections_pct": intersections_json
}

