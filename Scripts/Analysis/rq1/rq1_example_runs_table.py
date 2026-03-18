"""
Generate comparison tables with diverse example runs.
TTS: varied SET_OVERLAP values (0.0, 0.2, 0.4, 0.5, 0.6)
Waveform: random diverse transcriptions
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================================
# Configuration
# ============================================================================

TTS_CSV = "outputs/results/TTS/all_results.csv"
WF_CSV = "outputs/results/Waveform/all_results.csv"
OUTPUT_DIR = "outputs/thesis_analysis/RQ1"

# ============================================================================
# Load Data
# ============================================================================

print("="*80)
print("GENERATING EXAMPLE RUNS TABLE")
print("="*80)

tts_df = pd.read_csv(TTS_CSV)
wf_df = pd.read_csv(WF_CSV)

# Normalise column names from raw CSVs
for _df in [tts_df, wf_df]:
    _df.rename(columns={"score_SET_OVERLAP": "set_overlap", "score_PESQ": "pesq"}, inplace=True)
    if "set_overlap" not in _df.columns and "SET_OVERLAP" in _df.columns:
        _df.rename(columns={"SET_OVERLAP": "set_overlap"}, inplace=True)
    if "pesq" not in _df.columns and "PESQ" in _df.columns:
        _df.rename(columns={"PESQ": "pesq"}, inplace=True)

print(f"\n[*] Loaded TTS: {len(tts_df)} runs")
print(f"[*] Loaded Waveform: {len(wf_df)} runs")

# ============================================================================
# TTS: Select runs with reasonable transcription lengths, increasing SET_OVERLAP
# ============================================================================

print(f"\n[*] Selecting TTS runs with diverse SET_OVERLAP values...")

tts_all = tts_df.copy()
tts_all['asr_length'] = tts_all['asr_transcription'].str.len()

# Filter: reasonable transcription lengths (between 5 and 60 chars)
tts_suitable = tts_all[(tts_all['asr_length'] >= 5) & (tts_all['asr_length'] <= 60)].copy()

print(f"  Total runs: {len(tts_all)}, Suitable (5-60 chars): {len(tts_suitable)}")

# Get unique SET_OVERLAP values in suitable data, sorted
unique_overlaps = sorted(tts_suitable['set_overlap'].unique())
print(f"  Unique SET_OVERLAP values: {len(unique_overlaps)}")

tts_sample = []
selected_overlaps = []

# Iteratively select runs at increasing SET_OVERLAP thresholds (all available values)
for overlap_val in unique_overlaps:
    # Find runs at this exact SET_OVERLAP that haven't been selected
    candidates = tts_suitable[
        (tts_suitable['set_overlap'] == overlap_val) &
        (~tts_suitable.index.isin([row.name for row in tts_sample]))
    ]

    if len(candidates) > 0:
        # Pick one (prefer success if available)
        successful = candidates[candidates['success'] == True]
        chosen = successful.iloc[0] if len(successful) > 0 else candidates.iloc[0]
        tts_sample.append(chosen)
        selected_overlaps.append(overlap_val)
        status = "✓" if chosen['success'] else "✗"
        n_available = len(candidates)
        print(f"  SET_OVERLAP {overlap_val:.3f}: Selected (asr_len={chosen['asr_length']}, {n_available} available) {status}")

tts_sample = pd.DataFrame(tts_sample).reset_index(drop=True)
print(f"  Total TTS samples: {len(tts_sample)}")

# ============================================================================
# Waveform: Select runs with reasonable transcription lengths, increasing SET_OVERLAP
# ============================================================================

print(f"\n[*] Selecting Waveform runs with diverse SET_OVERLAP values...")

wf_all = wf_df.copy()
wf_all['asr_length'] = wf_all['asr_transcription'].str.len()

# Filter: reasonable transcription lengths (between 10 and 70 chars)
wf_suitable = wf_all[(wf_all['asr_length'] >= 10) & (wf_all['asr_length'] <= 70)].copy()

print(f"  Total runs: {len(wf_all)}, Suitable (10-70 chars): {len(wf_suitable)}")

# Get unique SET_OVERLAP values in suitable data, sorted
unique_overlaps_wf = sorted(wf_suitable['set_overlap'].unique())
print(f"  Unique SET_OVERLAP values: {len(unique_overlaps_wf)}")

wf_sample = []
selected_overlaps_wf = []

# Iteratively select runs at increasing SET_OVERLAP thresholds (all available values)
for overlap_val in unique_overlaps_wf:
    # Find runs at this exact SET_OVERLAP that haven't been selected
    candidates = wf_suitable[
        (wf_suitable['set_overlap'] == overlap_val) &
        (~wf_suitable.index.isin([row.name for row in wf_sample]))
    ]

    if len(candidates) > 0:
        # Pick one (prefer success if available)
        successful = candidates[candidates['success'] == True]
        chosen = successful.iloc[0] if len(successful) > 0 else candidates.iloc[0]
        wf_sample.append(chosen)
        selected_overlaps_wf.append(overlap_val)
        status = "✓" if chosen['success'] else "✗"
        n_available = len(candidates)
        print(f"  SET_OVERLAP {overlap_val:.3f}: Selected (asr_len={chosen['asr_length']}, {n_available} available) {status}")

wf_sample = pd.DataFrame(wf_sample).reset_index(drop=True)
print(f"  Total Waveform samples: {len(wf_sample)}")

print(f"  Total Waveform samples: {len(wf_sample)}")
print(f"  Unique transcriptions: {wf_sample['asr_transcription'].nunique()}")

# ============================================================================
# Create Figure
# ============================================================================

print(f"\n[*] Creating visualization...")

fig = plt.figure(figsize=(16, 9))
plt.tight_layout(pad=50.0)

# TTS Table
ax1 = plt.subplot(2, 1, 1)
ax1.axis('off')
ax1.set_title('TTS: Every Unique SET_OVERLAP Score with Example', fontsize=14, fontweight='bold', pad=-15)

tts_table_data = [['ID', 'Ground Truth', 'ASR Transcription', 'PESQ', 'SET_OVERLAP', 'Runs with that\nthreshold']]

# Get counts for each SET_OVERLAP across ALL runs
tts_overlap_counts = tts_df['set_overlap'].value_counts().to_dict()

for idx, row in tts_sample.iterrows():
    overlap_val = row['set_overlap']
    count = tts_overlap_counts.get(overlap_val, 0)
    tts_table_data.append([
        str(int(row['sentence_id'])),
        row['ground_truth_text'],
        row['asr_transcription'],
        f"{row['pesq']:.3f}",
        f"{overlap_val:.3f}",
        str(count)
    ])

table1 = ax1.table(cellText=tts_table_data, cellLoc='left', loc='upper center',
                   colWidths=[0.06, 0.28, 0.28, 0.08, 0.08, 0.12])
table1.auto_set_font_size(False)
table1.set_fontsize(9)
table1.scale(1, 3.5)

# Style TTS table
for i in range(len(tts_table_data)):
    for j in range(6):
        cell = table1[(i, j)]
        if i == 0:  # Header
            cell.set_facecolor('#3b82f6')
            cell.set_text_props(weight='bold', color='white')
        else:
            if i % 2 == 0:
                cell.set_facecolor('#f3f4f6')
            else:
                cell.set_facecolor('white')

# Waveform Table
ax2 = plt.subplot(2, 1, 2)
ax2.axis('off')
ax2.set_title('Waveform: Every Unique SET_OVERLAP Score with Example', fontsize=14, fontweight='bold', pad=-15)

wf_table_data = [['ID', 'Ground Truth', 'ASR Transcription', 'PESQ', 'SET_OVERLAP', 'With That\nThreshold']]

# Get counts for each SET_OVERLAP across ALL runs
wf_overlap_counts = wf_df['set_overlap'].value_counts().to_dict()

for idx, row in wf_sample.iterrows():
    overlap_val = row['set_overlap']
    count = wf_overlap_counts.get(overlap_val, 0)
    wf_table_data.append([
        str(int(row['sentence_id'])),
        row['ground_truth_text'],
        row['asr_transcription'],
        f"{row['pesq']:.3f}",
        f"{overlap_val:.3f}",
        str(count)
    ])

table2 = ax2.table(cellText=wf_table_data, cellLoc='left', loc='upper center',
                   colWidths=[0.06, 0.28, 0.28, 0.08, 0.08, 0.12])
table2.auto_set_font_size(False)
table2.set_fontsize(9)
table2.scale(1, 3.5)

# Style Waveform table
for i in range(len(wf_table_data)):
    for j in range(6):
        cell = table2[(i, j)]
        if i == 0:  # Header
            cell.set_facecolor('#06b6d4')
            cell.set_text_props(weight='bold', color='white')
        else:
            if i % 2 == 0:
                cell.set_facecolor('#f3f4f6')
            else:
                cell.set_facecolor('white')

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "rq1_comparison_tables.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"[✓] Saved: {output_path}")

# ============================================================================
# Export sample tables to CSV
# ============================================================================

# Combined examples CSV (TTS + Waveform)
tts_export = tts_sample[['sentence_id', 'ground_truth_text', 'asr_transcription', 'set_overlap']].copy()
tts_export.columns = ['Sentence_ID', 'Ground_Truth', 'ASR_Output', 'SET_OVERLAP']
tts_export.insert(0, 'Method', 'TTS')

wf_export = wf_sample[['sentence_id', 'ground_truth_text', 'asr_transcription', 'set_overlap']].copy()
wf_export.columns = ['Sentence_ID', 'Ground_Truth', 'ASR_Output', 'SET_OVERLAP']
wf_export.insert(0, 'Method', 'Waveform')

combined_csv_path = os.path.join(OUTPUT_DIR, "rq1_example_runs.csv")
pd.concat([tts_export, wf_export], ignore_index=True).to_csv(combined_csv_path, index=False)
print(f"[✓] Saved: {combined_csv_path}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nTTS SET_OVERLAP Values: {[f'{x:.3f}' for x in tts_sample['set_overlap'].values]}")
print(f"Waveform Transcriptions: {wf_sample['asr_transcription'].nunique()} unique")
print("="*80)
