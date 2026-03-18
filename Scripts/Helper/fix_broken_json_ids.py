import json
import os
from pathlib import Path
import re

# Paths
results_dir = "outputs/results"

def fix_json_ids_in_folder(folder_path):
    """Fix sentence_id and run_id in JSON files that don't match folder structure."""

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return 0, 0

    fixed_count = 0
    skipped_count = 0

    # Walk through all directories
    for root, dirs, files in os.walk(folder_path):
        # Extract sentence_id and run_id from path
        path_parts = root.replace(folder_path, "").strip(os.sep).split(os.sep)

        if len(path_parts) < 2:
            continue

        # Extract from "sentence_018" and "run_2"
        sentence_match = re.search(r'sentence_(\d+)', path_parts[0])
        run_match = re.search(r'run_(\d+)', path_parts[1])

        if not sentence_match or not run_match:
            continue

        expected_sentence_id = int(sentence_match.group(1))
        expected_run_id = int(run_match.group(1))

        # Process JSON files
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)

                try:
                    # Read JSON
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    # Check if metadata exists
                    if 'metadata' not in data:
                        continue

                    current_sentence_id = data['metadata'].get('sentence_id')
                    current_run_id = data['metadata'].get('run_id')

                    # Only update if they don't match
                    if current_sentence_id == expected_sentence_id and current_run_id == expected_run_id:
                        skipped_count += 1
                        continue

                    # Update metadata
                    data['metadata']['sentence_id'] = expected_sentence_id
                    data['metadata']['run_id'] = expected_run_id

                    # Write back
                    with open(json_path, 'w') as f:
                        json.dump(data, f, indent=2)

                    print(f"✓ {json_path}")
                    print(f"  Updated: sentence {current_sentence_id}→{expected_sentence_id}, run {current_run_id}→{expected_run_id}")
                    fixed_count += 1

                except Exception as e:
                    print(f"✗ Error processing {json_path}: {e}")

    return fixed_count, skipped_count

# Fix TTS and Waveform folders
print("Scanning outputs/results/TTS...")
tts_fixed, tts_skipped = fix_json_ids_in_folder(os.path.join(results_dir, "TTS"))

print(f"\nScanning outputs/results/Waveform...")
waveform_fixed, waveform_skipped = fix_json_ids_in_folder(os.path.join(results_dir, "Waveform"))

print(f"\n{'='*60}")
print(f"Total files fixed: {tts_fixed + waveform_fixed}")
print(f"  TTS: {tts_fixed}")
print(f"  Waveform: {waveform_fixed}")
print(f"\nTotal files skipped (already correct): {tts_skipped + waveform_skipped}")
print(f"  TTS: {tts_skipped}")
print(f"  Waveform: {waveform_skipped}")
