#!/bin/bash
# Setup script for the whisper_attack conda environment.
#
# robust-speech, speechbrain, and pyaudlib declare torch<=1.11 as a dependency,
# which conflicts with torch==2.8.0. They are installed separately with --no-deps
# after torch is already in place.
#
# Usage (from project root):
#   bash setup_whisper_attack_env.sh

set -e

echo "Creating whisper_attack conda environment from whisper_attack.yml..."
conda env create -f whisper_attack.yml

echo "Installing GitHub packages without dependency resolution..."
conda run -n whisper_attack pip install --no-deps \
    git+https://github.com/RaphaelOlivier/pyaudlib.git \
    git+https://github.com/RaphaelOlivier/robust_speech.git \
    git+https://github.com/RaphaelOlivier/speechbrain.git

echo "Done. Activate with: conda activate whisper_attack"
