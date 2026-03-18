#!/bin/bash
# Run PGD untargeted attack on Harvard sentences 1–100
# Run from project root: bash Scripts/Adversarial/PGD/RunScripts/run_pgd_harvard.sh

source ~/miniconda3/etc/profile.d/conda.sh

conda activate styletts2
python Scripts/Adversarial/generate_harvard_audios.py --start 1 --end 100

conda activate whisper_attack
python Scripts/Adversarial/PGD/adversarial_pgd_harvard.py --start 1 --end 100
