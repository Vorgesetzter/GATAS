#!/bin/bash
# Run SMACK untargeted attack on Harvard sentences 1–100
# Run from project root: bash Scripts/Adversarial/SMACK/RunScripts/run_smack_harvard.sh

source ~/miniconda3/etc/profile.d/conda.sh

conda activate styletts2
python Scripts/Adversarial/generate_harvard_audios.py --start 1 --end 100

conda activate smack
cd SMACK
python ../Scripts/Adversarial/SMACK/adversarial_smack_harvard.py --start 1 --end 100
