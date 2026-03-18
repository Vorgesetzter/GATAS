#!/bin/bash
# Run SMACK untargeted attack on Harvard sentences 1–100
# Run from project root: bash Scripts/Adversarial/SMACK/RunScripts/run_smack_harvard.sh

START=1
END=100

source ~/miniconda3/etc/profile.d/conda.sh

conda activate styletts2
python Scripts/Adversarial/generate_harvard_audios.py --start $START --end $END

conda activate smack
cd SMACK
python ../Scripts/Adversarial/SMACK/adversarial_smack_harvard.py --start $START --end $END
