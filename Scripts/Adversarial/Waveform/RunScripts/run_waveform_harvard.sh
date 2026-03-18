#!/bin/bash
# Run adversarial waveform attack on Harvard sentences 1–100 (1 run per sentence)
# Run from project root: bash Scripts/Adversarial/RunScripts/run_waveform_harvard.sh

python Scripts/Adversarial/adversarial_waveform_harvard.py \
    --sentence_start 1 \
    --sentence_end 100 \
    --loop_count 1 \
    --num_generations 100 \
    --pop_size 200 \
    --batch_size 100 \
    --noise_scale 0.05 \
    --objectives "PESQ=0.2, SET_OVERLAP=0.5" \
    --mode NOISE_UNTARGETED \
    --seed_target
