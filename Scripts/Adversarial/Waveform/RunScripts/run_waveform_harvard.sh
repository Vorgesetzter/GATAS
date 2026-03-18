#!/bin/bash
# Run adversarial waveform attack on Harvard sentences 1–100 (1 run per sentence)
# Run from project root: bash Scripts/Adversarial/Waveform/RunScripts/run_waveform_harvard.sh

START=1
END=100

python Scripts/Adversarial/generate_harvard_audios.py --start $START --end $END

python Scripts/Adversarial/Waveform/adversarial_waveform_harvard.py \
    --sentence_start $START \
    --sentence_end $END \
    --loop_count 1 \
    --num_generations 100 \
    --pop_size 200 \
    --batch_size 100 \
    --noise_scale 0.05 \
    --objectives "PESQ=0.2, SET_OVERLAP=0.5" \
    --mode NOISE_UNTARGETED \
    --seed_target
