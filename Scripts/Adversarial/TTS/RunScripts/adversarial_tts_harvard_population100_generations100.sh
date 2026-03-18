#!/bin/bash
# Hyperparameter comparison: pop_size=100, num_generations=100
# Run from project root: bash Scripts/Adversarial/RunScripts/adversarial_tts_harvard_population100_generations100.sh

python Scripts/Adversarial/adversarial_tts_harvard.py \
    --harvard_sentences_start 1 \
    --harvard_sentences_end 50 \
    --loop_count 1 \
    --num_generations 100 \
    --pop_size 100 \
    --batch_size 100 \
    --iv_scalar 0.5 \
    --size_per_phoneme 1 \
    --num_rms_candidates 1 \
    --objectives "PESQ=0.2, SET_OVERLAP=0.5" \
    --mode NOISE_UNTARGETED \
    --seed_target
