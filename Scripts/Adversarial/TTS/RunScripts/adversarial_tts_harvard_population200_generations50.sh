#!/bin/bash
# Hyperparameter comparison: pop_size=200, num_generations=50
# Run from project root: bash Scripts/Adversarial/TTS/RunScripts/adversarial_tts_harvard_population200_generations50.sh

START=1
END=50

python Scripts/Adversarial/generate_harvard_audios.py --start $START --end $END

python Scripts/Adversarial/TTS/adversarial_tts_harvard.py \
    --harvard_sentences_start $START \
    --harvard_sentences_end $END \
    --loop_count 1 \
    --num_generations 50 \
    --pop_size 200 \
    --batch_size 100 \
    --iv_scalar 0.5 \
    --size_per_phoneme 1 \
    --num_rms_candidates 1 \
    --objectives "PESQ=0.2, SET_OVERLAP=0.5" \
    --mode NOISE_UNTARGETED \
    --seed_target
