#!/bin/bash
# Run adversarial TTS attack on Harvard sentences 1–100 (1 run per sentence)
# Run from project root: bash Scripts/Adversarial/TTS/RunScripts/run_tts_harvard.sh

START=1
END=100

python Scripts/Adversarial/generate_harvard_audios.py --start $START --end $END

python Scripts/Adversarial/TTS/adversarial_tts_harvard.py \
    --harvard_sentences_start $START \
    --harvard_sentences_end $END \
    --loop_count 1 \
    --num_generations 100 \
    --pop_size 200 \
    --batch_size 100 \
    --iv_scalar 0.5 \
    --size_per_phoneme 1 \
    --num_rms_candidates 1 \
    --objectives "PESQ=0.2, SET_OVERLAP=0.5" \
    --mode NOISE_UNTARGETED \
    --seed_target
