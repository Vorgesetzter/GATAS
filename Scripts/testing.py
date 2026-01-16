"""
Simple Objective Testing Script

Tests objectives by:
1. Generating audio from text using StyleTTS2
2. Creating a test context with GT, target, and test data
3. Running objectives and printing scores

Usage:
    conda activate styletts2
    python Scripts/testing.py
"""

import sys
import os

os.chdir("..")

import torch
from dataclasses import dataclass
from typing import List, Dict

from Models.styletts2 import StyleTTS2
from Datastructures.dataclass import ModelData, StepContext
from Datastructures.enum import FitnessObjective, AttackMode
from Objectives.ObjectiveManager import ObjectiveManager


@dataclass
class TestContext:
    """Simple context for testing objectives."""
    text_ground_truth: str
    text_target: str
    text_test: str

    audio_ground_truth: torch.Tensor
    audio_target: torch.Tensor
    audio_test: torch.Tensor


def create_step_context(ctx: TestContext, device: str) -> StepContext:
    """Create StepContext for evaluate_batch() using the test audio/text."""
    # Convert to tensor if numpy array
    audio = torch.as_tensor(ctx.audio_test, device=device)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    return StepContext(
        audio_mixed=audio,
        asr_text=[ctx.text_test],
        clean_text=[ctx.text_test],
        interpolation_vector=torch.zeros(1, 512, device=device),
        mel_batch=None
    )


def run_objectives(
    objectives: List[FitnessObjective],
    ctx: TestContext,
    model_data: ModelData,
    device: str
) -> Dict[FitnessObjective, float]:
    """Run objectives and return scores."""
    step_context = create_step_context(ctx, device)

    # Convert audio_gt to tensor
    audio_gt = torch.as_tensor(ctx.audio_ground_truth, device=device)

    manager = ObjectiveManager(
        model_data=model_data,
        device=device,
        text_gt=ctx.text_ground_truth,
        text_target=ctx.text_target,
        mode=AttackMode.TARGETED,
        audio_gt=audio_gt,
    )

    manager.initialize_objectives(objectives)
    scores = manager.evaluate_batch(step_context)

    # Extract single score from batch
    return {obj: scores[obj][0] for obj in objectives}


def main():
    print("=" * 60)
    print("OBJECTIVE TESTING")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # --- Define objectives to test ---
    objectives = [
        FitnessObjective.PESQ,
        FitnessObjective.WER_GT,
    ]

    # --- Define test texts ---
    text_ground_truth = "This is the ground truth text"
    text_target = "This is the target text"
    text_test = "This is the ground truth text"

    # --- Load StyleTTS2 and generate audio ---
    print("\n[Loading] StyleTTS2...")
    tts = StyleTTS2()
    tts.load_models()
    tts.load_checkpoints()
    tts.sample_diffusion()

    print("\n[Generating] Audio from text...")
    # Noise tensor for style sampling (same noise = same speaking style)
    noise = torch.randn(1, 1, 256).to(device)

    audio_ground_truth = tts.inference(text_ground_truth, noise).flatten()
    audio_target = tts.inference(text_target, noise).flatten()
    audio_test = tts.inference(text_test, noise).flatten()

    # --- Create test context ---
    ctx = TestContext(
        text_ground_truth=text_ground_truth,
        text_target=text_target,
        text_test=text_test,
        audio_ground_truth=audio_ground_truth,
        audio_target=audio_target,
        audio_test=audio_test,
    )

    # --- Create model data (TTS not needed for objectives, ASR loaded lazily) ---
    model_data = ModelData(tts_model=tts, asr_model=None)

    # --- Run objectives ---
    print("\n[Running] Objectives...")
    print("-" * 40)

    scores = run_objectives(objectives, ctx, model_data, device)

    for obj, score in scores.items():
        print(f"  {obj.name}: {score:.4f}")

    print("-" * 40)
    print("\nDone!")


if __name__ == "__main__":
    main()
