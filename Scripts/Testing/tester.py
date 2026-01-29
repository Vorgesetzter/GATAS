import torch

from Models.styletts2 import StyleTTS2
from Models.whisper import Whisper

from Objectives.FitnessObjective import FitnessObjective
from Trainer.EnvironmentLoader import EnvironmentLoader
from Datastructures.dataclass import ModelData, ObjectiveContext
from Datastructures.enum import AttackMode

import os
os.chdir("../..")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    active_objectives = [FitnessObjective.WHISPER_PROB]
    mode = AttackMode.TARGETED

    print("Loading Environment...")
    loader = EnvironmentLoader(device)

    print("Loading TTS Model (StyleTTS2)...")
    tts = StyleTTS2(device=device)

    print("Loading ASR Model (Whisper)...")
    asr = Whisper(device=device)

    text_gt = "I think the NFL is lame and boring"
    text_target = "This is a very different sentence"

    noise = torch.randn(1, 1, 256).to(device)

    token_gt = tts.preprocess_text(text_gt)
    token_target = tts.preprocess_text(text_target)

    audio_gt = tts.inference_on_token(token_gt, noise)
    audio_target = tts.inference_on_token(token_target, noise)
    audio_target = audio_gt

    objectives = loader.initialize_objectives(
        active_objectives=active_objectives,
        model_data=ModelData(tts_model=tts, asr_model=asr),
        text_gt=text_gt,
        text_target=text_target,
        mode=mode,
        audio_gt=audio_gt,
    )

    asr_gt, mel_batch_gt = asr.inference(audio_gt)
    asr_target, mel_batch_target = asr.inference(audio_target)

    print(f"ASR Ground-Truth: {asr_gt}")
    print(f"ASR Target: {asr_target}")

    # Create context for evaluation (testing audio_1)
    context = ObjectiveContext(
        audio_mixed_batch=audio_target,
        asr_texts=asr_target,
        interpolation_vectors=torch.zeros(1, 1),
        mel_batch=mel_batch_target
    )

    # Evaluate each objective
    print("\n=== Objective Scores ===")
    for obj_enum, objective in objectives.items():
        scores = objective.calculate_score(context)
        print(f"{obj_enum.name}: {scores}")

if __name__ == "__main__":
    main()