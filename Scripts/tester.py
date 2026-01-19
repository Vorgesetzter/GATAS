import torch

from Models.styletts2 import StyleTTS2
from Models.whisper import Whisper

from Objectives.FitnessObjective import FitnessObjective
from Trainer.EnvironmentLoader import EnvironmentLoader
from Datastructures.dataclass import ModelData, ObjectiveContext
from Datastructures.enum import AttackMode

import os
os.chdir("..")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    active_objectives = [FitnessObjective.WER_GT, FitnessObjective.PESQ]
    mode = AttackMode.TARGETED

    print("Loading Environment...")
    loader = EnvironmentLoader(device)

    print("Loading TTS Model (StyleTTS2)...")
    tts = StyleTTS2(device=device)

    print("Loading ASR Model (Whisper)...")
    asr = Whisper(device=device)

    text_1 = "I think the NFL is lame and boring"
    text_2 = "I think we aint a furl is lame and boring"

    noise = torch.randn(1, 1, 256).to(device)

    token_1 = tts.preprocess_text(text_1)
    token_2 = tts.preprocess_text(text_2)

    audio_1 = tts.inference_on_token(token_1, noise)
    audio_2 = tts.inference_on_token(token_2, noise)

    objectives = loader.initialize_objectives(
        active_objectives=active_objectives,
        model_data=ModelData(tts_model=tts, asr_model=asr),
        text_gt=text_1,
        text_target=text_2,
        mode=mode,
        audio_gt=audio_1,
    )

    asr_1 = asr.inference(audio_1)
    asr_2 = asr.inference(audio_2)

    print(f"ASR 1: {asr_1}")
    print(f"ASR 2: {asr_2}")

    # Create context for evaluation (testing audio_1)
    context = ObjectiveContext(
        audio_mixed_batch=audio_2,
        asr_texts=asr_2,
        interpolation_vectors=torch.zeros(1, 1),  # Dummy - not used by WER/PESQ
    )

    # Evaluate each objective
    print("\n=== Objective Scores ===")
    for obj_enum, objective in objectives.items():
        scores = objective.calculate_score(context)
        print(f"{obj_enum.name}: {scores}")

if __name__ == "__main__":
    main()