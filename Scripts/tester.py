from Models.styletts2 import StyleTTS2
import torch
import soundfile as sf

import os
os.chdir("..")

def main():
    tts = StyleTTS2()
    tts.load_models()
    tts.load_checkpoints()
    tts.sample_diffusion()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    text_1 = "I think the NFL is quite fun"
    text_2 = "Tejas is back in Munich"
    noise = torch.randn(1, 1, 256).to(device)

    token_1 = tts.preprocess_text(text_1)
    token_2 = tts.preprocess_text(text_2)

    # token_1, token_2 = addNumbersPattern(token_1, token_2, [16,4])
    # tokens = torch.cat([token_1, token_2], dim=0)

    audio_1 = tts.inference_on_token(token_1, noise)
    audio_2 = tts.inference_on_token(token_2, noise)
    audios = [audio_1, audio_2]
    for i, audio in enumerate(audios):
        path = os.path.join("testing", f"audio_{i}.wav")
        sf.write(path, audio.flatten(), samplerate=24000)

if __name__ == "__main__":
    main()