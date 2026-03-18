import argparse
import os
import sys
import numpy as np
import librosa
import torch
import speech_recognition as sr
import soundfile as sf

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from Models.whisper import Whisper

# --- SHARED LOADING LOGIC (Same as Attack Script) ---
LOADED_MODELS = {}


def load_whisper():
    if "whisper" not in LOADED_MODELS:
        print("[*] Loading Whisper (tiny)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Whisper(device=device)
        LOADED_MODELS["whisper"] = model
    return LOADED_MODELS["whisper"]


def load_deepspeech(model_path, scorer_path):
    if "deepspeech" not in LOADED_MODELS:
        try:
            from deepspeech import Model
        except ImportError:
            print("[!] Error: 'deepspeech' library not installed.")
            sys.exit(1)

        if not model_path or not os.path.exists(model_path):
            print(f"[!] Error: DeepSpeech model file not found at: {model_path}")
            sys.exit(1)

        print(f"[*] Loading DeepSpeech model from {model_path}...")
        ds_model = Model(model_path)
        if scorer_path and os.path.exists(scorer_path):
            ds_model.enableExternalScorer(scorer_path)

        LOADED_MODELS["deepspeech"] = ds_model
    return LOADED_MODELS["deepspeech"]


def clean_text(text):
    if text is None: return ""
    return text.lower().replace('.', '').replace(',', '').replace('?', '').replace('!', '').strip()


def transcribe(audio_path, model_type, args=None):
    """
    Universal transcription router.
    """
    print(f"[*] Transcribing '{audio_path}' using [{model_type}]...")

    # 1. WHISPER
    if model_type == 'whisper':
        whisper_model = load_whisper()

        # Load audio at 16 kHz (Whisper.inference() expects 16 kHz)
        audio_array, _ = librosa.load(audio_path, sr=16000)
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)

        asr_texts, _ = whisper_model.inference(audio_tensor)
        asr_text = asr_texts[0] if isinstance(asr_texts, list) else asr_texts

        return clean_text(asr_text)

    # 2. DEEPSPEECH
    elif model_type == 'deepspeech':
        ds_model = load_deepspeech(args.ds_model, args.ds_scorer)

        # DeepSpeech expects 16kHz int16 audio
        audio_int16, fs = sf.read(audio_path, dtype='int16')
        if fs != 16000:
            audio_float, _ = librosa.load(audio_path, sr=16000)
            audio_int16 = (audio_float * 32767).astype(np.int16)

        return clean_text(ds_model.stt(audio_int16))

    # 3. CLOUD APIs (Google, Azure)
    else:
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)

        try:
            if model_type == 'google':
                return clean_text(r.recognize_google(audio_data))

            elif model_type == 'azure':
                if not args.azure_key or not args.azure_region:
                    print("[!] Error: Azure requires --azure_key and --azure_region")
                    sys.exit(1)
                return clean_text(r.recognize_azure(audio_data, key=args.azure_key, location=args.azure_region)[0])

        except sr.UnknownValueError:
            return "[UNINTELLIGIBLE] (Model heard nothing)"
        except sr.RequestError as e:
            return f"[API ERROR] {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Audio Transcription")

    parser.add_argument("-i", "--inputfile", default="outputs/best_mixed.wav", help="Path to input .wav file")
    parser.add_argument("-m", "--model", choices=['google', 'whisper', 'deepspeech', 'azure'], default='whisper', help="Model to use for verification")

    # Extra configs for offline/cloud models
    parser.add_argument("--ds_model", help="Path to deepspeech .pbmm model file")
    parser.add_argument("--ds_scorer", help="Path to deepspeech .scorer file")
    parser.add_argument("--azure_key", help="Microsoft Azure Speech API Key")
    parser.add_argument("--azure_region", help="Azure Region (e.g. eastus)")

    args = parser.parse_args()

    if not os.path.exists(args.inputfile):
        print(f"[!] Error: File '{args.inputfile}' does not exist.")
        sys.exit(1)

    result = transcribe(args.inputfile, args.model, args)

    print("-" * 50)
    print(f"OUTPUT: {result}")
    print("-" * 50)