"""
Re-run Whisper at batch_size=100 on all runs where the stored set_overlap score
does not match what is recomputed from the stored asr_transcription.
Reports whether the batch=100 transcription aligns with the stored score.
"""
import re, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torchaudio
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from Models.whisper import Whisper

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def content_lemmas(text):
    clean = re.sub(r"[^\w\s]", "", text.lower())
    words = set(clean.split()) - STOPWORDS
    result = set()
    for w in words:
        for pos in ("a", "v", "n", "r"):
            lemma = LEMMATIZER.lemmatize(w, pos=pos)
            if lemma != w:
                result.add(lemma)
                break
        else:
            result.add(w)
    return result


def set_overlap(gt, asr):
    gt_l = content_lemmas(gt)
    return len(gt_l & content_lemmas(asr)) / len(gt_l) if gt_l else 1.0


def load_audio(path, target_sr=24000):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform  # (1, samples)


def collect_mismatches(roots):
    mismatches = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for sent in sorted(os.listdir(root)):
            sent_path = os.path.join(root, sent)
            if not sent.startswith("sentence_") or not os.path.isdir(sent_path):
                continue
            for run in sorted(os.listdir(sent_path)):
                run_path = os.path.join(sent_path, run)
                summary_path = os.path.join(run_path, "run_summary.json")
                wav_path = os.path.join(run_path, "best_mixed.wav")
                if not os.path.exists(summary_path) or not os.path.exists(wav_path):
                    continue
                with open(summary_path) as f:
                    s = json.load(f)
                gt_text = s["text_data"]["ground_truth_text"]
                asr_text = s["text_data"]["asr_transcription"]
                stored = s["success_metrics"]["fitness_scores"].get("SET_OVERLAP")
                if stored is None:
                    continue
                recomp = set_overlap(gt_text, asr_text)
                if abs(recomp - stored) > 0.05:
                    mismatches.append({
                        "path": run_path,
                        "wav": wav_path,
                        "gt": gt_text,
                        "stored_asr": asr_text,
                        "stored_score": stored,
                        "recomp_from_stored_asr": recomp,
                    })
    return mismatches


def main():
    roots = ["outputs/results/TTS", "outputs/results/Waveform"]
    mismatches = collect_mismatches(roots)
    print(f"Runs with score/transcription mismatch: {len(mismatches)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    asr_model = Whisper(device=device)

    BATCH_SIZE = 100
    correct = still_wrong = 0

    for m in mismatches:
        audio = load_audio(m["wav"]).to(device).expand(BATCH_SIZE, -1)
        texts, _ = asr_model.inference(audio)
        new_asr = texts[0]
        new_score = set_overlap(m["gt"], new_asr)
        matches = abs(new_score - m["stored_score"]) <= 0.05

        status = "OK         " if matches else "STILL_WRONG"
        if matches:
            correct += 1
        else:
            still_wrong += 1

        print(f"[{status}] {os.path.relpath(m['path'])}")
        print(f"  stored_score={m['stored_score']:.3f}  "
              f"recomp_old={m['recomp_from_stored_asr']:.3f}  "
              f"recomp_new={new_score:.3f}")
        print(f"  old ASR: {m['stored_asr']}")
        print(f"  new ASR: {new_asr}")

    print(f"\n{'='*60}")
    print(f"Now correct:   {correct}/{len(mismatches)}")
    print(f"Still wrong:   {still_wrong}/{len(mismatches)}")


if __name__ == "__main__":
    main()
