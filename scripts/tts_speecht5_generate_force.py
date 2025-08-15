# scripts/tts_speecht5_minimal.py
import sys, os
from pathlib import Path

# ---- HARD FAIL EARLY ON MISSING DEPENDENCIES ----
try:
    import torch
except Exception as e:
    print("[FATAL] PyTorch import failed:", repr(e))
    sys.exit(1)

try:
    import sentencepiece as spm  # required by SpeechT5 tokenizer
except Exception as e:
    print("[FATAL] sentencepiece import failed:", repr(e))
    print("Fix: pip install sentencepiece==0.2.1  (in your current venv)")
    sys.exit(1)

try:
    import soundfile as sf
except Exception as e:
    print("[FATAL] soundfile import failed:", repr(e))
    print("Fix: pip install soundfile")
    sys.exit(1)

try:
    from transformers import (
        SpeechT5Processor,
        SpeechT5ForTextToSpeech,
        SpeechT5HifiGan,
    )
except Exception as e:
    print("[FATAL] transformers import failed:", repr(e))
    print("Fix: pip install 'transformers>=4.40'")
    sys.exit(1)

print(f"[ENV] torch={torch.__version__}  sentencepiece={spm.__version__}")

MODEL_ID = "microsoft/speecht5_tts"
VOCODER_ID = "microsoft/speecht5_hifigan"
OUT_WAV = "out_speecht5.wav"

# ---- SPEAKER EMBEDDING ----
spk_path = Path("assets/speaker_emb.pt")
if spk_path.exists():
    try:
        speaker_embeddings = torch.load(spk_path, map_location="cpu")
        if isinstance(speaker_embeddings, torch.Tensor):
            pass
        else:
            speaker_embeddings = torch.tensor(speaker_embeddings, dtype=torch.float32)
        if speaker_embeddings.ndim == 1:
            speaker_embeddings = speaker_embeddings.unsqueeze(0)
        print(f"[SPK] Using {spk_path} with shape {tuple(speaker_embeddings.shape)}")
    except Exception as e:
        print(f"[SPK] Failed to load {spk_path}: {e}. Using random fallback.")
        speaker_embeddings = torch.randn(1, 512)
else:
    print("[SPK] No assets/speaker_emb.pt found. Using random fallback (1,512).")
    speaker_embeddings = torch.randn(1, 512)

# ---- LOAD PROCESSOR / MODELS ----
print("[LOAD] Processor…")
processor = SpeechT5Processor.from_pretrained(MODEL_ID)

print("[LOAD] TTS model… (CPU)")
tts = SpeechT5ForTextToSpeech.from_pretrained(MODEL_ID)
tts.to("cpu")
tts.eval()

print("[LOAD] Vocoder…")
vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_ID)
vocoder.to("cpu")
vocoder.eval()

# ---- TEXT -> SPEECH ----
# Keep it short to avoid long CPU run.
bangla_text = "আজকে আবহাওয়া সুন্দর। ঢাকায় যানজট একটু বেশি।"

print("[RUN] Tokenize…")
inputs = processor(text=bangla_text, return_tensors="pt")

with torch.no_grad():
    print("[RUN] Generate acoustic features (SpeechT5)…")
    spectrogram = tts.generate_speech(
        input_ids=inputs["input_ids"],
        speaker_embeddings=speaker_embeddings,
        return_tensors=True,  # get mel as tensor
    )

    # Some versions return dict with "spectrogram"; handle both forms
    if isinstance(spectrogram, dict) and "spectrogram" in spectrogram:
        mel = spectrogram["spectrogram"]
    else:
        # older/newer API returns tensor directly
        mel = spectrogram

    if not isinstance(mel, torch.Tensor):
        raise RuntimeError("Unexpected spectrogram type, expected torch.Tensor")

    print("[RUN] Vocoder (HiFiGAN)…")
    audio = vocoder(mel)
    # audio: (batch, time) float tensor in [-1, 1]
    if audio.ndim == 2:
        audio = audio[0]  # take first sample

    audio_np = audio.cpu().numpy()

print(f"[SAVE] Writing {OUT_WAV} (16kHz)…")
sf.write(OUT_WAV, audio_np, 16000)
print("[DONE] Wrote", OUT_WAV)
