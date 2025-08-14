import os, sys, traceback
import torch, soundfile as sf
from transformers import AutoTokenizer, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# --- config ---
SR = 16000
TEXT = "আমি বাংলাদেশ থেকে এসেছি।"
OUT_WAV = "speecht5_test.wav"
SPEAKER_PATH = "assets/speaker_emb.pt"

# >>> REPLACE with your exact local cache paths (same ones you printed) <<<
TTS_DIR = r"C:\Users\syeda\.cache\huggingface\hub\models--microsoft--speecht5_tts\snapshots\30fcde30f19b87502b8435427b5f5068e401d5f6"
VOC_DIR = r"C:\Users\syeda\.cache\huggingface\hub\models--microsoft--speecht5_hifigan\snapshots\bb6f429406e86a9992357a972c0698b22043307d"

def log(m): print(f"[GEN-NP] {m}", flush=True)
def die(m): print(f"[GEN-NP][FATAL] {m}", flush=True); sys.exit(2)

try:
    if not os.path.isdir(TTS_DIR): die(f"TTS_DIR missing: {TTS_DIR}")
    if not os.path.isdir(VOC_DIR): die(f"VOC_DIR missing: {VOC_DIR}")
    if not os.path.exists(SPEAKER_PATH): die(f"Missing speaker emb: {SPEAKER_PATH}")

    # 1) Tokenizer ONLY (no Processor)
    log("Loading tokenizer (local)…")
    tok = AutoTokenizer.from_pretrained(TTS_DIR, use_fast=False, local_files_only=True)

    # 2) Model + Vocoder
    log("Loading model (local)…")
    model = SpeechT5ForTextToSpeech.from_pretrained(TTS_DIR, local_files_only=True).eval()

    log("Loading vocoder (local)…")
    vocoder = SpeechT5HifiGan.from_pretrained(VOC_DIR, local_files_only=True)

    # 3) Speaker embedding
    spk = torch.load(SPEAKER_PATH)
    if not (isinstance(spk, torch.Tensor) and spk.shape == (1, 512)):
        die(f"Bad speaker emb shape {getattr(spk, 'shape', None)}; expected (1, 512)")

    # 4) Tokenize
    log(f"Tokenizing: {TEXT}")
    inputs = tok(TEXT, return_tensors="pt")

    # 5) Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")
    model = model.to(device)
    spk = spk.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 6) Generate
    log("Generating speech…")
    with torch.no_grad():
        audio = model.generate_speech(inputs["input_ids"], spk, vocoder=vocoder)

    # 7) Save
    log(f"Writing WAV -> {OUT_WAV}")
    sf.write(OUT_WAV, audio.detach().cpu().numpy(), SR)
    secs = round(len(audio) / SR, 2)
    log(f"Done ✅ ({secs}s)  {os.path.abspath(OUT_WAV)}")

except Exception:
    traceback.print_exc()
    die("exception above")
