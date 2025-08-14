import os, sys, traceback
import torch, soundfile as sf
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_info()

# >>> PATHS: change if different on your PC
TTS_DIR = r"C:\Users\syeda\.cache\huggingface\hub\models--microsoft--speecht5_tts\snapshots\30fcde30f19b87502b8435427b5f5068e401d5f6"
VOC_DIR = r"C:\Users\syeda\.cache\huggingface\hub\models--microsoft--speecht5_hifigan\snapshots\bb6f429406e86a9992357a972c0698b22043307d"
SPEAKER = r".\assets\speaker_emb.pt"

TEXT = "আমি বাংলাদেশ থেকে এসেছি।"
OUT = "speecht5_test.wav"
SR = 16000

def log(m): print(f"[GEN] {m}", flush=True)
def die(m): print(f"[GEN][FATAL] {m}", flush=True); sys.exit(2)

try:
    if not os.path.isdir(TTS_DIR): die(f"TTS_DIR missing: {TTS_DIR}")
    if not os.path.isdir(VOC_DIR): die(f"VOC_DIR missing: {VOC_DIR}")
    if not os.path.exists(SPEAKER): die(f"Speaker emb missing: {SPEAKER}")

    from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan

    log("Loading processor…")
    proc = SpeechT5Processor.from_pretrained(TTS_DIR, local_files_only=True)

    log("Loading model…")
    model = SpeechT5ForTextToSpeech.from_pretrained(TTS_DIR, local_files_only=True).eval()

    log("Loading vocoder…")
    voc = SpeechT5HifiGan.from_pretrained(VOC_DIR, local_files_only=True)

    log("Loading speaker emb…")
    spk = torch.load(SPEAKER)
    if spk.ndim != 2 or spk.shape[1] != 512:
        die(f"Bad speaker emb shape {tuple(spk.shape)} (need [1,512])")

    log(f"Text: {TEXT}")
    inputs = proc(text=TEXT, return_tensors="pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    model = model.to(device)
    spk = spk.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    log("Generating…")
    with torch.no_grad():
        audio = model.generate_speech(inputs["input_ids"], spk, vocoder=voc)

    log(f"Saving -> {OUT}")
    sf.write(OUT, audio.detach().cpu().numpy(), SR)
    log("Done ✅")

except Exception:
    traceback.print_exc()
    die("exception above")
