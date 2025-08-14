import os, sys, traceback
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_debug()

# >>> PUT YOUR EXACT LOCAL PATHS HERE <<<
TTS_DIR = r"C:\Users\syeda\.cache\huggingface\hub\models--microsoft--speecht5_tts\snapshots\30fcde30f19b87502b8435427b5f5068e401d5f6"
VOC_DIR = r"C:\Users\syeda\.cache\huggingface\hub\models--microsoft--speecht5_hifigan\snapshots\bb6f429406e86a9992357a972c0698b22043307d"
SPEAKER_PATH = r".\assets\speaker_emb.pt"

def die(msg): 
    print(f"[PROBE][FATAL] {msg}", flush=True)
    sys.exit(2)

def step(n, name):
    print(f"[PROBE] Step {n}: {name}", flush=True)

try:
    step(0, "sanity paths")
    if not os.path.isdir(TTS_DIR): die(f"TTS_DIR missing: {TTS_DIR}")
    if not os.path.isdir(VOC_DIR): die(f"VOC_DIR missing: {VOC_DIR}")
    if not os.path.exists(SPEAKER_PATH): die(f"Missing speaker emb: {SPEAKER_PATH}")

    step(1, "import modules")
    import torch, soundfile as sf
    from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan

    step(2, "load processor (local_files_only)")
    proc = SpeechT5Processor.from_pretrained(TTS_DIR, local_files_only=True)
    print("[PROBE] processor OK")

    step(3, "load model (local_files_only)")
    mdl = SpeechT5ForTextToSpeech.from_pretrained(TTS_DIR, local_files_only=True)
    print("[PROBE] model OK")

    step(4, "load vocoder (local_files_only)")
    voc = SpeechT5HifiGan.from_pretrained(VOC_DIR, local_files_only=True)
    print("[PROBE] vocoder OK")

    step(5, "load speaker embedding")
    spk = torch.load(SPEAKER_PATH)
    print("[PROBE] speaker emb shape:", getattr(spk, "shape", None))

    step(6, "tokenize and dry-run")
    inputs = proc(text="টেস্ট", return_tensors="pt")
    with torch.no_grad():
        _ = mdl.generate_speech(inputs["input_ids"], spk, vocoder=voc)
    print("[PROBE] generate OK")

    print("[PROBE] ALL GOOD ✅")
    sys.exit(0)

except Exception:
    traceback.print_exc()
    die("exception above")
