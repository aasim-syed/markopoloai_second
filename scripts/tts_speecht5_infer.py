# scripts/tts_speecht5_infer.py
import os, sys, traceback
import torch, soundfile as sf
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan

SR = 16000
TEXT = "আমি বাংলাদেশ থেকে এসেছি।"
OUT_WAV = "finetuned_speecht5.wav"
SPEAKER_PATH = "assets/speaker_emb.pt"
FINETUNED_DIR = "runs/speecht5_bd/final"  # if you later fine-tune

# <<< REPLACE THESE WITH YOUR EXACT LOCAL CACHE PATHS >>>
TTS_DIR = r"C:\Users\syeda\.cache\huggingface\hub\models--microsoft--speecht5_tts\snapshots\30fcde30f19b87502b8435427b5f5068e401d5f6"
VOC_DIR = r"C:\Users\syeda\.cache\huggingface\hub\models--microsoft--speecht5_hifigan\snapshots\bb6f429406e86a9992357a972c0698b22043307d"

def log(m): print(f"[infer] {m}", flush=True)

def die(msg, code=1):
    print(f"[infer][FATAL] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)

def main():
    try:
        # Hard-fail if any path missing
        if not os.path.isdir(TTS_DIR): die(f"TTS_DIR not found: {TTS_DIR}")
        if not os.path.isdir(VOC_DIR): die(f"VOC_DIR not found: {VOC_DIR}")
        if not os.path.exists(SPEAKER_PATH): die(f"Missing {SPEAKER_PATH}. Create it first.")

        model_dir = FINETUNED_DIR if os.path.isdir(FINETUNED_DIR) else TTS_DIR
        log(f"Model dir  = {model_dir}")
        log(f"Processor  = {TTS_DIR}")
        log(f"Vocoder    = {VOC_DIR}")

        # 1) Processor
        log("Loading processor (local)…")
        processor = SpeechT5Processor.from_pretrained(TTS_DIR, local_files_only=True)

        # 2) Model + Vocoder
        log("Loading model (local)…")
        model = SpeechT5ForTextToSpeech.from_pretrained(model_dir, local_files_only=True).eval()

        log("Loading vocoder (local)…")
        vocoder = SpeechT5HifiGan.from_pretrained(VOC_DIR, local_files_only=True)

        # 3) Speaker embedding
        speaker_embeddings = torch.load(SPEAKER_PATH)
        if not (isinstance(speaker_embeddings, torch.Tensor) and speaker_embeddings.shape == (1, 512)):
            die(f"Bad speaker emb shape {getattr(speaker_embeddings, 'shape', None)}; expected (1, 512)")

        # 4) Inputs
        log(f"Tokenizing: {TEXT}")
        inputs = processor(text=TEXT, return_tensors="pt")

        # 5) Device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Using device: {device}")
        model = model.to(device)
        speaker_embeddings = speaker_embeddings.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 6) Generate
        log("Generating speech…")
        with torch.no_grad():
            audio = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

        # 7) Save
        log(f"Writing WAV -> {OUT_WAV}")
        sf.write(OUT_WAV, audio.detach().cpu().numpy(), SR)

        # 8) Done
        secs = round(len(audio) / SR, 2)
        log(f"Done ✅  ({secs}s)  File: {os.path.abspath(OUT_WAV)}")
        sys.exit(0)

    except Exception:
        traceback.print_exc()
        die("Unhandled exception")

if __name__ == "__main__":
    main()
