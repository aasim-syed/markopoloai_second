import os, sys, traceback
import torch, soundfile as sf
import sentencepiece as spm
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan

SR = 16000
TEXT = "আমি বাংলাদেশ থেকে এসেছি।"
OUT_WAV = "speecht5_test.wav"
SPEAKER_PATH = "assets/speaker_emb.pt"

# >>> set to your exact local cache paths (you already printed these) <<<
TTS_DIR = r"C:\Users\syeda\.cache\huggingface\hub\models--microsoft--speecht5_tts\snapshots\30fcde30f19b87502b8435427b5f5068e401d5f6"
VOC_DIR = r"C:\Users\syeda\.cache\huggingface\hub\models--microsoft--speecht5_hifigan\snapshots\bb6f429406e86a9992357a972c0698b22043307d"

def log(m): print(f"[GEN-SPM] {m}", flush=True)
def die(m): print(f"[GEN-SPM][FATAL] {m}", flush=True); sys.exit(2)

try:
    if not os.path.isdir(TTS_DIR): die(f"TTS_DIR missing: {TTS_DIR}")
    if not os.path.isdir(VOC_DIR): die(f"VOC_DIR missing: {VOC_DIR}")
    if not os.path.exists(SPEAKER_PATH): die(f"Missing speaker emb: {SPEAKER_PATH}")

    spm_path = os.path.join(TTS_DIR, "spm_char.model")
    if not os.path.exists(spm_path):
        die(f"SentencePiece model not found: {spm_path}")

    # 1) SentencePiece encode to IDs
    log("Loading SentencePiece model…")
    sp = spm.SentencePieceProcessor(model_file=spm_path)
    ids = sp.encode(TEXT, out_type=int)
    if not ids:
        die("SentencePiece returned empty ids")
    input_ids = torch.tensor([ids], dtype=torch.long)  # shape [1, T]

    # 2) Model + Vocoder
    log("Loading TTS model (local)…")
    model = SpeechT5ForTextToSpeech.from_pretrained(TTS_DIR, local_files_only=True).eval()

    log("Loading vocoder (local)…")
    vocoder = SpeechT5HifiGan.from_pretrained(VOC_DIR, local_files_only=True)

    # 3) Speaker embedding
    log("Loading speaker embedding…")
    spk = torch.load(SPEAKER_PATH)
    if not (isinstance(spk, torch.Tensor) and spk.shape == (1, 512)):
        die(f"Bad speaker emb shape {getattr(spk, 'shape', None)}; expected (1, 512)")

    # 4) Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")
    model = model.to(device)
    vocoder = vocoder.to(device)
    spk = spk.to(device)
    input_ids = input_ids.to(device)

    # 5) Generate
    log(f"Generating speech for: {TEXT}")
    with torch.no_grad():
        audio = model.generate_speech(input_ids, spk, vocoder=vocoder)

    # 6) Save
    log(f"Writing WAV -> {OUT_WAV}")
    sf.write(OUT_WAV, audio.detach().cpu().numpy(), SR)
    log(f"Done ✅  {os.path.abspath(OUT_WAV)}")

except Exception:
    traceback.print_exc()
    die("exception above")
import os, sys, traceback
import torch, soundfile as sf
import sentencepiece as spm
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan

SR = 16000
TEXT = "আমি বাংলাদেশ থেকে এসেছি।"
OUT_WAV = "speecht5_test.wav"
SPEAKER_PATH = "assets/speaker_emb.pt"

# >>> set to your exact local cache paths (you already printed these) <<<
TTS_DIR = r"C:\Users\syeda\.cache\huggingface\hub\models--microsoft--speecht5_tts\snapshots\30fcde30f19b87502b8435427b5f5068e401d5f6"
VOC_DIR = r"C:\Users\syeda\.cache\huggingface\hub\models--microsoft--speecht5_hifigan\snapshots\bb6f429406e86a9992357a972c0698b22043307d"

def log(m): print(f"[GEN-SPM] {m}", flush=True)
def die(m): print(f"[GEN-SPM][FATAL] {m}", flush=True); sys.exit(2)

try:
    if not os.path.isdir(TTS_DIR): die(f"TTS_DIR missing: {TTS_DIR}")
    if not os.path.isdir(VOC_DIR): die(f"VOC_DIR missing: {VOC_DIR}")
    if not os.path.exists(SPEAKER_PATH): die(f"Missing speaker emb: {SPEAKER_PATH}")

    spm_path = os.path.join(TTS_DIR, "spm_char.model")
    if not os.path.exists(spm_path):
        die(f"SentencePiece model not found: {spm_path}")

    # 1) SentencePiece encode to IDs
    log("Loading SentencePiece model…")
    sp = spm.SentencePieceProcessor(model_file=spm_path)
    ids = sp.encode(TEXT, out_type=int)
    if not ids:
        die("SentencePiece returned empty ids")
    input_ids = torch.tensor([ids], dtype=torch.long)  # shape [1, T]

    # 2) Model + Vocoder
    log("Loading TTS model (local)…")
    model = SpeechT5ForTextToSpeech.from_pretrained(TTS_DIR, local_files_only=True).eval()

    log("Loading vocoder (local)…")
    vocoder = SpeechT5HifiGan.from_pretrained(VOC_DIR, local_files_only=True)

    # 3) Speaker embedding
    log("Loading speaker embedding…")
    spk = torch.load(SPEAKER_PATH)
    if not (isinstance(spk, torch.Tensor) and spk.shape == (1, 512)):
        die(f"Bad speaker emb shape {getattr(spk, 'shape', None)}; expected (1, 512)")

    # 4) Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")
    model = model.to(device)
    vocoder = vocoder.to(device)
    spk = spk.to(device)
    input_ids = input_ids.to(device)

    # 5) Generate
    log(f"Generating speech for: {TEXT}")
    with torch.no_grad():
        audio = model.generate_speech(input_ids, spk, vocoder=vocoder)

    # 6) Save
    log(f"Writing WAV -> {OUT_WAV}")
    sf.write(OUT_WAV, audio.detach().cpu().numpy(), SR)
    log(f"Done ✅  {os.path.abspath(OUT_WAV)}")

except Exception:
    traceback.print_exc()
    die("exception above")
