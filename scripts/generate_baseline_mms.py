import os, soundfile as sf
from transformers import pipeline

os.makedirs("baseline_out", exist_ok=True)
tts = pipeline("text-to-speech", model="facebook/mms-tts-ben")

texts = [
    "আমি বাংলাদেশ থেকে এসেছি।",
    "ঢাকা শহরে অনেক মানুষ বাস করে।",
    "আজকে আবহাওয়া খুব সুন্দর।"
]
for i, t in enumerate(texts):
    out = tts(t)
    sf.write(f"baseline_out/mms_base_{i}.wav", out["audio"], out["sampling_rate"])
    print("Wrote baseline_out/mms_base_%d.wav" % i)
