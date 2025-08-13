import argparse, torchaudio, torch
from transformers import VitsModel, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="আমি বাংলাদেশ থেকে এসেছি।")
    parser.add_argument("--out", type=str, default="baseline.wav")
    args = parser.parse_args()

    model = VitsModel.from_pretrained("bangla-speech-processing/bangla_tts_female")
    tok = AutoTokenizer.from_pretrained("bangla-speech-processing/bangla_tts_female")

    inputs = tok(args.text, return_tensors="pt")
    with torch.no_grad():
        audio = model.generate(**inputs)  # shape: [1, T]
    audio = audio.squeeze(0).cpu()
    torchaudio.save(args.out, audio.unsqueeze(0), 22050)
    print(f"Saved baseline audio to {args.out}")

if __name__ == "__main__":
    main()