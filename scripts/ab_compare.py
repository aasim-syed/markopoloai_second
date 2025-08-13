import argparse, os, json, torch, librosa, soundfile as sf
from transformers import VitsModel, AutoTokenizer
from src.common.utils import load_jsonl, ensure_dir

def synth(model, tok, text):
    with torch.no_grad():
        y = model.generate(**tok(text, return_tensors="pt")).squeeze(0).cpu().numpy()
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=str, default="bangla-speech-processing/bangla_tts_female")
    ap.add_argument("--finetuned", type=str, required=True)
    ap.add_argument("--data", type=str, default="data/processed/valid_manifest.jsonl")
    ap.add_argument("--outdir", type=str, default="artifacts/ab_test")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--sr", type=int, default=22050)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    base_m = VitsModel.from_pretrained(args.baseline)
    base_t = AutoTokenizer.from_pretrained(args.baseline)
    fin_m = VitsModel.from_pretrained(args.finetuned)
    fin_t = AutoTokenizer.from_pretrained(args.finetuned)

    samples = list(load_jsonl(args.data))[:args.n]
    report = []
    for i, s in enumerate(samples):
        text = s["text"]
        y0 = synth(base_m, base_t, text)
        y1 = synth(fin_m, fin_t, text)
        sf.write(os.path.join(args.outdir, f"{i:02d}_baseline.wav"), y0, args.sr)
        sf.write(os.path.join(args.outdir, f"{i:02d}_finetuned.wav"), y1, args.sr)
        report.append({"idx": i, "text": text, "baseline": f"{i:02d}_baseline.wav", "finetuned": f"{i:02d}_finetuned.wav"})

    with open(os.path.join(args.outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Wrote A/B samples to", args.outdir)

if __name__ == "__main__":
    main()