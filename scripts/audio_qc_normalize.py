"""QC normalize + (optional) resample, write new manifest."""
import os, json, argparse, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--out_dir', default='data/processed/qc_wav')
    ap.add_argument('--sr', type=int, default=22050)
    ap.add_argument('--max_len', type=float, default=15.0)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.inp, "r", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as w:
        for line in f:
            j = json.loads(line)
            in_path = j['audio']
            if not os.path.exists(in_path): 
                continue
            y, sr_in = librosa.load(in_path, sr=None, mono=True)
            if len(y) < sr_in * 0.2:  # drop ultra-short
                continue
            if len(y) > sr_in * args.max_len:
                y = y[: int(sr_in * args.max_len)]
            if sr_in != args.sr:
                y = librosa.resample(y, orig_sr=sr_in, target_sr=args.sr)
            m = np.max(np.abs(y))
            if m > 0: y = y / m
            out_path = os.path.join(args.out_dir, Path(in_path).stem + f"_{args.sr}.wav")
            sf.write(out_path, y, args.sr)
            j['audio'], j['sr'] = out_path, args.sr
            w.write(json.dumps(j, ensure_ascii=False) + "\n")
            written += 1
    print(f"QC wrote {written} -> {args.out}")

if __name__ == "__main__":
    main()
