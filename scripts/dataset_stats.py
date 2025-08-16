#!/usr/bin/env python
import argparse, json, os, librosa, numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--out_csv", default="docs/dataset_stats.csv")
    ap.add_argument("--out_fig", default="docs/figs/duration_hist.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    rows = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Analyzing")):
            j = json.loads(line)
            try:
                y, _ = librosa.load(j["audio"], sr=args.sr, mono=True)
                dur = len(y) / args.sr
            except Exception:
                dur = np.nan
            rows.append({"idx": i, "audio": j["audio"], "duration_s": dur, "text_len": len(j.get("text",""))})

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print("Wrote", args.out_csv)

    plt.figure()
    plt.hist(df["duration_s"].dropna(), bins=50)
    plt.xlabel("Duration (s)"); plt.ylabel("Count"); plt.title("Utterance Duration Histogram")
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=150)
    print("Wrote", args.out_fig)

if __name__ == "__main__":
    main()
