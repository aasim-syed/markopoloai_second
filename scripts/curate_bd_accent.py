"""Filter a manifest to BD-only using trained RF classifier."""
import os, sys, argparse, json
import numpy as np
import joblib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.features import extract_accent_features
from src.utils.audio import load_audio
from src.utils.logging import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--clf', required=True)
    ap.add_argument('--sr', type=int, default=22050)
    ap.add_argument('--thr', type=float, default=0.7)
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out))
    if not os.path.exists(args.clf):
        raise FileNotFoundError(f"Classifier not found: {args.clf}. Train it first.")

    clf = joblib.load(args.clf)

    kept = 0
    with open(args.inp, 'r', encoding='utf-8') as f, open(args.out, 'w', encoding='utf-8') as w:
        for line in f:
            j = json.loads(line)
            if not os.path.exists(j['audio']): 
                continue
            y = load_audio(j['audio'], sr=args.sr)
            feats = extract_accent_features(y, args.sr)
            X = np.array([list(feats.values())], dtype=float)
            proba = clf.predict_proba(X)[0]  # [p_IN, p_BD]
            p_bd = float(proba[1]) if len(proba) == 2 else float(proba[-1])
            if p_bd >= args.thr:
                j['accent_prob_bd'] = p_bd
                w.write(json.dumps(j, ensure_ascii=False) + "\n")
                kept += 1
    print(f"Kept {kept} BD items -> {args.out}")

if __name__ == '__main__':
    main()
