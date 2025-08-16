import os, sys, argparse, json, numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.features import extract_accent_features
from src.utils.audio import load_audio
from src.utils.logging import ensure_dir
from src.model.accent_classifier import train_random_forest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True, help='jsonl with audio,text,label in {BD,IN}')
    ap.add_argument('--sr', type=int, default=22050)
    ap.add_argument('--out', default='outputs/accent_clf/random_forest.joblib')
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out))

    X, y = [], []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Featurizing"):
            j = json.loads(line)
            if not os.path.exists(j['audio']): 
                continue
            ysig = load_audio(j['audio'], sr=args.sr)
            feats = extract_accent_features(ysig, args.sr)
            X.append([*feats.values()])
            y.append(1 if j.get('label','BD')=='BD' else 0)

    if len(X) < 20:
        raise RuntimeError("Not enough labeled items in manifest; need at least ~20.")

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    acc = train_random_forest(X, y, args.out)
    print(f"Accent classifier accuracy: {acc*100:.2f}% -> {args.out}")

if __name__ == '__main__':
    main()
