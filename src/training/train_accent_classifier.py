import argparse, json, numpy as np
from tqdm import tqdm
from ..data.features import extract_accent_features
from ..utils.audio import load_audio
from ..models.accent_classifier import train_random_forest

"""Train BD vs IN RandomForest classifier on a labeled manifest (jsonl with {'audio','text','label'})."""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True, help='jsonl with audio,text,label where label in {BD,IN}')
    ap.add_argument('--sr', type=int, default=22050)
    ap.add_argument('--out', default='outputs/accent_clf/random_forest.joblib')
    args = ap.parse_args()

    X, y = [], []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            j = json.loads(line)
            ysig = load_audio(j['audio'], sr=args.sr)
            feats = extract_accent_features(ysig, args.sr)
            X.append([*feats.values()])
            y.append(1 if j.get('label','BD')=='BD' else 0)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    acc = train_random_forest(X, y, args.out)
    print(f"Accent classifier accuracy: {acc*100:.2f}% -> {args.out}")

if __name__ == '__main__':
    main()