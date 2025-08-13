import argparse, os, json, joblib, numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.common.utils import load_jsonl, ensure_dir
from src.features.feat_extract import extract_accent_features

# For demo, uses manifest labels if present; otherwise creates dummy labels.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/processed")
    ap.add_argument("--out", type=str, default="artifacts/accent_clf")
    args = ap.parse_args()

    ensure_dir(args.out)

    manifest = list(load_jsonl(os.path.join(args.input, "train_manifest.jsonl")))
    if not manifest:
        raise RuntimeError("No training data found. Preprocess first.")

    X, y = [], []
    for row in tqdm(manifest, desc="Extracting features"):
        feat = extract_accent_features(row["audio"])
        X.append(feat)
        # Use existing label or assign dummy
        label = row.get("label", "BD")
        if label not in ("BD", "IN"):
            label = "BD" if np.random.rand() > 0.5 else "IN"
        y.append(0 if label == "BD" else 1)

    X = np.stack(X, axis=0)
    y = np.array(y)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)

    # Naive split for quick metric (use proper CV in practice)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)

    joblib.dump(clf, os.path.join(args.out, "model.joblib"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"train_acc_proxy": float(acc)}, f, indent=2)

    print(f"Saved accent classifier to {args.out}, proxy-acc={acc:.3f}")

if __name__ == "__main__":
    main()