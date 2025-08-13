import argparse, os, json, numpy as np, torch, librosa
from transformers import VitsModel, AutoTokenizer
from src.common.utils import load_jsonl
import joblib

def mel_spectral_distance_db(a_mel_db, b_mel_db):
    # mean absolute difference in dB space
    return float(np.mean(np.abs(a_mel_db - b_mel_db)))

def spectral_convergence(a_mel, b_mel):
    num = np.linalg.norm(a_mel - b_mel, ord='fro')
    den = np.linalg.norm(b_mel, ord='fro') + 1e-9
    return float(num / den)

def f0_contour(y, sr):
    try:
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        return f0
    except Exception:
        return np.array([])

def f0_correlation(y_pred, y_ref, sr):
    f0_p = f0_contour(y_pred, sr)
    f0_r = f0_contour(y_ref, sr)
    L = min(len(f0_p), len(f0_r))
    if L < 5:
        return 0.0
    return float(np.corrcoef(f0_p[:L], f0_r[:L])[0,1])

def audio_to_mel_db(y, sr=22050, n_mels=80):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(S + 1e-9)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data", type=str, default="data/processed")
    ap.add_argument("--report", type=str, default="artifacts/eval/report.json")
    ap.add_argument("--accent_clf", type=str, default="artifacts/accent_clf/model.joblib")
    ap.add_argument("--max_samples", type=int, default=20)
    ap.add_argument("--sample_rate", type=int, default=22050)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.report), exist_ok=True)

    model = VitsModel.from_pretrained(args.checkpoint)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)

    valid = list(load_jsonl(os.path.join(args.data, "valid_manifest.jsonl")))
    if not valid:
        raise RuntimeError("No validation samples.")

    try:
        clf = joblib.load(args.accent_clf)
    except Exception:
        clf = None

    msd, f0c, sc, accent_scores = [], [], [], []
    n = min(args.max_samples, len(valid))
    for row in valid[:n]:
        # load ref
        import soundfile as sf
        y_ref, sr_ref = sf.read(row["audio"])
        if sr_ref != args.sample_rate:
            y_ref = librosa.resample(y_ref, orig_sr=sr_ref, target_sr=args.sample_rate)

        # predict
        inputs = tok(row["text"], return_tensors="pt")
        with torch.no_grad():
            y_pred = model.generate(**inputs).squeeze(0).cpu().numpy()

        # metrics
        A = audio_to_mel_db(y_pred, args.sample_rate)
        B = audio_to_mel_db(y_ref, args.sample_rate)
        msd.append(mel_spectral_distance_db(A, B))
        sc.append(spectral_convergence(librosa.db_to_amplitude(A), librosa.db_to_amplitude(B)))
        f0c.append(f0_correlation(y_pred, y_ref, args.sample_rate))

        if clf is not None:
            # simple accent score: classifier prob that generated audio is BD
            from src.features.feat_extract import extract_accent_features
            feat = extract_accent_features(row["audio"])
            prob = clf.predict_proba([feat])[0][0]  # assume index 0 is BD
            accent_scores.append(float(prob))

    report = {
        "mel_spectral_distance_mean": float(np.mean(msd)) if msd else None,
        "f0_correlation_mean": float(np.mean(f0c)) if f0c else None,
        "spectral_convergence_mean": float(np.mean(sc)) if sc else None,
        "accent_score_mean": float(np.mean(accent_scores)) if accent_scores else None,
        "samples_evaluated": int(n),
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print("Wrote evaluation report to", args.report, report)

if __name__ == "__main__":
    main()