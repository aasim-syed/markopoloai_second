import os, argparse, json, numpy as np
import librosa
import torch
from src.utils.audio import MelConfig, melspectrogram, load_audio, f0_contour
from src.utils.metrics import mel_spectral_distance, f0_correlation, spectral_convergence
from src.model.bd_vits import BDVitsModel
from src.data.features import extract_accent_features  # used only if accent_clf provided

def evaluate_pair(model, text: str, ref_audio_path: str, mel_cfg: MelConfig, accent_clf=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    tok = model.tokenize([text])
    with torch.no_grad():
        mel_pred, _ = model(tok['input_ids'].to(device), tok.get('attention_mask', None))
    mel_pred = mel_pred.squeeze(0).detach().cpu().numpy()

    y = load_audio(ref_audio_path, sr=mel_cfg.sample_rate)
    mel_ref = melspectrogram(y, mel_cfg)
    f0_ref = f0_contour(y, mel_cfg.sample_rate)
    f0_pred = f0_ref[:len(f0_ref)]  # placeholder

    out = {
        'mel_spectral_distance': mel_spectral_distance(mel_pred, mel_ref),
        'spectral_convergence': spectral_convergence(mel_pred, mel_ref),
        'f0_correlation': f0_correlation(f0_pred, f0_ref),
    }

    if accent_clf is not None:
        import numpy as np
        feats = extract_accent_features(y, mel_cfg.sample_rate)
        X = np.array([list(feats.values())], dtype=float)
        proba = accent_clf.predict_proba(X)[0]
        out['accent_score_ref_bd'] = float(proba[1]) if len(proba) == 2 else float(proba[-1])

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--ckpt', default='outputs/checkpoints/best.pt')
    ap.add_argument('--out', default='outputs/eval_metrics.json')
    ap.add_argument('--sr', type=int, default=22050)
    ap.add_argument('--accent_clf', default=None)
    args = ap.parse_args()

    mel_cfg = MelConfig(sample_rate=args.sr)
    model = BDVitsModel()
    if args.ckpt and args.ckpt.endswith('.pt') and os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(sd['model'], strict=False)

    clf = None
    if args.accent_clf and os.path.exists(args.accent_clf):
        import joblib
        clf = joblib.load(args.accent_clf)

    metrics = []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            j = json.loads(line)
            m = evaluate_pair(model, j['text'], j['audio'], mel_cfg, accent_clf=clf)
            m['idx'] = i
            metrics.append(m)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as w:
        json.dump({
            'avg_msd': float(np.mean([m['mel_spectral_distance'] for m in metrics])),
            'avg_sc': float(np.mean([m['spectral_convergence'] for m in metrics])),
            'avg_f0_corr': float(np.mean([m['f0_correlation'] for m in metrics])),
            'per_item': metrics
        }, w, indent=2, ensure_ascii=False)
    print('Saved metrics to', args.out)

if __name__ == '__main__':
    main()
