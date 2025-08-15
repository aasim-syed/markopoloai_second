import argparse, json, os, numpy as np, tempfile, subprocess, shutil
import librosa, soundfile as sf
import torch

from ..utils.audio import MelConfig, melspectrogram, load_audio, f0_contour
from ..utils.metrics import mel_spectral_distance, f0_correlation, spectral_convergence
from ..model.bd_vits import BDVitsModel
from ..vocoder.griffinlim import VocoderConfig, mel_db_to_audio

# ---- Accent classifier (sklearn RandomForest) ----
from ..model.accent_classifier import predict_proba as rf_predict_proba
from ..data.features import extract_accent_features

def synth_to_mel(model: BDVitsModel, text: str, mel_cfg: MelConfig) -> np.ndarray:
    device = next(model.parameters()).device
    tok = model.tokenize([text])  # model.tokenize already uses padding+truncation in our BDVitsModel
    # If your BDVitsModel.tokenize didn't set truncation, you can replace with:
    # tok = model.tokenizer([text], return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        mel_pred, _ = model(tok['input_ids'].to(device), tok.get('attention_mask', None))
    return mel_pred.squeeze(0).detach().cpu().numpy()


def mel_to_wav(mel_db: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Invert mel with Griffin-Lim for downstream metrics/classifier."""
    vcfg = VocoderConfig(sample_rate=sr)
    return mel_db_to_audio(mel_db, vcfg)


def evaluate_pair(model: BDVitsModel, text: str, ref_audio_path: str, mel_cfg: MelConfig,
                  accent_clf_path: str = None):
    # Predict mel
    mel_pred = synth_to_mel(model, text, mel_cfg)

    # Reference mel/f0
    y_ref = load_audio(ref_audio_path, sr=mel_cfg.sample_rate)
    mel_ref = melspectrogram(y_ref, mel_cfg)
    f0_ref = f0_contour(y_ref, mel_cfg.sample_rate)

    # Pred f0 (approx via ref length; best: compute from pred wav)
    y_pred = mel_to_wav(mel_pred, sr=mel_cfg.sample_rate)
    f0_pred = f0_contour(y_pred, mel_cfg.sample_rate)

    metrics = {
        'mel_spectral_distance': mel_spectral_distance(mel_pred, mel_ref),
        'spectral_convergence': spectral_convergence(mel_pred, mel_ref),
        'f0_correlation': f0_correlation(f0_pred, f0_ref),
    }

    # Accent authenticity: P(BD) from RF classifier if provided
    if accent_clf_path and os.path.exists(accent_clf_path):
        feats = extract_accent_features(y_pred, mel_cfg.sample_rate)
        # sklearn expects 2D array
        import numpy as  _np
        X = _np.array([[*feats.values()]], dtype=float)
        proba = rf_predict_proba(accent_clf_path, X)[0]  # [P(IN), P(BD)]
        metrics['accent_score_bd'] = float(proba[1])
    else:
        metrics['accent_score_bd'] = None

    return metrics


# -------- Optional: phoneme accuracy via Montreal Forced Aligner (MFA) ----------
def _which(cmd):  # tiny shutil.which that also supports .bat/.exe on Windows
    return shutil.which(cmd)

def phoneme_accuracy_mfa(texts, wavs, lang='bengali', work_dir='outputs/mfa_eval'):
    """
    Requires MFA installed & language model 'bengali'. If MFA is missing, returns None.
    Steps:
      - write a corpus folder with wav + lab
      - run MFA align to produce TextGrid
      - (proxy) compute phone coverage ratio (#aligned phones / #expected phones)
    """
    mfa = _which('mfa')
    if not mfa:
        return None  # MFA not installed; skip

    os.makedirs(work_dir, exist_ok=True)
    corpus = os.path.join(work_dir, 'corpus')
    outdir = os.path.join(work_dir, 'aligned')
    os.makedirs(corpus, exist_ok=True)

    # Write wav & lab
    for i, (t, w) in enumerate(zip(texts, wavs)):
        base = os.path.join(corpus, f'utt_{i:03d}')
        # wav already exists; we symlink or copy
        dest_wav = base + '.wav'
        if os.name == 'nt':  # Windows: symlink may need admin; copy instead
            shutil.copyfile(w, dest_wav)
        else:
            if not os.path.exists(dest_wav):
                os.symlink(os.path.abspath(w), dest_wav)
        with open(base + '.lab', 'w', encoding='utf-8') as f:
            f.write(t)

    # Run MFA
    # mfa align <corpus_dir> <dictionary_or_lang> <acoustic_model> <output_dir>
    # For language 'bengali', MFA bundles model as 'bengali'
    subprocess.run([mfa, 'align', corpus, lang, lang, outdir, '--clean'], check=True)

    # Proxy metric: % of files that produced a non-empty TextGrid (phones aligned)
    grids = [f for f in os.listdir(outdir) if f.endswith('.TextGrid')]
    if not grids:
        return 0.0
    return float(len(grids)) / float(len(texts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True, help='jsonl with audio,text for evaluation')
    ap.add_argument('--ckpt', default='outputs/checkpoints/best.pt')
    ap.add_argument('--out', default='outputs/eval_metrics.json')
    ap.add_argument('--sr', type=int, default=22050)
    ap.add_argument('--accent_clf', default='outputs/accent_clf/random_forest.joblib',
                    help='path to trained RandomForest joblib for accent score')
    ap.add_argument('--compute_mfa', action='store_true', help='attempt phoneme accuracy with MFA')
    args = ap.parse_args()

    mel_cfg = MelConfig(sample_rate=args.sr)
    model = BDVitsModel()
    if args.ckpt and args.ckpt.endswith('.pt') and os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(sd['model'], strict=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    metrics = []
    texts = []
    pred_wavs = []
    with open(args.manifest, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            j = json.loads(line)
            m = evaluate_pair(model, j['text'], j['audio'], mel_cfg, accent_clf_path=args.accent_clf)
            m['idx'] = i
            metrics.append(m)
            # keep for MFA
            mel_pred = synth_to_mel(model, j['text'], mel_cfg)
            y_pred = mel_to_wav(mel_pred, sr=mel_cfg.sample_rate)
            temp_wav = os.path.join('outputs', 'eval_pred_wavs')
            os.makedirs(temp_wav, exist_ok=True)
            wav_path = os.path.join(temp_wav, f'pred_{i:03d}.wav')
            sf.write(wav_path, y_pred, mel_cfg.sample_rate)
            texts.append(j['text'])
            pred_wavs.append(wav_path)

    summary = {
        'avg_msd': float(np.mean([m['mel_spectral_distance'] for m in metrics])),
        'avg_sc': float(np.mean([m['spectral_convergence'] for m in metrics])),
        'avg_f0_corr': float(np.mean([m['f0_correlation'] for m in metrics])),
        'avg_accent_score_bd': float(np.mean([m['accent_score_bd'] for m in metrics if m['accent_score_bd'] is not None])) if any(m['accent_score_bd'] is not None for m in metrics) else None,
        'per_item': metrics
    }

    # Optional: MFA phoneme accuracy proxy
    if args.compute_mfa:
        pa = phoneme_accuracy_mfa(texts, pred_wavs, lang='bengali', work_dir='outputs/mfa_eval')
        summary['phoneme_accuracy_proxy_mfa'] = pa

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as w:
        json.dump(summary, w, indent=2, ensure_ascii=False)
    print('Saved metrics to', args.out)

if __name__ == '__main__':
    main()
