import argparse, json, numpy as np, torch, librosa
from ..model.bd_vits import BDVitsModel
from ..utils.audio import MelConfig, melspectrogram, load_audio
from ..utils.metrics import mel_spectral_distance, f0_correlation, spectral_convergence
from ..vocoder.griffinlim import VocoderConfig, mel_db_to_audio

def eval_one(model, text: str, ref_wav: str, sr: int):
    device = next(model.parameters()).device
    # predict mel
    tok = model.tokenize([text])
    tok = {k: v.to(device) for k, v in tok.items()}
    with torch.no_grad():
        mel_pred = model(tok['input_ids'], tok.get('attention_mask'))  # [1, n_mels, T]
    mel_pred = mel_pred.squeeze(0).detach().cpu().numpy()

    # ref mel + audio
    mel_cfg = MelConfig(sample_rate=sr)
    y_ref = load_audio(ref_wav, sr=sr)
    mel_ref = melspectrogram(y_ref, mel_cfg)

    # vocode pred to waveform for f0
    vcfg = VocoderConfig(sample_rate=sr)
    y_pred = mel_db_to_audio(mel_pred, vcfg)

    # metrics
    msd = mel_spectral_distance(mel_pred, mel_ref)
    sc = spectral_convergence(mel_pred, mel_ref)
    # robust f0 correlation
    f0_p = librosa.yin(y_pred, fmin=50, fmax=400, sr=sr)
    f0_r = librosa.yin(y_ref, fmin=50, fmax=400, sr=sr)
    L = min(len(f0_p), len(f0_r))
    f0c = 0.0 if L < 2 else float(np.corrcoef(f0_p[:L], f0_r[:L])[0,1])

    return {'mel_spectral_distance': float(msd), 'spectral_convergence': float(sc), 'f0_correlation': float(f0c)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='outputs/checkpoints/epoch_0.pt')
    ap.add_argument('--manifest', default='data/processed/valid_bd.jsonl')  # jsonl with {"audio","text","sr"}
    ap.add_argument('--out', default='outputs/eval_metrics.json')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BDVitsModel().to(device).eval()
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd['model'], strict=False)

    per = []
    with open(args.manifest, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            j = json.loads(line.strip())
            sr = int(j.get('sr', model.sample_rate))
            m = eval_one(model, j['text'], j['audio'], sr)
            m['idx'] = i
            per.append(m)

    out = {
        'avg_msd': float(np.mean([x['mel_spectral_distance'] for x in per])) if per else None,
        'avg_sc': float(np.mean([x['spectral_convergence'] for x in per])) if per else None,
        'avg_f0_corr': float(np.mean([x['f0_correlation'] for x in per])) if per else None,
        'per_item': per,
    }
    import os
    os.makedirs('outputs', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as w:
        json.dump(out, w, indent=2, ensure_ascii=False)
    print('Saved metrics to', args.out)

if __name__ == '__main__':
    main()
