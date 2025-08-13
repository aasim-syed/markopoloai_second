import librosa, numpy as np

def _formants_lpc(y, sr, order=16, n_formants=4):
    # crude LPC-based formant estimator: returns first few peaks (Hz)
    y = librosa.util.normalize(y)
    a = librosa.lpc(y, order=order)
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0.01]  # positive frequency
    angs = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angs * (sr / (2*np.pi))
    freqs = np.sort(freqs)
    return freqs[:n_formants] if len(freqs) >= n_formants else np.pad(freqs, (0, n_formants-len(freqs)), constant_values=0.0)

def extract_accent_features(audio_path, sr_target=22050, n_mfcc=20):
    y, sr = librosa.load(audio_path, sr=sr_target)
    # energy
    rms = np.mean(librosa.feature.rms(y=y))
    # spectral
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    # pitch
    try:
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        f0_stats = np.array([np.nanmean(f0), np.nanstd(f0)])
    except Exception:
        f0_stats = np.array([0.0, 0.0])
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    # formants
    formants = _formants_lpc(y, sr)
    # rhythm via onset/tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
    tempo_stats = np.array([np.nanmean(tempo), np.nanstd(tempo)])
    # tempogram energy
    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    rhythm = np.array([np.mean(tg), np.std(tg)])

    return np.concatenate([[rms, spec_centroid], f0_stats, formants, tempo_stats, rhythm, mfcc_mean])