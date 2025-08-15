from typing import Dict
import numpy as np, librosa


def extract_accent_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
    # Simple stats
    feats = {
        **{f"mfcc_mean_{i}": float(mfcc[i].mean()) for i in range(mfcc.shape[0])},
        **{f"mfcc_std_{i}": float(mfcc[i].std()) for i in range(mfcc.shape[0])},
        "spec_centroid_mean": float(spec_centroid.mean()),
        "spec_centroid_std": float(spec_centroid.std()),
        "f0_mean": float(np.nanmean(f0)),
        "f0_std": float(np.nanstd(f0)),
    }
    return feats