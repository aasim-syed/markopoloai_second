import numpy as np
from .audio import spectral_convergence

# Mel-Spectral Distance (MSD) using L2 on log-mel

def mel_spectral_distance(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2) ** 0.5)


def f0_correlation(pred_f0: np.ndarray, tgt_f0: np.ndarray) -> float:
    # handle length mismatch by min length
    L = min(len(pred_f0), len(tgt_f0))
    if L < 2: return 0.0
    a = pred_f0[:L]
    b = tgt_f0[:L]
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

__all__ = ["mel_spectral_distance", "f0_correlation", "spectral_convergence"]