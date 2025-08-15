import numpy as np

def _align_time(pred: np.ndarray, target: np.ndarray, mode: str = "min"):
    """
    Align two [n_mels, T] arrays along time.
    mode="min": crop both to the shorter length (fast & stable).
    mode="pad": zero-pad the shorter to the longer.
    """
    assert pred.ndim == 2 and target.ndim == 2, "expect [n_mels, T]"
    if mode == "pad":
        T = max(pred.shape[1], target.shape[1])
        def pad(m):
            if m.shape[1] == T: return m
            out = np.zeros((m.shape[0], T), dtype=m.dtype)
            out[:, :m.shape[1]] = m
            return out
        return pad(pred), pad(target)
    else:
        # default: crop to common length
        T = min(pred.shape[1], target.shape[1])
        return pred[:, :T], target[:, :T]

# Mel-Spectral Distance (RMS on log-mel diff)
def mel_spectral_distance(pred: np.ndarray, target: np.ndarray, time_align: str = "min") -> float:
    pred_a, tgt_a = _align_time(pred, target, mode=time_align)
    return float(np.sqrt(np.mean((pred_a - tgt_a) ** 2)))


def spectral_convergence(pred: np.ndarray, target: np.ndarray, time_align: str = "min") -> float:
    pred_a, tgt_a = _align_time(pred, target, mode=time_align)
    num = np.linalg.norm(tgt_a - pred_a)
    den = np.linalg.norm(tgt_a) + 1e-8
    return float(num / den)


def f0_correlation(pred_f0: np.ndarray, tgt_f0: np.ndarray) -> float:
    L = min(len(pred_f0), len(tgt_f0))
    if L < 2: 
        return 0.0
    a = pred_f0[:L]
    b = tgt_f0[:L]
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

__all__ = ["mel_spectral_distance", "f0_correlation", "spectral_convergence"]
