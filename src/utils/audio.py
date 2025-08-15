from dataclasses import dataclass
import librosa, numpy as np
import torch

@dataclass
class MelConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: int = 0
    fmax: int = 8000

def load_audio(path: str, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def melspectrogram(y: np.ndarray, cfg: MelConfig) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        win_length=cfg.win_length, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def f0_contour(y: np.ndarray, sr: int) -> np.ndarray:
    f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
    return f0

def spectral_convergence(pred: np.ndarray, target: np.ndarray) -> float:
    num = np.linalg.norm(target - pred)
    den = np.linalg.norm(target) + 1e-8
    return float(num / den)