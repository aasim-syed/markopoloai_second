
from dataclasses import dataclass
import numpy as np
import librosa

@dataclass
class VocoderConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: int = 0
    fmax: int = 8000
    griffinlim_iters: int = 60


def mel_db_to_audio(mel_db: np.ndarray, cfg: VocoderConfig) -> np.ndarray:
    """
    mel_db: [n_mels, T] in dB scale
    returns: waveform np.ndarray float32 in [-1,1]
    """
    # Convert dB mel back to power mel
    mel_power = librosa.db_to_power(mel_db)
    # Invert mel to linear magnitude spectrogram
    S = librosa.feature.inverse.mel_to_stft(
        M=mel_power,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        power=1.0,
        fmin=cfg.fmin,
        fmax=cfg.fmax
    )
    # Reconstruct waveform with Griffin-Lim
    y = librosa.griffinlim(
        S,
        n_iter=cfg.griffinlim_iters,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length
    )
    # Normalize
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y.astype(np.float32)