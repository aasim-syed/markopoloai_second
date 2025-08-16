import torch, torch.nn as nn
from transformers import VitsModel, AutoTokenizer
from ..utils.audio import MelConfig
import numpy as np, librosa

class BDVitsModel(nn.Module):
    """
    Wrap MMS VITS (facebook/mms-tts-ben) and expose log-mel consistent with dataset params.
    """
    def __init__(self, base_model_name: str = "facebook/mms-tts-ben", n_mels: int = 80, sample_rate: int = 22050):
        super().__init__()
        self.vits = VitsModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.sample_rate = getattr(self.vits.config, 'sampling_rate', sample_rate)

        self.mel_cfg = MelConfig(
            sample_rate=self.sample_rate,
            n_fft=1024, hop_length=256, win_length=1024,
            n_mels=n_mels, fmin=0, fmax=8000
        )
        self.n_mels = n_mels

        d = getattr(self.vits.config, 'hidden_size', 256)
        self.adapter = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        from .accent_classifier import AccentDiscriminator
        self.disc = AccentDiscriminator(n_mels=n_mels)

    def _waveform_to_log_mel(self, waveform_1d: np.ndarray) -> torch.Tensor:
        mel = librosa.feature.melspectrogram(
            y=waveform_1d, sr=self.mel_cfg.sample_rate, n_fft=self.mel_cfg.n_fft,
            hop_length=self.mel_cfg.hop_length, win_length=self.mel_cfg.win_length,
            n_mels=self.mel_cfg.n_mels, fmin=self.mel_cfg.fmin, fmax=self.mel_cfg.fmax
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return torch.tensor(mel_db, dtype=torch.float32)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        out = self.vits(input_ids=input_ids, attention_mask=attention_mask)
        waveform = getattr(out, 'waveform', None)
        if waveform is not None:
            mels = []
            for b in range(waveform.size(0)):
                y = waveform[b].detach().cpu().numpy()
                mels.append(self._waveform_to_log_mel(y))
            mel = torch.stack(mels, dim=0)  # [B, n_mels, T]
        else:
            mel = getattr(out, 'spectrogram', None)
            if mel is None:
                hidden = out.last_hidden_state
                hidden = hidden + self.adapter(hidden)
                mel = hidden.transpose(1, 2)
        logits_disc = self.disc(mel)
        return mel, logits_disc

    def tokenize(self, texts: list):
        return self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
