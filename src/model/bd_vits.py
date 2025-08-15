import torch, torch.nn as nn
from transformers import VitsModel, AutoTokenizer

class BDVitsModel(nn.Module):
    """
    Wraps HF VitsModel and adds optional adapters. Defaults to **facebook/mms-tts-ben**
    which returns **waveform** directly. For training code expecting mels, we derive
    a log-mel from the waveform on-the-fly.
    """
    def __init__(self, base_model_name: str = "facebook/mms-tts-ben", n_mels: int = 80, sample_rate: int = 22050):
        super().__init__()
        self.vits = VitsModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.sample_rate = getattr(self.vits.config, 'sampling_rate', sample_rate)
        # Minimal adapter kept for compatibility; not used unless fine-tuning hidden states
        d = getattr(self.vits.config, 'hidden_size', 256)
        self.adapter = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        from .accent_classifier import AccentDiscriminator
        self.disc = AccentDiscriminator(n_mels=n_mels)
        self.n_mels = n_mels

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        out = self.vits(input_ids=input_ids, attention_mask=attention_mask)
        # MMS returns waveform at out.waveform [B, T]
        waveform = getattr(out, 'waveform', None)
        if waveform is not None:
            # Convert to log-mel for training losses
            import librosa, numpy as np
            mels = []
            for b in range(waveform.size(0)):
                y = waveform[b].detach().cpu().numpy()
                mel = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=self.n_mels)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mels.append(torch.tensor(mel_db, dtype=torch.float32))
            mel = torch.stack(mels, dim=0)  # [B, n_mels, T]
        else:
            # Fallback: some VITS return spectrogram
            mel = getattr(out, 'spectrogram', None)
            if mel is None:
                hidden = out.last_hidden_state
                hidden = hidden + self.adapter(hidden)
                mel = hidden.transpose(1,2)
        logits_disc = self.disc(mel)
        return mel, logits_disc

    def tokenize(self, texts: list):
        return self.tokenizer(texts, return_tensors='pt', padding=True, truncation