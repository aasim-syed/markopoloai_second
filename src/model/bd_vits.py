import torch
import torch.nn as nn
import torchaudio
from transformers import VitsModel, AutoTokenizer
from ..utils.audio import MelConfig

class BDVitsModel(nn.Module):
    """
    MMS Bengali wrapper that:
      - produces log-mel with *torch/torchaudio* (keeps autograd intact)
      - uses dataset-consistent mel params
      - no numpy/librosa inside forward()
    """
    def __init__(self, base_model_name: str = "facebook/mms-tts-ben",
                 n_mels: int = 80, sample_rate: int = 22050):
        super().__init__()
        self.vits = VitsModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.sample_rate = getattr(self.vits.config, "sampling_rate", sample_rate)

        self.mel_cfg = MelConfig(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=n_mels,
            fmin=0,
            fmax=8000,
        )
        self._mel_spec = None
        self._amp2db = None

        # Minimal adapter if we ever fall back to hidden states
        d = getattr(self.vits.config, "hidden_size", 256)
        self.adapter = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def _ensure_transforms(self, device: torch.device, dtype: torch.dtype):
        # Lazily create on the correct device/dtype so autograd + CUDA/CPU are happy
        if self._mel_spec is None:
            self._mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.mel_cfg.sample_rate,
                n_fft=self.mel_cfg.n_fft,
                hop_length=self.mel_cfg.hop_length,
                win_length=self.mel_cfg.win_length,
                f_min=float(self.mel_cfg.fmin),
                f_max=float(self.mel_cfg.fmax),
                n_mels=self.mel_cfg.n_mels,
                center=True,
                power=2.0,
                norm="slaney",
                mel_scale="htk",
            )
        if self._amp2db is None:
            self._amp2db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=None)

        # move to device/dtype if needed
        self._mel_spec = self._mel_spec.to(device=device, dtype=dtype)
        self._amp2db = self._amp2db.to(device=device, dtype=dtype)

    def _waveform_to_log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: [B, T] float tensor on correct device
        returns:  [B, n_mels, T_frames] log-mel (dB)
        """
        # Torchaudio expects [B, T]; mel_spec outputs [B, n_mels, frames]
        mel = self._mel_spec(waveform)
        mel_db = self._amp2db(mel)
        return mel_db

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        device = input_ids.device
        dtype = next(self.vits.parameters()).dtype
        self._ensure_transforms(device, dtype)

        out = self.vits(input_ids=input_ids, attention_mask=attention_mask)
        waveform = getattr(out, "waveform", None)  # [B, T] if MMS returns audio

        if waveform is not None:
            # Keep as torch Tensor (no numpy!) so grads flow
            # Some MMS models emit float32 on CPU; cast to model param dtype for safety
            wf = waveform.to(device=device, dtype=dtype)
            mel = self._waveform_to_log_mel(wf)  # [B, n_mels, T']
        else:
            # Fallback: derive a spectrogram-like rep from hidden states
            hidden = out.last_hidden_state  # [B, L, D]
            hidden = hidden + self.adapter(hidden)
            mel = hidden.transpose(1, 2).contiguous()  # [B, C(~n_mels), T]
        return mel

    def tokenize(self, texts: list):
        # Avoid the truncation warning: do NOT pass truncation=True for this model
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
