import json
from dataclasses import dataclass
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from ..utils.audio import load_audio, melspectrogram, MelConfig
from ..utils.text import to_phonemes

@dataclass
class TTSItem:
    audio: str
    text: str
    sr: int

# src/data/dataset.py  (only the __init__ reading block shown)
class TTSDataset(Dataset):
    def __init__(self, manifest_path: str, mel_cfg: MelConfig, use_phonemes: bool = True):
        self.items: List[TTSItem] = []
        # NOTE: utf-8-sig makes BOM transparent on Windows
        with open(manifest_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                self.items.append(TTSItem(j['audio'], j['text'], j.get('sr', mel_cfg.sample_rate)))
        self.mel_cfg = mel_cfg
        self.use_phonemes = use_phonemes


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        y = load_audio(it.audio, sr=self.mel_cfg.sample_rate)
        mel = melspectrogram(y, self.mel_cfg)  # [n_mels, T]
        text = to_phonemes(it.text) if self.use_phonemes else it.text
        return {"text": text, "mel": torch.tensor(mel, dtype=torch.float32), "sr": self.mel_cfg.sample_rate}

class Collator:
    def __call__(self, batch: List[Dict]):
        lengths = [b["mel"].shape[1] for b in batch]
        max_T = max(lengths)
        n_mels = batch[0]["mel"].shape[0]
        mels = torch.zeros(len(batch), n_mels, max_T)
        texts = []
        for i, b in enumerate(batch):
            T = b["mel"].shape[1]
            mels[i, :, :T] = b["mel"]
            texts.append(b["text"])
        return {"texts": texts, "mels": mels, "mel_lengths": torch.tensor(lengths, dtype=torch.long)}
