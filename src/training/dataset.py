import torch
import torchaudio
from src.common.utils import load_jsonl
from src.model.text_normalize import bd_text_normalize
from transformers import AutoTokenizer

class TTSDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, tokenizer_name="bangla-speech-processing/bangla_tts_female", sample_rate=22050):
        self.rows = list(load_jsonl(manifest_path))
        self.sr = sample_rate
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        wav, sr = torchaudio.load(row["audio"])
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        text = bd_text_normalize(row["text"])
        tok = self.tok(text, return_tensors="pt")
        return {
            "audio": wav.squeeze(0),  # [T]
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0)
        }

def collate(batch):
    # Very simple padder; for production use smarter padding/bucketing.
    import torch.nn.utils.rnn as rnn
    audios = [b["audio"] for b in batch]
    max_len = max(a.shape[-1] for a in audios)
    padded_audio = torch.stack([torch.nn.functional.pad(a, (0, max_len - a.shape[-1])) for a in audios])

    input_ids = rnn.pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0)
    attention_mask = rnn.pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    return {"audio": padded_audio, "input_ids": input_ids, "attention_mask": attention_mask}