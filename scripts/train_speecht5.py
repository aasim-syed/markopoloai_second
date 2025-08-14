import os, json, math, random, torch, librosa, soundfile as sf
from dataclasses import dataclass
from typing import Dict, List, Any
from datasets import Dataset
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
    TrainingArguments,
    Trainer,
)

# -------- settings --------
TRAIN_MAN = "data/processed/train_manifest.jsonl"
VALID_MAN = "data/processed/valid_manifest.jsonl"
MODEL_ID   = "microsoft/speecht5_tts"        # acoustic model
VOCODER_ID = "microsoft/speecht5_hifigan"    # used only for inference
SR = 22050

# -------- data --------
def read_manifest(path):
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if os.path.exists(r["audio"]) and isinstance(r["text"], str) and r["text"].strip():
                rows.append(r)
    return rows

train_rows = read_manifest(TRAIN_MAN)
valid_rows = read_manifest(VALID_MAN)

# small sanity limit for first run
random.shuffle(train_rows)
random.shuffle(valid_rows)
if len(train_rows) > 5000: train_rows = train_rows[:5000]
if len(valid_rows) > 500: valid_rows = valid_rows[:500]

train_ds = Dataset.from_list(train_rows)
valid_ds = Dataset.from_list(valid_rows)

# -------- processor & model --------
processor = SpeechT5Processor.from_pretrained(MODEL_ID)
model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_ID)

# -------- featurization --------
def load_audio(path, sr=SR):
    y, s = librosa.load(path, sr=sr, mono=True)
    return y, sr

def preprocess(batch):
    text = batch["text"]
    input_ids = processor.tokenizer(text, return_tensors="pt").input_ids[0]
    y, _ = load_audio(batch["audio"], SR)
    # processor expects raw audio -> extracts acoustic targets internally
    # we pass speech as "labels" via processor feature extractor
    targets = processor.feature_extractor(y, sampling_rate=SR, return_tensors="pt")
    batch["input_ids"] = input_ids
    batch["labels"] = targets["input_values"][0]  # raw waveform as labels (model learns acoustic tokens)
    return batch

train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names, desc="preprocess-train")
valid_ds = valid_ds.map(preprocess, remove_columns=valid_ds.column_names, desc="preprocess-valid")

# -------- data collator --------
@dataclass
class Collator:
    processor: Any
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        batch_in = processor.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
        # pad raw waveforms to max length in batch
        maxlen = max(l.shape[-1] for l in labels)
        lab = torch.zeros(len(labels), maxlen, dtype=torch.float32)
        att = torch.zeros(len(labels), maxlen, dtype=torch.long)
        for i, w in enumerate(labels):
            L = w.shape[-1]
            lab[i, :L] = w
            att[i, :L] = 1
        batch = {
            "input_ids": batch_in["input_ids"],
            "labels": lab,
            "labels_attention_mask": att,
        }
        return batch

collate_fn = Collator(processor)

# -------- training --------
bsz = 4
args = TrainingArguments(
    output_dir="runs/speecht5_bd",
    learning_rate=1e-4,
    per_device_train_batch_size=bsz,
    per_device_eval_batch_size=bsz,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    fp16=torch.cuda.is_available(),
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=3,
    report_to=["none"],  # or ["wandb"]
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=collate_fn,
)

trainer.train()
trainer.save_model("runs/speecht5_bd/final")

print("Training done. To synthesize a sample:")
print("  python scripts/tts_speecht5_infer.py")
