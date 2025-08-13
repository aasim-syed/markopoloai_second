import argparse, os, json, random, soundfile as sf
from datasets import load_dataset, Audio
from src.common.utils import ensure_dir, save_jsonl
from src.model.text_normalize import bd_text_normalize

BD_LEXICON = {
    "আসি", "হইতেছে", "এইটা", "ওইটা", "কই", "কাম", "বস", "ঢাকা", "চট্টগ্রাম", "বরিশাল",
    "বাংলাদেশ", "টাঙ্গাইল", "গাড়ি", "আইজ", "আব্দুল", "কেমন আছেন"
}

def guess_accent(text: str) -> str:
    t = text.replace("়", "")
    for w in BD_LEXICON:
        if w in t:
            return "BD"
    return "UNK"

def write_audio(example, out_wav):
    y = example["audio"]["array"]
    s = example["audio"]["sampling_rate"]
    sf.write(out_wav, y, s)

def ingest_common_voice(split, out_dir):
    # No trust_remote_code (datasets>=3)
    # NOTE: You must be logged in and have access granted on the dataset page.
    ds = load_dataset("mozilla-foundation/common_voice_17_0", "bn", split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=22050))
    rows = []
    for i, ex in enumerate(ds):
        if ex.get("audio") is None or ex.get("sentence") is None:
            continue
        text = bd_text_normalize(ex["sentence"])
        accent = ex.get("accent", None)
        label = "BD" if accent and "Bangladesh" in str(accent) else guess_accent(text)
        wav_path = os.path.join(out_dir, f"cv_{split}_{i}.wav")
        write_audio(ex, wav_path)
        rows.append({"audio": wav_path, "text": text, "label": label})
    return rows

def ingest_bengali_ai(split, out_dir):
    ds = load_dataset("thesven/bengali-ai-train-set-tiny", split=split)
    rows = []
    for i, ex in enumerate(ds):
        if "audio" not in ex or "sentence" not in ex:
            continue
        text = bd_text_normalize(ex["sentence"])
        wav_path = os.path.join(out_dir, f"bai_{split}_{i}.wav")
        write_audio(ex, wav_path)
        rows.append({"audio": wav_path, "text": text, "label": guess_accent(text)})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/processed")
    ap.add_argument("--use_common_voice", action="store_true")
    ap.add_argument("--use_bengali_ai", action="store_true")
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    args = ap.parse_args()

    ensure_dir(args.out)
    rows = []

    if args.use_common_voice:
        cv_out = os.path.join(args.out, "cv_audio")
        ensure_dir(cv_out)
        try:
            rows += ingest_common_voice("train", cv_out)
        except Exception as e:
            print("COMMON VOICE FAILED — SKIPPING. Reason:", e)

    if args.use_bengali_ai:
        bai_out = os.path.join(args.out, "bai_audio")
        ensure_dir(bai_out)
        try:
            rows += ingest_bengali_ai("train", bai_out)
        except Exception as e:
            print("BENGALI.AI FAILED — SKIPPING. Reason:", e)

    random.shuffle(rows)
    n_valid = max(1, int(len(rows) * args.valid_ratio)) if rows else 1
    valid = rows[:n_valid]
    train = rows[n_valid:]

    save_jsonl(os.path.join(args.out, "train_manifest.jsonl"), train)
    save_jsonl(os.path.join(args.out, "valid_manifest.jsonl"), valid)

    stats = {"num_total": len(rows), "num_train": len(train), "num_valid": len(valid)}
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("Wrote manifests and stats:", stats)

if __name__ == "__main__":
    main()