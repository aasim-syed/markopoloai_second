"""Create a small labeled_mix.jsonl for accent RF training.

Default: derive from your local QC manifest (no external downloads).
Labeling modes:
- random  : assign ~50/50 BD/IN with fixed seed (fast, unblock pipeline)
- keyword : weak heuristic using BD/IN keyword lists in text
Optional: try Common Voice v17/16/15 if --use_common_voice is set.

Usage examples:
  python -m scripts.generate_labeled_mix --out data/processed/labeled_mix.jsonl
  python -m scripts.generate_labeled_mix --mode keyword --count 400
  python -m scripts.generate_labeled_mix --use_common_voice
"""
import os, sys, json, argparse, random

def load_local_manifest(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            if j.get("audio") and j.get("text"):
                items.append(j)
    return items

BD_KW = ["বাংলাদেশ","ঢাকা","চট্টগ্রাম","বরিশাল","খুলনা","টাকা","বাংলা","রাজশাহী","সিলেট"]
IN_KW = ["ভারত","কলকাতা","পশ্চিমবঙ্গ","রুপি","দিল্লি","মুম্বাই","বেঙ্গালুরু"]

def label_keyword(text):
    t = str(text)
    score_bd = sum(k in t for k in BD_KW)
    score_in = sum(k in t for k in IN_KW)
    if score_bd > score_in: return "BD"
    if score_in > score_bd: return "IN"
    return random.choice(["BD","IN"])

def try_load_cv(split):
    from datasets import load_dataset
    tried = [
        ("mozilla-foundation/common_voice_17_0", "bn"),
        ("mozilla-foundation/common_voice_16_1", "bn"),
        ("mozilla-foundation/common_voice_15_0", "bn"),
    ]
    last_err = None
    for ds_id, conf in tried:
        try:
            return load_dataset(ds_id, conf, split=split)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load Common Voice bn (v17/16/15). Last error: {last_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/labeled_mix.jsonl")
    ap.add_argument("--from_manifest", default="data/processed/openslr53_qc.jsonl")
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--mode", choices=["random","keyword"], default="random")
    ap.add_argument("--use_common_voice", action="store_true", help="Try CV v17/16/15 instead of local manifest")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    random.seed(42)

    records = []
    if args.use_common_voice:
        ds = try_load_cv("train[:500]")
        for ex in ds:
            path = ex.get("path")
            text = ex.get("sentence") or ex.get("sentence_original") or ""
            if not path or not text: continue
            # derive label from metadata if present, else heuristic/random
            meta = (ex.get("locale") or ex.get("accent") or "")
            mu = str(meta).upper()
            if "BD" in mu or "BANGLADESH" in mu:
                label = "BD"
            elif "IN" in mu or "INDIA" in mu:
                label = "IN"
            else:
                label = label_keyword(text) if args.mode=="keyword" else random.choice(["BD","IN"])
            records.append({"audio": path, "text": text, "label": label})
            if len(records) >= args.count: break
    else:
        src = load_local_manifest(args.from_manifest)
        random.shuffle(src)
        for j in src[:args.count]:
            text = j["text"]
            label = label_keyword(text) if args.mode=="keyword" else random.choice(["BD","IN"])
            records.append({"audio": j["audio"], "text": text, "label": label})

    with open(args.out, "w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote labeled file with {len(records)} items -> {args.out}")

if __name__ == "__main__":
    main()
