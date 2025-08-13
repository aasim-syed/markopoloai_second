# src/data/preprocess.py

import argparse, os, json, random
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio  # still used for Bengali.AI parquet fallback
from src.common.utils import ensure_dir, save_jsonl
from src.model.text_normalize import bd_text_normalize

BD_LEXICON = {
    "आসি", "হইতেছে", "এইটা", "ওইটা", "কই", "কাম", "বস", "ঢাকা", "চট্টগ্রাম", "বরিশাল",
    "বাংলাদেশ", "টাঙ্গাইল", "গাড়ি", "আইজ", "আব্দুল", "কেমন আছেন"
}

def guess_accent(text: str) -> str:
    t = text.replace("়", "")  # nukta normalization
    for w in BD_LEXICON:
        if w in t:
            return "BD"
    return "UNK"

def _pick_text(ex):
    for k in ("text", "sentence", "transcription", "normalized_text"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None

def _safe_resample(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y, orig_sr
    try:
        import librosa
        y_rs = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
        return y_rs, target_sr
    except Exception:
        ratio = float(target_sr) / float(orig_sr)
        n = int(round(len(y) * ratio))
        if n <= 0:
            return y, orig_sr
        y_idx = np.linspace(0, len(y) - 1, num=n)
        y_rs = np.interp(y_idx, np.arange(len(y)), y.astype(np.float32))
        return y_rs.astype(np.float32), target_sr

def write_audio_from_path(path, out_wav, target_sr=22050):
    """Try soundfile; fallback to librosa. Returns True/False (never raises)."""
    if not path:
        return False
    try:
        y, sr = sf.read(path, always_2d=False)
        if y is None:
            return False
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y, sr = _safe_resample(y, sr, target_sr)
        sf.write(out_wav, y, sr)
        return True
    except Exception:
        try:
            import librosa
            y, sr = librosa.load(path, sr=target_sr, mono=True)
            sf.write(out_wav, y, sr)
            return True
        except Exception:
            return False

def write_audio_from_array(array, sr, out_wav, target_sr=22050):
    """Accept list or ndarray; resample if needed."""
    if array is None or sr is None:
        return False
    if not isinstance(array, np.ndarray):
        try:
            array = np.array(array, dtype=np.float32)
        except Exception:
            return False
    if array.ndim > 1:
        array = np.mean(array, axis=1)
    if sr != target_sr:
        array, sr = _safe_resample(array, sr, target_sr)
    try:
        sf.write(out_wav, array, sr)
        return True
    except Exception:
        return False

# ---------------------------
# COMMON VOICE (manual reader)
# ---------------------------
def ingest_common_voice(split, out_dir, cv_limit=None, target_sr=22050, debug=False):
    """
    Common Voice v17 Bengali (bn) local reader:
    - Downloads ONLY audio/bn/<split>/*.tar and transcript/bn/<split>.tsv(.gz)
    - Builds stem -> sentence map from the BN TSV
    - Extracts mp3/flac/ogg/etc from bn tar and writes 22.05k WAVs
    """
    import os, glob, tarfile
    import pandas as pd
    from huggingface_hub import snapshot_download
    import soundfile as sf
    import numpy as np

    # Normalize split to CV naming
    if split == "valid":
        split = "validated"
    if split not in ("train", "dev", "test", "validated"):
        raise ValueError("split must be one of: train, dev, test, validated")

    # 1) Snapshot ONLY what we need (bn audio + bn transcript for this split)
    snap_dir = snapshot_download(
        "mozilla-foundation/common_voice_17_0",
        repo_type="dataset",
        allow_patterns=[
            f"audio/bn/{split}/*",
            f"transcript/bn/{split}.tsv",
            f"transcript/bn/{split}.tsv.gz",
        ],
    )

    # 2) Find the BN audio tar(s)
    shard_pattern = os.path.join(snap_dir, "audio", "bn", split, "*.tar")
    shard_paths = sorted(glob.glob(shard_pattern))
    if not shard_paths:
        print(f"[CV] No local bn {split} tar files under {shard_pattern}")
        return []

    # 3) Load the BN transcript for this split
    tsv_path = None
    for cand in [
        os.path.join(snap_dir, "transcript", "bn", f"{split}.tsv"),
        os.path.join(snap_dir, "transcript", "bn", f"{split}.tsv.gz"),
    ]:
        if os.path.exists(cand):
            tsv_path = cand
            break
    if tsv_path is None:
        print(f"[CV] No BN transcript found for split '{split}' under transcript/bn/{split}.tsv(.gz)")
        return []

    # Read TSV (tab-separated). New pandas: use on_bad_lines='skip' for safety.
    df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8", dtype=str, on_bad_lines="skip", engine="python")

    # Determine filename and sentence columns
    # CV typically has 'path' (e.g. 'common_voice_bn_123.mp3' or 'clips/common_voice_bn_123.mp3') and 'sentence'
    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        # case-insensitive fallback
        low = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        return None

    path_col = find_col(["path", "clip", "filename", "file"])
    sent_col = find_col(["sentence", "text", "transcription"])
    if path_col is None or sent_col is None:
        print(f"[CV] Could not find path/sentence columns in BN TSV. Columns={list(df.columns)[:10]}")
        return []

    # Build stem -> sentence map (stem = filename without extension)
    text_map = {}
    for pth, sent in zip(df[path_col].fillna(""), df[sent_col].fillna("")):
        s = str(sent).strip()
        if not s:
            continue
        base = os.path.basename(str(pth))
        stem = os.path.splitext(base)[0]  # 'common_voice_bn_31564335'
        text_map[stem] = s

    if debug:
        print(f"[CV {split}] BN text_map size:", len(text_map))

    os.makedirs(out_dir, exist_ok=True)

    # helper: write wav from raw bytes (torchaudio preferred; fallback to librosa)
    def write_wav_from_bytes(raw_bytes, ext, dst_wav):
        tmp_path = dst_wav + ".tmp" + ext
        try:
            with open(tmp_path, "wb") as f:
                f.write(raw_bytes)
            try:
                import torchaudio
                wav, sr = torchaudio.load(tmp_path)  # [C, T]
                if wav.dim() == 2 and wav.size(0) > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != target_sr:
                    wav = torchaudio.functional.resample(wav, sr, target_sr)
                    sr = target_sr
                sf.write(dst_wav, wav.squeeze(0).detach().cpu().numpy().astype(np.float32), sr)
                os.remove(tmp_path)
                return True
            except Exception:
                import librosa
                y, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
                sf.write(dst_wav, y, sr)
                os.remove(tmp_path)
                return True
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return False

    rows = []
    kept = skipped = 0
    limit = int(cv_limit) if (cv_limit is not None) else None

    # debug counters
    no_text_match = decode_fail = 0
    printed = 0

    def process_shard(tar_path, start_idx):
        nonlocal kept, skipped, no_text_match, decode_fail, printed
        count = 0
        with tarfile.open(tar_path, "r:*") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                name = os.path.basename(m.name)
                if "." not in name:
                    continue
                stem, ext = name.rsplit(".", 1)
                ext = ext.lower()
                if ext not in ("mp3", "wav", "flac", "ogg", "m4a", "opus", "webm"):
                    continue

                if limit is not None and kept >= limit:
                    break

                # Try direct stem and base-only stem
                sent = text_map.get(stem)
                if not sent:
                    base_stem = os.path.splitext(os.path.basename(name))[0]
                    sent = text_map.get(base_stem)
                if not sent:
                    no_text_match += 1
                    skipped += 1
                    continue

                try:
                    with tf.extractfile(m) as f:
                        aud_bytes = f.read()
                    wav_path = os.path.join(out_dir, f"cv_{split}_{start_idx + count}.wav")
                    if not write_wav_from_bytes(aud_bytes, "." + ext, wav_path):
                        decode_fail += 1
                        skipped += 1
                        continue

                    text_norm = bd_text_normalize(sent)
                    label = guess_accent(text_norm)  # heuristic BD/UNK
                    rows.append({"audio": wav_path, "text": text_norm, "label": label})
                    kept += 1
                    count += 1

                    if debug and printed < 8:
                        print("DEBUG[CV] paired:", name, "->", text_norm[:60])
                        printed += 1

                except Exception:
                    decode_fail += 1
                    skipped += 1
                    continue
        return count

    start = 0
    for tar_path in shard_paths:
        if limit is not None and kept >= limit:
            break
        got = process_shard(tar_path, start)
        start += got
        if debug:
            print(f"[CV {split}] processed shard:", os.path.basename(tar_path), "(kept so far=", kept, ")")

    if debug:
        print(f"[CV {split}] skip reasons -> no_text_match={no_text_match}, decode_fail={decode_fail}")

    print(f"[CV {split}] kept={kept} skipped={skipped} (limit={cv_limit})")
    return rows


# ---------------------------
# Bengali.AI (features-only guard)
# ---------------------------
def ingest_bengali_ai(split, out_dir, target_sr=22050, bai_limit=None, debug=False):
    """
    Bengali-AI via local parquet snapshot (if present) else online; applies bai_limit.
    Handles audio from path or array (list/ndarray).
    Skips gracefully if the split only has features (input_features/labels) and no raw audio/text.
    """
    import glob
    from huggingface_hub import snapshot_download

    ds = None
    try:
        snap_dir = snapshot_download("thesven/bengali-ai-train-set-tiny", repo_type="dataset")
        pat = os.path.join(snap_dir, "**", f"{split}-*.parquet")
        files = glob.glob(pat, recursive=True)
        if files:
            from datasets import load_dataset as _ld
            ds = _ld("parquet", data_files={split: files}, split=split)
    except Exception:
        ds = None

    if ds is None:
        ds = load_dataset("thesven/bengali-ai-train-set-tiny", split=split)

    # If it doesn't have raw audio/text columns, skip
    cols = set(getattr(ds, "column_names", []))
    if not (("audio" in cols) or any(k in cols for k in ("sentence", "text", "transcription"))):
        print("[Bengali-AI] No raw audio/text columns (saw:", cols, ") — skipping.")
        return []

    if bai_limit is not None and len(ds) > bai_limit:
        ds = ds.select(range(bai_limit))

    if debug and len(ds) > 0:
        try:
            print("DEBUG[BAI]: first example keys:", list(ds[0].keys()))
            ex0 = ds[0]
            print("DEBUG[BAI]: text-like field:", _pick_text(ex0) or ex0.get("sentence"))
            if isinstance(ex0.get("audio"), dict):
                ak = list(ex0["audio"].keys())
            else:
                ak = type(ex0.get("audio")).__name__
            print("DEBUG[BAI]: audio field type/keys:", ak)
        except Exception:
            pass

    rows, kept, skipped = [], 0, 0
    total = len(ds)
    for i, ex in enumerate(ds):
        try:
            text_raw = _pick_text(ex) or ex.get("sentence")
            if not text_raw:
                skipped += 1
                continue
            text = bd_text_normalize(text_raw)

            wav_path = os.path.join(out_dir, f"bai_{split}_{i}.wav")
            aud = ex.get("audio")
            ok = False
            if isinstance(aud, dict):
                if aud.get("path"):
                    ok = write_audio_from_path(aud["path"], wav_path, target_sr=target_sr)
                if not ok and "array" in aud and "sampling_rate" in aud:
                    ok = write_audio_from_array(aud["array"], aud["sampling_rate"], wav_path, target_sr=target_sr)
            elif isinstance(aud, str):
                ok = write_audio_from_path(aud, wav_path, target_sr=target_sr)

            if not ok:
                skipped += 1
                continue

            rows.append({"audio": wav_path, "text": text, "label": guess_accent(text)})
            kept += 1

            if debug and ((i + 1) % 50 == 0 or (i + 1) == total):
                print(f"[Bengali-AI {split}] progress: {i+1}/{total}")
        except Exception:
            skipped += 1
            continue

    print(f"[Bengali-AI {split}] kept={kept} skipped={skipped} (limit={bai_limit})")
    return rows

def main():
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data")
    ap.add_argument("--out", type=str, default="data/processed")
    ap.add_argument("--use_common_voice", action="store_true")
    ap.add_argument("--use_bengali_ai", action="store_true")
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--cv_limit", type=int, default=None, help="Limit rows from each source")
    ap.add_argument("--target_sr", type=int, default=22050)
    ap.add_argument("--debug", action="store_true", help="Print sample and progress")
    args = ap.parse_args()

    ensure_dir(args.out)
    rows = []

    if args.use_common_voice:
        cv_out = os.path.join(args.out, "cv_audio")
        ensure_dir(cv_out)
        try:
            rows += ingest_common_voice("train", cv_out, cv_limit=args.cv_limit,
                                        target_sr=args.target_sr, debug=args.debug)
        except Exception as e:
            print("COMMON VOICE FAILED — SKIPPING. Reason:", e, file=sys.stderr)

    if args.use_bengali_ai:
        bai_out = os.path.join(args.out, "bai_audio")
        ensure_dir(bai_out)
        try:
            rows += ingest_bengali_ai("train", bai_out, target_sr=args.target_sr,
                                      bai_limit=args.cv_limit, debug=args.debug)
        except Exception as e:
            print("BENGALI.AI FAILED — SKIPPING. Reason:", e, file=sys.stderr)

    random.shuffle(rows)
    if rows:
        n_valid = max(1, int(len(rows) * args.valid_ratio))
        valid = rows[:n_valid]
        train = rows[n_valid:]
    else:
        valid, train = [], []

    save_jsonl(os.path.join(args.out, "train_manifest.jsonl"), train)
    save_jsonl(os.path.join(args.out, "valid_manifest.jsonl"), valid)

    stats = {"num_total": len(rows), "num_train": len(train), "num_valid": len(valid)}
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("Wrote manifests and stats:", stats)

if __name__ == "__main__":
    main()
