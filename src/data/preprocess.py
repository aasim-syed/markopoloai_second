# src/data/preprocess.py

import argparse, os, json, random, glob, tarfile, shutil, sys
import numpy as np
import soundfile as sf

# use local helpers to avoid cross-package imports
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_jsonl(path: str, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

# BD specific
from ..utils.text import bd_text_normalize  # our project
# optional ML classifier
try:
    from ..models.accent_classifier import predict_proba as rf_predict_proba
    HAVE_RF = True
except Exception:
    HAVE_RF = False

BD_LEXICON = {
    "আসি", "হইতেছে", "এইটা", "ওইটা", "কই", "কাম", "বস", "ঢাকা", "চট্টগ্রাম", "বরিশাল",
    "বাংলাদেশ", "টাঙ্গাইল", "গাড়ি", "আইজ", "আব্দুল", "কেমন আছেন"
}

def guess_accent(text: str) -> str:
    # quick heuristic based on BD lexical cues
    t = text.replace("়", "")
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
# COMMON VOICE (local reader)
# ---------------------------
def ingest_common_voice(split, out_dir, cv_limit=None, target_sr=22050, debug=False):
    """
    Common Voice v17 Bengali (bn) local reader:
    - Downloads ONLY audio/bn/<split>/*.tar and transcript/bn/<split>.tsv(.gz)
    - Builds stem -> sentence map from the BN TSV
    - Extracts clips and writes 22.05k WAVs
    """
    import pandas as pd
    from huggingface_hub import snapshot_download

    # normalize split names
    if split == "valid":
        split = "validated"
    if split not in ("train", "dev", "test", "validated"):
        raise ValueError("split must be one of: train, dev, test, validated")

    snap_dir = snapshot_download(
        "mozilla-foundation/common_voice_17_0",
        repo_type="dataset",
        allow_patterns=[
            f"audio/bn/{split}/*",
            f"transcript/bn/{split}.tsv",
            f"transcript/bn/{split}.tsv.gz",
        ],
    )

    shard_paths = sorted(glob.glob(os.path.join(snap_dir, "audio", "bn", split, "*.tar")))
    if not shard_paths:
        print(f"[CV] No bn {split} tar files")
        return []

    # transcript
    tsv_path = None
    for cand in [
        os.path.join(snap_dir, "transcript", "bn", f"{split}.tsv"),
        os.path.join(snap_dir, "transcript", "bn", f"{split}.tsv.gz"),
    ]:
        if os.path.exists(cand):
            tsv_path = cand
            break
    if tsv_path is None:
        print(f"[CV] No BN transcript for split={split}")
        return []

    df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8", dtype=str, on_bad_lines="skip", engine="python")

    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        low = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        return None

    path_col = find_col(["path", "clip", "filename", "file"])
    sent_col = find_col(["sentence", "text", "transcription"])
    if path_col is None or sent_col is None:
        print(f"[CV] Could not find path/sentence columns in TSV; columns head={list(df.columns)[:10]}")
        return []

    # stem -> sentence
    text_map = {}
    for pth, sent in zip(df[path_col].fillna(""), df[sent_col].fillna("")):
        s = str(sent).strip()
        if not s:
            continue
        base = os.path.basename(str(pth))
        stem = os.path.splitext(base)[0]
        text_map[stem] = s

    ensure_dir(out_dir)

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

    def process_shard(tar_path, start_idx):
        nonlocal kept, skipped
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
                sent = text_map.get(stem)
                if not sent:
                    base_stem = os.path.splitext(os.path.basename(name))[0]
                    sent = text_map.get(base_stem)
                if not sent:
                    skipped += 1
                    continue
                try:
                    with tf.extractfile(m) as f:
                        aud_bytes = f.read()
                    wav_path = os.path.join(out_dir, f"cv_{split}_{start_idx + count}.wav")
                    if not write_wav_from_bytes(aud_bytes, "." + ext, wav_path):
                        skipped += 1
                        continue
                    text_norm = bd_text_normalize(sent)
                    label = guess_accent(text_norm)
                    rows.append({"audio": wav_path, "text": text_norm, "label": label})
                    kept += 1
                    count += 1
                except Exception:
                    skipped += 1
                    continue
        return count

    start = 0
    for tar_path in shard_paths:
        if limit is not None and kept >= limit:
            break
        got = process_shard(tar_path, start)
        start += got

    print(f"[CV {split}] kept={kept} skipped={skipped} (limit={cv_limit})")
    return rows

# ---------------------------
# Bengali.AI (best effort)
# ---------------------------
def ingest_bengali_ai(split, out_dir, target_sr=22050, bai_limit=None, debug=False):
    from datasets import load_dataset
    rows, kept, skipped = [], 0, 0
    ds = load_dataset("thesven/bengali-ai-train-set-tiny", split=split)

    for i, ex in enumerate(ds):
        try:
            text_raw = _pick_text(ex) or ex.get("label")
            if not text_raw:
                skipped += 1
                continue
            text = bd_text_normalize(str(text_raw))
            # No canonical audio in this tiny set; treat as text-only augmentation
            # You may skip or create TTS synthetic later; here we skip adding rows.
            continue
        except Exception:
            skipped += 1
            continue

    print(f"[Bengali-AI {split}] used as text augmentation only")
    return rows

# ---------------------------
# Filtering helpers
# ---------------------------
def filter_only_bd(rows, accent_clf_path=None, prob_threshold=0.5, target_sr=22050):
    """
    Keep BD rows only.
    - If classifier provided and available: compute P(BD) on waveform, keep if >= threshold.
    - Else: heuristic label == 'BD' from guess_accent().
    """
    if accent_clf_path and os.path.exists(accent_clf_path) and HAVE_RF:
        # ML filtering
        from ..data.features import extract_accent_features
        from ..models.accent_classifier import predict_proba as _predict
        kept = []
        for r in rows:
            try:
                y, sr = sf.read(r["audio"], always_2d=False)
                if isinstance(y, np.ndarray) and y.ndim > 1:
                    y = np.mean(y, axis=1)
                if sr != target_sr:
                    y, _ = _safe_resample(y, sr, target_sr)
                feats = extract_accent_features(y, target_sr)
                X = np.array([[*feats.values()]], dtype=float)
                p = _predict(accent_clf_path, X)[0][1]  # P(BD)
                if p >= prob_threshold:
                    r2 = dict(r)
                    r2["accent_score_bd"] = float(p)
                    kept.append(r2)
            except Exception:
                # if anything fails, fall back to heuristic
                if r.get("label") == "BD":
                    kept.append(r)
        return kept
    else:
        # heuristic
        return [r for r in rows if r.get("label") == "BD"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data")
    ap.add_argument("--out", type=str, default="data/processed")
    ap.add_argument("--use_common_voice", action="store_true")
    ap.add_argument("--use_bengali_ai", action="store_true")
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--cv_limit", type=int, default=None, help="Limit rows from each source")
    ap.add_argument("--target_sr", type=int, default=22050)
    ap.add_argument("--only_bd_accent", action="store_true", help="Filter to BD-accent only (heuristic or classifier)")
    ap.add_argument("--accent_clf", type=str, default=None, help="Path to RandomForest .joblib for BD/IN")
    ap.add_argument("--bd_threshold", type=float, default=0.55, help="P(BD) threshold when using classifier")
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

    # --- optional BD-only filtering ---
    if args.only_bd_accent and rows:
        rows = filter_only_bd(rows, accent_clf_path=args.accent_clf,
                              prob_threshold=args.bd_threshold, target_sr=args.target_sr)

    # split
    random.shuffle(rows)
    n_valid = max(1, int(len(rows) * args.valid_ratio)) if rows else 0
    valid = rows[:n_valid]
    train = rows[n_valid:]

    # filenames (match downstream expectation)
    if args.only_bd_accent:
        train_name, valid_name = "train_bd.jsonl", "valid_bd.jsonl"
    else:
        train_name, valid_name = "train_manifest.jsonl", "valid_manifest.jsonl"

    save_jsonl(os.path.join(args.out, train_name), train)
    save_jsonl(os.path.join(args.out, valid_name), valid)
    stats = {"num_total": len(rows), "num_train": len(train), "num_valid": len(valid),
             "only_bd": bool(args.only_bd_accent)}
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("Wrote manifests and stats:", stats)
    print("Train manifest:", os.path.join(args.out, train_name))
    print("Valid manifest:", os.path.join(args.out, valid_name))

if __name__ == "__main__":
    main()
