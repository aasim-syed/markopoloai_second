"""Prepare OpenSLR-53 (Large Bengali ASR) -> jsonl manifest.

- Uses Hugging Face datasets: dataset id `openslr`, config `SLR53`
- Writes jsonl {audio, text, sr}; supports --limit for quick sampling
"""
import os, sys, json, argparse
from pathlib import Path
from datasets import load_dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.logging import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/openslr53_all.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.manifest))

    # Load SLR53 via HF Hub
    ds = load_dataset("openslr", "SLR53", split="train")
    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    written = 0
    with open(args.manifest, "w", encoding="utf-8") as w:
        for i in range(n):
            ex = ds[i]
            audio = ex.get("audio")
            text = ex.get("text", "")
            path, sr = None, None

            if isinstance(audio, dict):
                path = audio.get("path")
                sr = audio.get("sampling_rate", 16000)
            elif isinstance(audio, str):
                path = audio
                sr = 16000

            # If no on-disk path, dump the array to wav
            if not path or not os.path.exists(path):
                arr = ex.get("audio", {}).get("array", None) if isinstance(audio, dict) else None
                if arr is None:
                    continue
                import soundfile as sf
                out_dir = Path("data/raw/openslr53_cache")
                out_dir.mkdir(parents=True, exist_ok=True)
                path = str(out_dir / f"slr53_{i:07d}.wav")
                sr = ex["audio"].get("sampling_rate", 16000)
                sf.write(path, arr, sr)

            item = {"audio": path, "text": text, "sr": sr}
            w.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} items -> {args.manifest}")

if __name__ == "__main__":
    main()
