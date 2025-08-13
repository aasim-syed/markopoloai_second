import argparse, os
from datasets import load_dataset
from src.common.utils import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default="data")
    args = ap.parse_args()

    ensure_dir(args.target)

    # OpenSLR 53 (Bengali)
    try:
        openslr = load_dataset("openslr", "SLR53")
        openslr.save_to_disk(os.path.join(args.target, "openslr53"))
        print("Saved OpenSLR 53 to disk.")
    except Exception as e:
        print("OpenSLR 53 download failed (you may need to fetch manually):", e)

    # Common Voice
    try:
        cv = load_dataset("mozilla-foundation/common_voice_11_0", "bn")
        cv.save_to_disk(os.path.join(args.target, "common_voice_bn"))
        print("Saved Common Voice bn to disk.")
    except Exception as e:
        print("Common Voice download failed:", e)

    # Bengali.AI tiny
    try:
        bengali_ai = load_dataset("thesven/bengali-ai-train-set-tiny")
        bengali_ai.save_to_disk(os.path.join(args.target, "bengali_ai_tiny"))
        print("Saved Bengali.AI tiny to disk.")
    except Exception as e:
        print("Bengali.AI tiny download failed:", e)

if __name__ == "__main__":
    main()