import argparse, os
from transformers import VitsModel, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--repo", type=str, required=True, help="e.g., username/bd-bangla-tts-female")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    model = VitsModel.from_pretrained(args.checkpoint)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model.push_to_hub(args.repo, private=args.private)
    tok.push_to_hub(args.repo, private=args.private)
    print("Pushed to HF:", args.repo)

if __name__ == "__main__":
    main()