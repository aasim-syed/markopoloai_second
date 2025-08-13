import argparse, os, torch
from transformers import VitsModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out", type=str, default="artifacts/export/bd_bangla_tts_optimized.pt")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = VitsModel.from_pretrained(args.checkpoint)
    model.eval()

    # TorchScript export (placeholder — real VITS export may require custom wrapper)
    scripted = torch.jit.script(model)
    scripted.save(args.out)
    print("Saved optimized model:", args.out)

if __name__ == "__main__":
    main()