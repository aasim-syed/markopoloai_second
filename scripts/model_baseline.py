import os, json, math
import torch
from huggingface_hub import snapshot_download

REPO = "bangla-speech-processing/bangla_tts_female"

snap_dir = snapshot_download(REPO, repo_type="model")
cfg_path = os.path.join(snap_dir, "config.json")

print("Snapshot dir:", snap_dir)
print("Config path:", cfg_path)

# --- read config.json (Coqui-style) ---
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

def peek(d, k):
    v = d.get(k)
    if isinstance(v, dict): return list(v.keys())
    return v

print("\n=== CONFIG SUMMARY ===")
print("model:", peek(cfg, "model"))
print("audio:", peek(cfg, "audio"))
print("num_chars:", cfg.get("num_chars"))
print("use_phonemes:", cfg.get("use_phonemes"))
print("phoneme_language:", cfg.get("phoneme_language"))
print("symbols:", len(cfg.get("characters", {}).get("symbols", [])) if "characters" in cfg else None)

# --- load checkpoint and estimate parameter count ---
# Coqui VITS checkpoints are usually plain state_dicts or wrapped under certain keys.
ckpt_candidates = [
    "pytorch_model.pth", "best_model.pth", "model.pth", "checkpoint.pth"
]
ckpt_path = None
for name in ckpt_candidates:
    p = os.path.join(snap_dir, name)
    if os.path.exists(p):
        ckpt_path = p
        break

if ckpt_path is None:
    # fallback: pick the largest .pth in the folder
    pths = [os.path.join(snap_dir, f) for f in os.listdir(snap_dir) if f.endswith(".pth")]
    ckpt_path = max(pths, key=os.path.getsize) if pths else None

if ckpt_path is None:
    print("\nNo .pth checkpoint found in the snapshot.")
else:
    print("\nCheckpoint:", ckpt_path)
    sd = torch.load(ckpt_path, map_location="cpu")
    # try common wrappers
    if isinstance(sd, dict):
        for k in ["state_dict", "model", "generator", "tts_model", "module"]:
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    # count params
    total = 0
    for v in (sd.values() if isinstance(sd, dict) else []):
        if torch.is_tensor(v):
            total += v.numel()
    print("Approx. parameters (M):", round(total / 1e6, 2))
