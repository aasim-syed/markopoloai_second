import os, torch
os.makedirs("assets", exist_ok=True)
# 512-dim x-vector; deterministic for repeatability
g = torch.Generator().manual_seed(7)
emb = torch.randn(1, 512, generator=g)   # shape [1,512]
torch.save(emb, "assets/speaker_emb.pt")
print("Saved assets/speaker_emb.pt with shape", tuple(emb.shape))
