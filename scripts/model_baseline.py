from transformers import VitsModel, AutoTokenizer
import torch
mid="bangla-speech-processing/bangla_tts_female"
print("Loading:", mid)
tok = AutoTokenizer.from_pretrained(mid)
model = VitsModel.from_pretrained(mid)
print("Config:", model.config)
print("Params (M):", round(sum(p.numel() for p in model.parameters())/1e6, 2))
print("CUDA:", torch.cuda.is_available())
