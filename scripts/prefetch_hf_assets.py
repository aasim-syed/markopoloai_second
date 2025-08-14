from huggingface_hub import snapshot_download

print("Downloading SpeechT5 (tts)…")
tts_dir = snapshot_download("microsoft/speecht5_tts", repo_type="model")
print(" ->", tts_dir)

print("Downloading HifiGAN vocoder…")
voc_dir = snapshot_download("microsoft/speecht5_hifigan", repo_type="model")
print(" ->", voc_dir)

print("Downloading speaker x-vectors…")
spk_dir = snapshot_download("Matthijs/cmu-arctic-xvectors", repo_type="dataset")
print(" ->", spk_dir)
