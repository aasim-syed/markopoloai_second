# Bangladeshi Bangla TTS Finetuning (VITS)

End-to-end scaffold to fine-tune the `bangla-speech-processing/bangla_tts_female` VITS model
for authentic Bangladeshi Bangla accent, including:
- Environment setup & baseline inference
- Dataset download & curation
- Accent classification (BD vs IN)
- Model adaptation hooks (BD phoneme normalization, custom loss)
- Training loop (progressive unfreezing, transfer learning)
- Evaluation (objective + accent authenticity via classifier)
- Optimization, export, and Gradio demo

## Quickstart

```bash
# 1) Create env and install deps
pip install -r requirements.txt

# 2) Baseline inference (Phase 1)
python src/model/baseline.py --text "আমি বাংলাদেশ থেকে এসেছি।" --out baseline.wav

# 3) Download datasets (Phase 2)
python src/data/download.py --target data

# 4) Preprocess + curate BD subset (Phase 2)
python src/data/preprocess.py --input data --out data/processed

# 5) Train BD vs IN accent classifier (Phase 3)
python src/features/train_accent_classifier.py --input data/processed --out artifacts/accent_clf

# 6) Fine-tune (Phase 4-5)
accelerate launch src/training/train_vits.py --config configs/train_config.json

# 7) Evaluate (Phase 6)
python src/eval/evaluate.py --checkpoint artifacts/checkpoints/best --data data/processed --report artifacts/eval/report.json

# 8) Export + Demo (Phase 7)
python src/training/export_model.py --checkpoint artifacts/checkpoints/best --out artifacts/export/bd_bangla_tts_optimized.pt
python src/demo/app.py
```

> Notes:
- This repository provides *working scaffolds* and hooks. You can plug in more sophisticated components
  (e.g., a stronger accent classifier or improved phoneme mapping) without changing the overall pipeline.
- Large-scale training requires a GPU; configure `accelerate` before training.

## Extras (fully wired)
- **WandB** logging enabled in `train_vits.py`.
- **Progressive unfreezing** schedule controlled in `configs/train_config.json`.
- **HF Hub push**: `python scripts/hf_push.py --checkpoint artifacts/checkpoints/best --repo <username>/bd-bangla-tts-female`
- **A/B samples**: `python scripts/ab_compare.py --finetuned artifacts/checkpoints/best`
- **Improved evaluation**: `python src/eval/evaluate.py --checkpoint artifacts/checkpoints/best --data data/processed --accent_clf artifacts/accent_clf/model.joblib`
