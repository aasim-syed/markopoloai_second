# Technical Report — Bangladeshi Bangla TTS Finetuning (VITS)

## 1. Objective
Finetune `bangla-speech-processing/bangla_tts_female` to produce authentic Bangladeshi Bangla.

## 2. Datasets
- OpenSLR 53 (Bengali)
- Mozilla Common Voice (bn)
- Bengali.AI
Preprocessing details, filters, and BD heuristics are documented in `src/data/preprocess.py`.

## 3. Model & Training
- Base: VITS (Transformers)
- Progressive unfreezing schedule: encoder/decoder/postnet
- Optimizer/scheduler: AdamW + linear warmup
- Mixed precision: Accelerate
- WandB experiment tracking

## 4. Accent Classification
- Features: MFCC, F0 stats, LPC formants, rhythm (tempogram), spectral stats
- Model: RandomForest
- Goal: >85% BD-vs-IN accuracy (requires good labels; see BD heuristics + manual review)

## 5. Evaluation
- MSD (mel dB distance)
- F0 correlation
- Spectral convergence
- Accent authenticity (classifier prob on generated audio)
- A/B listening tests

## 6. Results
| Metric | Value |
|-------:|------:|
| MSD (↓) | TBA |
| F0 corr (↑) | TBA |
| Spectral conv (↓) | TBA |
| Accent score (↑) | TBA |

Audio samples are in `artifacts/ab_test`.

## 7. Deployment
- TorchScript export (`export_model.py`)
- HF Hub push (`scripts/hf_push.py`)
- Gradio demo (`src/demo/app.py`)

## 8. Future Work
- Better BD lexicon & metadata-driven labeling
- Reranker for accent fidelity
- Phoneme-aware losses using explicit alignments