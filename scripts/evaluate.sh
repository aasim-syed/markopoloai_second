#!/usr/bin/env bash
set -euo pipefail
python src/eval/evaluate.py --checkpoint artifacts/checkpoints/best --data data/processed --report artifacts/eval/report.json