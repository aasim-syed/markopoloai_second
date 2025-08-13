#!/usr/bin/env bash
set -euo pipefail
accelerate launch src/training/train_vits.py --config configs/train_config.json