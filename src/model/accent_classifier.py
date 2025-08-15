"""Two parts: (1) Offline RandomForest accent classifier (sklearn).
(2) A tiny PyTorch discriminator used as auxiliary loss during TTS training.
"""
from typing import List
import numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn

class AccentDiscriminator(nn.Module):
    def __init__(self, n_mels: int = 80, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, 2)  # [BD, IN]
        )
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.net(mel)

# ---- sklearn pipeline helpers ----

def train_random_forest(X: np.ndarray, y: np.ndarray, out_path: str):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    joblib.dump(clf, out_path)
    return acc


def predict_proba(path: str, feats: np.ndarray) -> np.ndarray:
    clf = joblib.load(path)
    return clf.predict_proba(feats)