import torch, torch.nn.functional as F


def mel_loss(pred_mel: torch.Tensor, tgt_mel: torch.Tensor) -> torch.Tensor:
    # L1 on log-mel
    return F.l1_loss(pred_mel, tgt_mel)


def bd_accent_loss(pred_mel: torch.Tensor, pred_disc_logits: torch.Tensor, target_accent: torch.Tensor, phoneme_loss: torch.Tensor = None,
                   phoneme_weight: float = 0.5, accent_weight: float = 0.2) -> torch.Tensor:
    # Accent BCE with logits (2 classes): target_accent in {0: IN, 1: BD}
    accent = F.cross_entropy(pred_disc_logits, target_accent)
    total = accent_weight * accent
    if phoneme_loss is not None:
        total = total + phoneme_weight * phoneme_loss
    return total