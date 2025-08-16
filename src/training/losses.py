import torch, torch.nn.functional as F

def mel_loss(pred_mel, tgt_mel):
    # Crop both to the same time length to avoid broadcasting errors
    T = min(pred_mel.size(-1), tgt_mel.size(-1))
    return F.l1_loss(pred_mel[..., :T], tgt_mel[..., :T])

def bd_accent_loss(pred_mel: torch.Tensor, pred_disc_logits: torch.Tensor, target_accent: torch.Tensor,
                   phoneme_loss: torch.Tensor = None, phoneme_weight: float = 0.5, accent_weight: float = 0.2) -> torch.Tensor:
    accent = F.cross_entropy(pred_disc_logits, target_accent)
    total = accent_weight * accent
    if phoneme_loss is not None:
        total = total + phoneme_weight * phoneme_loss
    return total
