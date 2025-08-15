import torch
import torch.nn.functional as F

def mel_loss(pred_mel: torch.Tensor, tgt_mel: torch.Tensor, tgt_lengths: torch.Tensor | None = None) -> torch.Tensor:
    """
    pred_mel: [B, n_mels, T_pred]
    tgt_mel : [B, n_mels, T_tgt] (padded)
    tgt_lengths: [B] true target lengths (frames). If None, use full length.
    """
    # Align time dimension
    if pred_mel.size(2) != tgt_mel.size(2):
        # interpolate pred along time to match target padded length
        pred_mel = F.interpolate(pred_mel, size=tgt_mel.size(2), mode='linear', align_corners=False)

    if tgt_lengths is None:
        return F.l1_loss(pred_mel, tgt_mel)

    # Masked L1 over true frames only
    B, C, T = tgt_mel.shape
    device = tgt_mel.device
    idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]
    mask = (idx < tgt_lengths.unsqueeze(1))                        # [B, T]
    mask = mask.unsqueeze(1).expand(B, C, T)                        # [B, C, T]
    diff = torch.abs(pred_mel - tgt_mel) * mask
    denom = mask.sum().clamp_min(1)
    return diff.sum() / denom

def bd_accent_loss(pred_mel: torch.Tensor, pred_disc_logits: torch.Tensor, target_accent: torch.Tensor,
                   phoneme_loss: torch.Tensor = None, phoneme_weight: float = 0.5, accent_weight: float = 0.2) -> torch.Tensor:
    accent = F.cross_entropy(pred_disc_logits, target_accent)
    total = accent_weight * accent
    if phoneme_loss is not None:
        total = total + phoneme_weight * phoneme_loss
    return total
