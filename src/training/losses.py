import torch
import torch.nn.functional as F

def mel_loss(pred_mel: torch.Tensor, tgt_mel: torch.Tensor, tgt_lengths: torch.Tensor | None = None) -> torch.Tensor:
    """
    pred_mel: [B, n_mels, T_pred]
    tgt_mel : [B, n_mels, T_tgt] (padded)
    tgt_lengths: [B] real target lengths in frames; if None, use full padded length.
    """
    # Align time dimension if needed
    if pred_mel.size(2) != tgt_mel.size(2):
        pred_mel = F.interpolate(pred_mel, size=tgt_mel.size(2), mode="linear", align_corners=False)

    if tgt_lengths is None:
        return F.l1_loss(pred_mel, tgt_mel)

    B, C, T = tgt_mel.shape
    device = tgt_mel.device
    idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)   # [B, T]
    mask = (idx < tgt_lengths.unsqueeze(1)).unsqueeze(1).expand(B, C, T)  # [B, C, T]
    diff = torch.abs(pred_mel - tgt_mel) * mask
    denom = mask.sum().clamp_min(1)
    return diff.sum() / denom

# Optional accent loss (safe no-op if unused)
def bd_accent_loss(*args, **kwargs) -> torch.Tensor:
    return torch.tensor(0.0)
