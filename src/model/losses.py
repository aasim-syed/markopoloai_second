import torch
import torch.nn.functional as F

def mel_loss(pred_mel: torch.Tensor, tgt_mel: torch.Tensor, tgt_lengths: torch.Tensor | None = None) -> torch.Tensor:
    # align time dim
    if pred_mel.size(2) != tgt_mel.size(2):
        pred_mel = F.interpolate(pred_mel, size=tgt_mel.size(2), mode='linear', align_corners=False)
    if tgt_lengths is None:
        return F.l1_loss(pred_mel, tgt_mel)
    B, C, T = tgt_mel.shape
    device = tgt_mel.device
    idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    mask = (idx < tgt_lengths.unsqueeze(1)).unsqueeze(1).expand(B, C, T)
    diff = torch.abs(pred_mel - tgt_mel) * mask
    denom = mask.sum().clamp_min(1)
    return diff.sum() / denom
