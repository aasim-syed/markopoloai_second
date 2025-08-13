import torch.nn.functional as F

def bd_accent_loss(pred_mel, target_mel, pred_phonemes=None, target_phonemes=None,
                   mel_weight=1.0, phoneme_weight=0.5, accent_weight=0.2, accent_disc=None, audio=None):
    mel_loss = F.mse_loss(pred_mel, target_mel)
    pho_loss = 0.0
    if pred_phonemes is not None and target_phonemes is not None:
        pho_loss = F.cross_entropy(pred_phonemes, target_phonemes)

    acc_loss = 0.0
    if accent_disc is not None and audio is not None:
        # Expect accent_disc to return a scalar loss encouraging BD accent
        acc_loss = accent_disc(audio)

    return mel_weight * mel_loss + phoneme_weight * pho_loss + accent_weight * acc_loss