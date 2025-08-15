import os, yaml, argparse, time
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
from ..utils.seed import set_seed
from ..utils.logging import get_logger, ensure_dir
from ..utils.audio import MelConfig
from ..data.dataset import TTSDataset, Collator
from ..models.bd_vits import BDVitsModel
from .losses import mel_loss, bd_accent_loss
from .scheduler import get_warmup_decay


def freeze_all(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_module(model: torch.nn.Module, module_name: str):
    for n, p in model.named_parameters():
        if n.startswith(module_name):
            p.requires_grad = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='configs/datasets.yaml')
    ap.add_argument('--cfg', default='configs/training.yaml')
    args = ap.parse_args()

    logger = get_logger('train_tts')

    with open(args.train) as f: dcfg = yaml.safe_load(f)
    with open(args.cfg) as f: tcfg = yaml.safe_load(f)

    set_seed(tcfg.get('seed',42))

    mel_cfg = MelConfig(
        sample_rate=dcfg['sample_rate'], n_fft=dcfg['n_fft'], hop_length=dcfg['hop_length'],
        win_length=dcfg['win_length'], n_mels=dcfg['n_mels'], fmin=dcfg['fmin'], fmax=dcfg['fmax']
    )

    train_ds = TTSDataset(dcfg['train_manifest'], mel_cfg, dcfg.get('use_phonemes', True))
    valid_ds = TTSDataset(dcfg['valid_manifest'], mel_cfg, dcfg.get('use_phonemes', True))

    collate = Collator()
    dl_tr = DataLoader(train_ds, batch_size=tcfg['batch_size'], shuffle=True, num_workers=2, collate_fn=collate)
    dl_va = DataLoader(valid_ds, batch_size=tcfg['batch_size'], shuffle=False, num_workers=2, collate_fn=collate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = BDVitsModel().to(device)
    freeze_all(model)
    # Unfreeze as per schedule epoch milestones
    unfreeze_schedule = tcfg['progressive_unfreeze']

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=tcfg['learning_rate'])
    total_steps = tcfg.get('max_steps') or (tcfg['num_epochs'] * len(dl_tr))
    sched = get_warmup_decay(optim, tcfg['warmup_steps'], total_steps)
    scaler = GradScaler(enabled=tcfg.get('mixed_precision', True))

    if tcfg['wandb']['enabled']:
        wandb.init(project=tcfg['wandb']['project'], config={**dcfg, **tcfg})

    global_step = 0
    best_val = float('inf')
    ensure_dir(tcfg['checkpoint_dir'])

    for epoch in range(tcfg['num_epochs']):
        # Progressive unfreezing
        for ent in unfreeze_schedule:
            if epoch >= ent['epoch']:
                unfreeze_module(model, ent['module'])

        model.train()
        running = 0.0
        for i, batch in enumerate(dl_tr):
            texts = batch['texts']
            mels = batch['mels'].to(device)
            tok = model.tokenize(texts)
            input_ids = tok['input_ids'].to(device)
            attn = tok.get('attention_mask', None)
            if attn is not None: attn = attn.to(device)

            with autocast(enabled=tcfg.get('mixed_precision', True)):
                pred_mel, disc_logits = model(input_ids, attn)
                L_mel = mel_loss(pred_mel, mels)
                # Here we assume all curated train data is BD (label=1)
                target_accent = torch.ones(pred_mel.size(0), dtype=torch.long, device=device)
                L_acc = bd_accent_loss(pred_mel, disc_logits, target_accent)
                loss = L_mel + L_acc

            scaler.scale(loss).backward()
            if (i+1) % tcfg['gradient_accumulation_steps'] == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

                if tcfg['wandb']['enabled'] and global_step % 10 == 0:
                    wandb.log({'train/loss': loss.item(), 'train/mel': L_mel.item()}, step=global_step)

                if tcfg['save_strategy'] == 'steps' and global_step % tcfg['save_steps'] == 0:
                    ckpt = os.path.join(tcfg['checkpoint_dir'], f'step_{global_step}.pt')
                    torch.save({'model': model.state_dict(), 'step': global_step}, ckpt)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dl_va:
                texts = batch['texts']
                mels = batch['mels'].to(device)
                tok = model.tokenize(texts)
                input_ids = tok['input_ids'].to(device)
                attn = tok.get('attention_mask', None)
                if attn is not None: attn = attn.to(device)
                pred_mel, disc_logits = model(input_ids, attn)
                val_loss += mel_loss(pred_mel, mels).item()
        val_loss /= max(1, len(dl_va))
        if tcfg['wandb']['enabled']:
            wandb.log({'val/mel': val_loss, 'epoch': epoch}, step=global_step)
        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(tcfg['checkpoint_dir'], 'best.pt')
            torch.save({'model': model.state_dict(), 'step': global_step, 'epoch': epoch}, ckpt)
            logger.info(f"New best {best_val:.4f} saved -> {ckpt}")

    logger.info('Training completed')

if __name__ == '__main__':
    main()