import os, yaml, argparse
import torch
from torch.utils.data import DataLoader
from ..utils.audio import MelConfig
from ..utils.text import bd_text_normalize  # ensures import touches text module
from ..data.dataset import TTSDataset, Collator
from ..model.bd_vits import BDVitsModel
from .losses import mel_loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='configs/datasets.yaml')
    ap.add_argument('--cfg', default='configs/training.yaml')
    args = ap.parse_args()

    with open(args.train) as f: dcfg = yaml.safe_load(f)
    with open(args.cfg) as f: tcfg = yaml.safe_load(f)

    mel_cfg = MelConfig(
        sample_rate=dcfg['sample_rate'], n_fft=dcfg['n_fft'], hop_length=dcfg['hop_length'],
        win_length=dcfg['win_length'], n_mels=dcfg['n_mels'], fmin=dcfg['fmin'], fmax=dcfg['fmax']
    )

    train_ds = TTSDataset(dcfg['train_manifest'], mel_cfg, dcfg.get('use_phonemes', False))
    valid_ds = TTSDataset(dcfg['valid_manifest'], mel_cfg, dcfg.get('use_phonemes', False))

    collate = Collator()
    dl_tr = DataLoader(train_ds, batch_size=tcfg['batch_size'], shuffle=True, num_workers=0, collate_fn=collate)
    dl_va = DataLoader(valid_ds, batch_size=tcfg['batch_size'], shuffle=False, num_workers=0, collate_fn=collate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BDVitsModel().to(device)

    # simple optimizer on all params
    optim = torch.optim.AdamW(model.parameters(), lr=tcfg['learning_rate'])

    os.makedirs(tcfg['checkpoint_dir'], exist_ok=True)

    for epoch in range(tcfg['num_epochs']):
        model.train()
        for batch in dl_tr:
            texts = batch['texts']
            mels = batch['mels'].to(device)
            lengths = batch.get('mel_lengths')
            if lengths is None:
                lengths = torch.full((mels.size(0),), mels.size(2), dtype=torch.long, device=device)
            else:
                lengths = lengths.to(device)

            tok = model.tokenize(texts)
            input_ids = tok['input_ids'].to(device)
            attn = tok.get('attention_mask')
            if attn is not None: attn = attn.to(device)

            pred_mel = model(input_ids, attn)
            loss = mel_loss(pred_mel, mels, lengths)

            optim.zero_grad()
            loss.backward()
            optim.step()

        # one simple val pass
        model.eval()
        val = 0.0
        with torch.no_grad():
            for batch in dl_va:
                texts = batch['texts']
                mels = batch['mels'].to(device)
                lengths = batch.get('mel_lengths')
                if lengths is None:
                    lengths = torch.full((mels.size(0),), mels.size(2), dtype=torch.long, device=device)
                else:
                    lengths = lengths.to(device)
                tok = model.tokenize(texts)
                input_ids = tok['input_ids'].to(device)
                attn = tok.get('attention_mask')
                if attn is not None: attn = attn.to(device)
                pred_mel = model(input_ids, attn)
                val += float(mel_loss(pred_mel, mels, lengths))
        torch.save({'model': model.state_dict(), 'epoch': epoch}, os.path.join(tcfg['checkpoint_dir'], f'epoch_{epoch}.pt'))
    print("Training finished.")

if __name__ == '__main__':
    main()
