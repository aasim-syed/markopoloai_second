import argparse, torch, soundfile as sf
from ..model.bd_vits import BDVitsModel
from ..utils.text import bd_text_normalize
from ..vocoder.griffinlim import VocoderConfig, mel_db_to_audio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='outputs/checkpoints/epoch_0.pt')
    ap.add_argument('--text', default='আমি বাংলাদেশ থেকে এসেছি।')
    ap.add_argument('--out_wav', default='data/samples/finetuned.wav')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BDVitsModel().to(device).eval()

    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd['model'], strict=False)

    text = bd_text_normalize(args.text)
    tok = model.tokenize([text])
    tok = {k: v.to(device) for k, v in tok.items()}

    with torch.no_grad():
        mel = model(tok['input_ids'], tok.get('attention_mask'))

    mel = mel.squeeze(0).detach().cpu().numpy()   # [n_mels, T]
    vcfg = VocoderConfig(sample_rate=model.sample_rate)
    y = mel_db_to_audio(mel, vcfg)
    sf.write(args.out_wav, y, vcfg.sample_rate)
    print('Saved audio to', args.out_wav)

if __name__ == '__main__':
    main()
