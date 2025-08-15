import argparse, soundfile as sf, torch
from transformers import VitsModel, AutoTokenizer
from ..utils.text import bd_text_normalize

"""Synthesize using MMS Bengali (no external TTS libs needed).
If the model exposes `.waveform`, save it directly.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='facebook/mms-tts-ben')
    ap.add_argument('--text', default='আম বলদশ থক এসছ')
    ap.add_argument('--out_wav', default='data/samples/sample.wav')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VitsModel.from_pretrained(args.model).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.model)

    text = bd_text_normalize(args.text)
    inputs = tok([text], return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inputs)
    if hasattr(out, 'waveform'):
        wav = out.waveform.squeeze(0).cpu().numpy()
        sr = getattr(model.config, 'sampling_rate', 22050)
        sf.write(args.out_wav, wav, sr)
        print('Saved audio to', args.out_wav)
    else:
        raise RuntimeError('Model output has no waveform; choose a model like facebook/mms-tts-ben')

if __name__ == '__main__':
    main()
