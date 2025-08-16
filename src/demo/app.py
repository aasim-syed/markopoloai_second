import gradio as gr, torch
from ..model.bd_vits import BDVitsModel
from ..utils.text import bd_text_normalize
from ..vocoder.griffinlim import VocoderConfig, mel_db_to_audio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_model = None
_ckpt_path = None

def load_model(ckpt: str):
    global _model, _ckpt_path
    if _model is None or ckpt != _ckpt_path:
        m = BDVitsModel().to(device).eval()
        sd = torch.load(ckpt, map_location=device)
        m.load_state_dict(sd['model'], strict=False)
        _model, _ckpt_path = m, ckpt
    return _model

def tts(text: str, ckpt: str):
    m = load_model(ckpt)
    text = bd_text_normalize(text)
    tok = m.tokenize([text])
    tok = {k: v.to(device) for k, v in tok.items()}
    with torch.no_grad():
        mel = m(tok['input_ids'], tok.get('attention_mask'))
    mel = mel.squeeze(0).cpu().numpy()
    vcfg = VocoderConfig(sample_rate=m.sample_rate)
    y = mel_db_to_audio(mel, vcfg)
    return (vcfg.sample_rate, y)

with gr.Blocks() as demo:
    gr.Markdown('# Bangla TTS (BD) — Finetuned Demo')
    ckpt = gr.Textbox(value='outputs/checkpoints/epoch_0.pt', label='Checkpoint path')
    inp = gr.Textbox(value='আমি বাংলাদেশ থেকে এসেছি।', label='Text')
    out = gr.Audio(label='Speech', type='numpy')
    gr.Button('Synthesize').click(tts, inputs=[inp, ckpt], outputs=out)

if __name__ == '__main__':
    demo.launch()
