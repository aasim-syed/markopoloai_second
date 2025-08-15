import gradio as gr
import torch
from transformers import VitsModel, AutoTokenizer
from ..utils.text import bd_text_normalize

model = None
tok = None

def load_model():
    global model, tok
    if model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = VitsModel.from_pretrained('facebook/mms-tts-ben').to(device).eval()
        tok = AutoTokenizer.from_pretrained('facebook/mms-tts-ben')
    return model, tok

def bd_tts_interface(text: str):
    m, tokenizer = load_model()
    text = bd_text_normalize(text)
    inputs = tokenizer([text], return_tensors='pt').to(next(m.parameters()).device)
    with torch.no_grad():
        out = m(**inputs)
    if hasattr(out, 'waveform'):
        sr = getattr(m.config, 'sampling_rate', 22050)
        wav = out.waveform.squeeze(0).cpu().numpy()
        return (sr, wav)
    else:
        return (22050, None)

with gr.Blocks() as demo:
    gr.Markdown('# Bangladeshi Bengali TTS — Demo (MMS, no Coqui)')
    inp = gr.Textbox(label='Bengali Text (Bangladeshi Style)', value='আমি বাংলাদেশ থেকে এসেছি।')
    out = gr.Audio(label='Generated Speech', type='numpy')
    btn = gr.Button('Synthesize')
    btn.click(bd_tts_interface, inputs=inp, outputs=out)

if __name__ == '__main__':
    demo.launch()
