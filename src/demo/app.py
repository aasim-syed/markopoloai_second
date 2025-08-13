import gradio as gr, torch
from transformers import VitsModel, AutoTokenizer

model = VitsModel.from_pretrained("bangla-speech-processing/bangla_tts_female")
tok = AutoTokenizer.from_pretrained("bangla-speech-processing/bangla_tts_female")

def bd_tts_interface(text: str):
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        audio = model.generate(**inputs).squeeze(0).cpu().numpy()
    return (22050, audio)

examples = [
    "আমি বাংলাদেশ থেকে এসেছি।",
    "ঢাকা শহরে অনেক মানুষ বাস করে।",
    "আজকে আবহাওয়া খুবই সুন্দর।"
]

demo = gr.Interface(
    fn=bd_tts_interface,
    inputs=gr.Textbox(label="Bengali Text (Bangladeshi Style)"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
    title="Bangladeshi Bengali TTS",
    examples=examples,
)

if __name__ == "__main__":
    demo.launch()