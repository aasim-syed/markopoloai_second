import json
import gradio as gr
import torch
from transformers import VitsModel, AutoTokenizer
from ..utils.text import bd_text_normalize

TEST_FILE = "data/processed/test_sentences_bd.jsonl"

# -------- Model (loaded once) --------
_model = None
_tok = None

def load_model():
    global _model, _tok
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = VitsModel.from_pretrained("facebook/mms-tts-ben").to(device).eval()
        _tok = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
    return _model, _tok

def tts_generate(text: str):
    """Return (sr, wav) tuple for Gradio Audio(type='numpy')."""
    m, tok = load_model()
    text = bd_text_normalize(text)
    inputs = tok([text], return_tensors="pt").to(next(m.parameters()).device)
    with torch.no_grad():
        out = m(**inputs)
    if hasattr(out, "waveform"):
        sr = getattr(m.config, "sampling_rate", 22050)
        wav = out.waveform.squeeze(0).cpu().numpy()
        return (sr, wav)
    raise RuntimeError("Model output has no waveform; use facebook/mms-tts-ben")

# -------- JSONL helpers --------
def load_test_items(path=TEST_FILE):
    """
    Returns a list of dicts: [{'idx':int,'tag':str,'text':str,'label':str}, ...]
    label format: '### | tag | truncated text…'
    Robust to BOM, blank, and comment lines (# ...).
    """
    items = []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            for i, raw in enumerate(f):
                s = raw.strip()
                if not s or s.startswith("#"):
                    continue
                try:
                    j = json.loads(s)
                except Exception:
                    # skip malformed lines silently
                    continue
                text = j.get("text", "").strip()
                if not text:
                    continue
                tag = j.get("tag", f"case{i:03d}")
                trunc = (text[:40] + "…") if len(text) > 40 else text
                label = f"{i:03d} | {tag} | {trunc}"
                items.append({"idx": i, "tag": tag, "text": text, "label": label})
    except FileNotFoundError:
        pass
    return items

# ---------- Gradio App ----------
def refresh_dropdown():
    """Reload JSONL and update dropdown choices + preview text."""
    items = load_test_items(TEST_FILE)
    labels = [it["label"] for it in items]
    default = labels[0] if labels else None
    default_text = items[0]["text"] if items else ""
    # Return: state, dropdown update, preview text
    return items, gr.Dropdown(choices=labels, value=default), default_text

def show_selected_text(choice_label, items):
    """Show full text of the selected dropdown item."""
    if not choice_label or not items:
        return ""
    for it in items:
        if it["label"] == choice_label:
            return it["text"]
    return ""

def synth_from_choice(choice_label, items):
    """Synthesize audio from the selected dropdown item."""
    if not choice_label or not items:
        return None
    for it in items:
        if it["label"] == choice_label:
            return tts_generate(it["text"])
    return None

with gr.Blocks() as demo:
    gr.Markdown("## Bangla TTS — Synthesize from JSONL dropdown")

    # Keep items in app state
    items_state = gr.State(load_test_items(TEST_FILE))
    initial_labels = [it["label"] for it in items_state.value] if items_state.value else []
    initial_value = initial_labels[0] if initial_labels else None
    initial_text = items_state.value[0]["text"] if items_state.value else ""

    with gr.Row():
        dropdown = gr.Dropdown(choices=initial_labels, value=initial_value, label="Test Sentences (from JSONL)")
        refresh = gr.Button("↻ Refresh JSONL")
    text_preview = gr.Textbox(label="Full text", value=initial_text, interactive=False, lines=2)
    synth = gr.Button("🎙️ Synthesize Selected")
    audio = gr.Audio(label="Output", type="numpy")

    # Wire up interactions
    refresh.click(fn=refresh_dropdown, inputs=None, outputs=[items_state, dropdown, text_preview])
    dropdown.change(fn=show_selected_text, inputs=[dropdown, items_state], outputs=text_preview)
    synth.click(fn=synth_from_choice, inputs=[dropdown, items_state], outputs=audio)

if __name__ == "__main__":
    demo.launch()  # add share=True if you want a public link
