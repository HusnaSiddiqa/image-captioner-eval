"""Gradio demo: upload any image (+ optional prompt) and get a caption.

Run locally:  python app/app.py
Deploy free:  push this file + requirements.txt to a Hugging Face Space.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr
from PIL import Image

from captioner import get_captioner

CAPTIONER = get_captioner()


def describe(image: Image.Image, prompt: str) -> str:
    if image is None:
        return "Please upload an image first."
    return CAPTIONER.caption(image, prompt=prompt)


with gr.Blocks(title="Image Captioner") as demo:
    gr.Markdown(
        "# 🖼️ Image Captioner\n"
        "Upload any image and get a caption. Add an optional prompt "
        "(e.g. *a photo of*) to steer the description."
    )
    with gr.Row():
        image_in = gr.Image(type="pil", label="Image")
        with gr.Column():
            prompt_in = gr.Textbox(
                label="Optional prompt",
                placeholder="e.g. a photo of",
            )
            run = gr.Button("Generate caption", variant="primary")
            caption_out = gr.Textbox(label="Caption", lines=3)
    run.click(describe, [image_in, prompt_in], caption_out)


if __name__ == "__main__":
    demo.launch()
