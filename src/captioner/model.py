"""Image captioning with BLIP.

One class, one job: take an image (and an optional text prompt) and return a
caption. The model is pre-trained, so this works on any image with no training.
Weights load lazily on first use and are cached, so importing is cheap.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from PIL import Image

MODEL_ID = "Salesforce/blip-image-captioning-base"


class Captioner:
    def __init__(self, model_id: str = MODEL_ID, adapter_dir: Optional[str] = None):
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id)

        # Optionally load a LoRA adapter produced by the training notebook.
        if adapter_dir:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)

        self.model = self.model.to(self.device).eval()

    def caption(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_new_tokens: int = 40,
    ) -> str:
        """Caption an image.

        If `prompt` is given (e.g. "a photo of"), BLIP continues from it —
        this is "conditional" captioning and lets the user steer the output.
        With no prompt, BLIP writes a caption from scratch.
        """
        import torch

        image = image.convert("RGB")
        text = prompt.strip() if prompt and prompt.strip() else None
        inputs = self.processor(image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


@lru_cache(maxsize=1)
def get_captioner(adapter_dir: Optional[str] = None) -> Captioner:
    """Cached singleton so repeated calls don't reload the weights."""
    return Captioner(adapter_dir=adapter_dir)
