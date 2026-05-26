"""Caption evaluation metrics.

Two complementary measures:

- CLIPScore: reference-free. Measures how well a caption matches the image
  using CLIP embeddings (cosine similarity, scaled to 0-100). Good for caption
  quality where there is no single "correct" answer.
- BLEU: reference-based. Measures n-gram overlap with one or more human
  reference captions. Standard in the captioning literature.

Reporting both is the point of this project: they can disagree, and explaining
why is a strong signal that you understand evaluation, not just generation.

torch and heavy libs are imported lazily so this module imports without them.
"""
from __future__ import annotations

from typing import List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def clip_score(
    images: "Sequence[Image.Image]",
    captions: Sequence[str],
    model_id: str = CLIP_MODEL_ID,
) -> float:
    """Mean image-caption cosine similarity, scaled to 0-100.

    Higher = the caption matches the image better. Reference-free.
    """
    assert len(images) == len(captions), "images and captions must align"
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()

    sims: List[float] = []
    with torch.no_grad():
        for img, cap in zip(images, captions):
            inputs = processor(
                text=[cap], images=img.convert("RGB"),
                return_tensors="pt", padding=True, truncation=True,
            ).to(device)
            out = model(**inputs)
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            sims.append(float((ie @ te.T).item()))
    return round(sum(sims) / len(sims) * 100, 2)


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def bleu_score(
    candidates: Sequence[str],
    references: Sequence[Sequence[str]],
    max_n: int = 4,
) -> float:
    """Corpus BLEU (0-100) with a simple, dependency-free implementation.

    `references[i]` is the list of human captions for candidate `i`.
    Self-contained so it runs anywhere; for papers use sacrebleu instead.
    """
    import math
    from collections import Counter

    assert len(candidates) == len(references)

    # Effective order: never higher than the shortest candidate, so short
    # captions don't force a zero n-gram precision and collapse the score.
    shortest = min((len(_tokenize(c)) for c in candidates), default=1)
    eff_n = max(1, min(max_n, shortest))

    clipped = [0] * eff_n
    totals = [0] * eff_n
    cand_len = ref_len = 0

    for cand, refs in zip(candidates, references):
        ctoks = _tokenize(cand)
        cand_len += len(ctoks)
        # closest reference length for brevity penalty
        rlens = [len(_tokenize(r)) for r in refs]
        ref_len += min(rlens, key=lambda rl: (abs(rl - len(ctoks)), rl))

        for n in range(1, eff_n + 1):
            cgrams = Counter(tuple(ctoks[i:i + n]) for i in range(len(ctoks) - n + 1))
            maxref: Counter = Counter()
            for r in refs:
                rtoks = _tokenize(r)
                rgrams = Counter(tuple(rtoks[i:i + n]) for i in range(len(rtoks) - n + 1))
                for g, c in rgrams.items():
                    maxref[g] = max(maxref[g], c)
            overlap = sum(min(c, maxref[g]) for g, c in cgrams.items())
            clipped[n - 1] += overlap
            totals[n - 1] += max(sum(cgrams.values()), 1)

    precisions = [(clipped[i] / totals[i]) if totals[i] else 0.0 for i in range(eff_n)]
    if min(precisions) > 0:
        geo = math.exp(sum(math.log(p) for p in precisions) / eff_n)
    else:
        geo = 0.0
    bp = 1.0 if cand_len > ref_len else math.exp(1 - ref_len / max(cand_len, 1))
    return round(bp * geo * 100, 2)
