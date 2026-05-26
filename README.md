# 🖼️ Image Captioner + caption-quality evaluation

Upload any image, optionally add a prompt, and get a caption (pre-trained BLIP).
On top of the app sits the real focus of the project: a **rigorous
evaluation-and-experimentation pipeline** that measures caption quality across
decoding strategies and a LoRA fine-tune — built and run on free-tier compute.

> The point isn't a flashy number. It's demonstrating that you can *measure*
> a generative model properly: reference-free and reference-based metrics,
> controlled experiments, and honest interpretation (including null results).

## What's here

The **app** (`app/app.py`) is a simple Gradio UI: image + optional prompt → caption.

The **evaluation notebook** (`notebooks/evaluation_and_experiments.ipynb`) is the
substance:
1. Baseline caption quality (CLIPScore + BLEU).
2. A decoding-strategy sweep (greedy, beam search, temperature/top-p sampling).
3. A LoRA fine-tune of BLIP, pushed toward more detailed captions.
4. A before/after comparison on the same eval set.
5. An honest write-up of what moved and what didn't.

## Architecture

```
UI (upload + optional prompt)  ->  BLIP  ->  caption
                                    |
                          evaluation pipeline
                CLIPScore . BLEU . decoding sweep . LoRA before/after
```

The app is deliberately simple; the rigor lives in the notebook and in
`src/captioner/evaluation.py`.

## Repository layout

```
caption-app/
|- src/captioner/
|    |- model.py            # Captioner: load BLIP, caption()
|    |- evaluation.py       # clip_score() + dependency-free bleu_score()
|- app/app.py               # Gradio demo
|- notebooks/
|    |- evaluation_and_experiments.ipynb   # the main deliverable
|    |- optional_lora_finetune.ipynb       # minimal LoRA-only version
|- tests/                   # fast, torch-free unit tests (incl. BLEU checks)
|- pyproject.toml
|- requirements.txt
|- README.md
```

## Quickstart

### Local
```bash
pip install -e ".[demo]"
python app/app.py          # Gradio demo
pytest -q                  # unit tests (no GPU needed)
```

### Run the evaluation (Colab)
Open `notebooks/evaluation_and_experiments.ipynb`, set runtime to GPU, run top
to bottom. It produces the metrics table and before/after comparison you'll put
in this README.

### Deploy the demo
Push `app/app.py` + `requirements.txt` to a Hugging Face Space (Gradio SDK).

## Results

Evaluated on 100 Flickr30k images, comparing the base BLIP model against a
LoRA fine-tune trained to produce more detailed captions.

### Before vs after LoRA fine-tune

| | CLIPScore | BLEU |
|---|---|---|
| before fine-tune | 28.16 | 29.54 |
| after LoRA fine-tune | **33.25** | **13.94** |
| delta | **+5.09** | **-15.60** |

### Example captions (before -> after)

| Image | Before | After |
|---|---|---|
| 0 | a man standing in the grass | a man in a blue shirt and jeans is standing in a garden with a green bush and a white fence behind him |
| 2 | a little girl in a pink dress | a little girl in a pink dress is standing on a wooden platform with a chicken coop in the background |

### Decoding-strategy sweep

Same model, no training — only the generation strategy changes.

| Decoding | CLIPScore | BLEU | Example |
|---|---|---|---|
| greedy (baseline) | 28.16 | **29.54** | a man standing in the grass |
| beam search (5) | 28.25 | 23.30 | the man is wearing a blue shirt |
| sampling (t=0.7) | 26.42 | 20.32 | the man is outside |
| sampling (top-p 0.9) | 26.20 | 13.30 | the tree near the man |
| beam search (5), longer | **31.30** | 15.76 | a man in a blue shirt is standing in front of ... |

The same CLIPScore-vs-BLEU tension appears here too. Plain greedy decoding
scores highest on BLEU (its short outputs match the short references), while
longer beam search scores highest on CLIPScore (more detail = better
image match) but lowest-but-one on BLEU. Sampling adds variety at a cost to
both metrics. There is no single "best" strategy — the right choice depends on
whether you want reference-matching brevity or descriptive completeness.

### Interpretation

Fine-tuning BLIP toward more detailed captions produced a clear behavioral
shift: captions became substantially longer and more descriptive. The two
evaluation metrics then **disagreed about whether this was an improvement**.
CLIPScore rose (+5.09) because the longer captions capture more true visual
detail and therefore match the image more completely. BLEU fell sharply
(-15.60) because the generated captions no longer resemble the short human
reference captions it compares against word-for-word.

This divergence is the key finding: reference-free semantic metrics (CLIPScore)
and reference-based n-gram metrics (BLEU) reward different things, so "caption
quality" is not a single number — it depends on what you optimize for. A
secondary observation is that pushing for length occasionally introduced
repetition (e.g. "a helmet on his head and a helmet on his head"), revealing a
real quality/length trade-off rather than a free improvement.

## Skills demonstrated

- Rigorous model **evaluation**: reference-free (CLIPScore) and reference-based
  (BLEU) metrics, and understanding when they disagree.
- Controlled **experimentation** across decoding strategies.
- Parameter-efficient **fine-tuning** (LoRA) under free-GPU constraints.
- Clean package design, a deployable demo, and a tested metric implementation.

### Resume line
> Built an evaluation pipeline (CLIPScore, BLEU) for a BLIP image-captioning
> model; ran controlled decoding experiments and a LoRA fine-tune, and deployed
> an interactive Gradio demo -- all on free-tier compute.

## License
MIT -- see `LICENSE`.
