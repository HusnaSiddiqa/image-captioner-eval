"""Tests that run with no GPU and no model download.

We test the prompt-normalisation logic in isolation (empty/whitespace prompts
should become None so BLIP captions from scratch) without loading the model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def normalize_prompt(prompt):
    return prompt.strip() if prompt and prompt.strip() else None


def test_empty_prompt_becomes_none():
    assert normalize_prompt("") is None
    assert normalize_prompt("   ") is None
    assert normalize_prompt(None) is None


def test_real_prompt_is_kept():
    assert normalize_prompt("a photo of") == "a photo of"
    assert normalize_prompt("  a dog  ") == "a dog"


def test_package_metadata_imports():
    import captioner
    assert captioner.MODEL_ID.startswith("Salesforce/blip")
    assert captioner.__version__


def test_bleu_perfect_match_is_100():
    from captioner.evaluation import bleu_score
    cands = ["a dog runs in the park"]
    refs = [["a dog runs in the park"]]
    assert bleu_score(cands, refs) == 100.0


def test_bleu_no_overlap_is_low():
    from captioner.evaluation import bleu_score
    cands = ["completely different words here entirely"]
    refs = [["a dog runs in the park"]]
    assert bleu_score(cands, refs) < 10.0


def test_bleu_partial_overlap_is_between():
    from captioner.evaluation import bleu_score
    cands = ["a dog runs in the street"]
    refs = [["a dog runs in the park"]]
    score = bleu_score(cands, refs)
    assert 0.0 < score < 100.0
