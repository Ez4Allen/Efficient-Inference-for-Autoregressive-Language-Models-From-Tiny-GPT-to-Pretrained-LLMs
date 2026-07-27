from __future__ import annotations

import torch

from src.evaluation.model_pair_alignment import (
    analyze_model_pair_logits,
    entropy_from_logits,
    js_divergence_from_logits,
    topk_overlap,
)


def test_identical_logits_have_full_agreement_and_zero_js():
    logits = torch.tensor([[[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]]])
    report = analyze_model_pair_logits(logits, logits, top_k=2)
    assert report.top1_agreement == 1.0
    assert report.mean_topk_overlap == 1.0
    assert report.mean_js_divergence < 1e-7


def test_alignment_detects_different_top1():
    draft = torch.tensor([[[4.0, 1.0], [3.0, 0.0]]])
    target = torch.tensor([[[1.0, 4.0], [3.0, 0.0]]])
    report = analyze_model_pair_logits(draft, target, top_k=1)
    assert report.top1_agreement == 0.5
    assert 0.0 <= report.mean_js_divergence


def test_entropy_and_topk_shapes():
    logits = torch.randn(2, 4, 10)
    assert entropy_from_logits(logits).shape == (2, 4)
    assert js_divergence_from_logits(logits, logits).shape == (2, 4)
    assert topk_overlap(logits, logits, k=5).shape == (2, 4)
