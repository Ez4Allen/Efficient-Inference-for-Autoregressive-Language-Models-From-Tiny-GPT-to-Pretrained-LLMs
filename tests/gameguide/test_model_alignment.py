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


def test_sequence_analysis_prefills_once_and_scores_completion_only():
    from src.evaluation.model_pair_alignment import analyze_model_pair_on_sequence
    from src.models.tiny_qwen_draft import TinyQwenDraft, TinyQwenDraftConfig

    config = TinyQwenDraftConfig(
        vocab_size=19,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    draft = TinyQwenDraft(config)
    target = TinyQwenDraft(config)
    target.load_state_dict(draft.state_dict())
    input_ids = torch.tensor([[1, 3, 4, 5, 6, 7]], dtype=torch.long)

    report = analyze_model_pair_on_sequence(
        draft,
        target,
        input_ids,
        completion_start=3,
        top_k=3,
    )

    assert report.positions == 3
    assert report.top1_agreement == 1.0
    assert report.mean_topk_overlap == 1.0
    assert report.mean_js_divergence < 1e-7
