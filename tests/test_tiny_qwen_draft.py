from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.tiny_qwen_draft import (
    TinyQwenDraft,
    TinyQwenDraftConfig,
)


def tiny_config(**overrides) -> TinyQwenDraftConfig:
    values = {
        "vocab_size": 41,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    values.update(overrides)
    return TinyQwenDraftConfig(**values)


def test_tiny_qwen_config_validates_gqa_and_token_ids() -> None:
    tiny_config().validate()
    with pytest.raises(ValueError, match="divisible"):
        tiny_config(num_attention_heads=3).validate()
    with pytest.raises(ValueError, match="outside vocab_size"):
        tiny_config(eos_token_id=100).validate()
    with pytest.raises(ValueError, match="positive integer"):
        tiny_config(hidden_size=16.0).validate()


def test_forward_loss_weight_tying_and_cache_shape() -> None:
    torch.manual_seed(0)
    model = TinyQwenDraft(tiny_config()).eval()
    input_ids = torch.tensor([[1, 3, 4, 5]], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :2] = -100

    output = model(input_ids=input_ids, labels=labels, use_cache=True)

    assert output.logits.shape == (1, 4, 41)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.past_key_values is not None
    assert output.past_key_values.sequence_length == 4
    assert len(output.past_key_values) == 2
    assert model.lm_head.weight is model.embed_tokens.weight


def test_incremental_cache_matches_full_forward() -> None:
    torch.manual_seed(1)
    model = TinyQwenDraft(tiny_config()).eval()
    input_ids = torch.tensor([[1, 7, 8, 9, 10, 11]], dtype=torch.long)

    full = model(input_ids=input_ids, use_cache=True)
    prefix = model(input_ids=input_ids[:, :3], use_cache=True)
    cache = prefix.past_key_values
    pieces = [prefix.logits]

    for index in range(3, input_ids.shape[1]):
        step = model(
            input_ids=input_ids[:, index : index + 1],
            past_key_values=cache,
            use_cache=True,
        )
        cache = step.past_key_values
        pieces.append(step.logits)

    incremental_logits = torch.cat(pieces, dim=1)
    torch.testing.assert_close(
        incremental_logits,
        full.logits,
        atol=1e-5,
        rtol=1e-5,
    )
    assert cache is not None and cache.sequence_length == input_ids.shape[1]


def test_cache_crop_supports_speculative_mismatch_recovery() -> None:
    torch.manual_seed(2)
    model = TinyQwenDraft(tiny_config()).eval()
    prompt = torch.tensor([[1, 3, 4, 5]], dtype=torch.long)
    output = model(input_ids=prompt, use_cache=True)
    cache = output.past_key_values
    assert cache is not None

    extension = model(
        input_ids=torch.tensor([[6, 7, 8]], dtype=torch.long),
        past_key_values=cache,
        use_cache=True,
    )
    cache = extension.past_key_values
    assert cache is not None and cache.sequence_length == 7

    cache.crop(5)
    assert cache.sequence_length == 5
    corrected = model(
        input_ids=torch.tensor([[9]], dtype=torch.long),
        past_key_values=cache,
        use_cache=True,
    )
    assert corrected.past_key_values is not None
    assert corrected.past_key_values.sequence_length == 6


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = TinyQwenDraft(tiny_config()).eval()
    input_ids = torch.tensor([[1, 12, 13]], dtype=torch.long)
    expected = model(input_ids=input_ids, use_cache=False).logits

    checkpoint = model.save_pretrained(tmp_path / "draft")
    reloaded = TinyQwenDraft.from_pretrained(checkpoint).eval()
    actual = reloaded(input_ids=input_ids, use_cache=False).logits

    torch.testing.assert_close(actual, expected)
    assert reloaded.config.model_type == "tiny_qwen_draft"
    assert (checkpoint / "config.json").exists()
    assert (checkpoint / "pytorch_model.bin").exists()


def test_loss_only_projects_supervised_positions_only() -> None:
    model = TinyQwenDraft(tiny_config()).eval()
    input_ids = torch.tensor([[1, 3, 4, 5, 6]], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    labels[:, -2:] = input_ids[:, -2:]

    output = model(
        input_ids=input_ids,
        labels=labels,
        use_cache=False,
        loss_only=True,
    )

    assert output.logits.shape == (2, 41)
    assert output.loss is not None and torch.isfinite(output.loss)


def test_custom_draft_runs_in_project_speculative_decoder() -> None:
    from src.inference.autoregressive import greedy_decode
    from src.inference.speculative import greedy_speculative_decode

    torch.manual_seed(4)
    model = TinyQwenDraft(tiny_config()).eval()
    prompt = torch.tensor([[1, 3, 4]], dtype=torch.long)

    baseline = greedy_decode(model, prompt, max_new_tokens=7)
    speculative = greedy_speculative_decode(
        model,
        model,
        prompt,
        max_new_tokens=7,
        draft_tokens_per_round=3,
    )

    assert torch.equal(
        baseline.generated_token_ids,
        speculative.generated_token_ids,
    )
    assert speculative.acceptance_rate == pytest.approx(1.0)
