from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data.tiny_lm_dataset import TextDataset, create_train_val_split
from src.models.tiny_lm import CharTokenizer, TinyGPT, sample_next_token
from src.models.tiny_lm.train import load_training_config


def test_char_tokenizer_round_trip_and_persistence(tmp_path: Path) -> None:
    tokenizer = CharTokenizer().fit("abc abc")
    encoded = tokenizer.encode("cab")
    assert tokenizer.decode(encoded) == "cab"
    saved = tokenizer.save(tmp_path / "tokenizer.json")
    loaded = CharTokenizer.load(saved)
    assert loaded.decode(encoded) == "cab"
    with pytest.raises(ValueError):
        loaded.encode("z")


def test_tiny_gpt_forward_and_top_k_generation() -> None:
    torch.manual_seed(0)
    model = TinyGPT(
        vocab_size=11,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=8,
        d_ff=16,
        dropout=0.0,
    )
    inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    logits, loss = model(inputs, inputs)
    assert logits.shape == (1, 4, 11)
    assert loss is not None and torch.isfinite(loss)

    generated = model.generate(inputs, max_new_tokens=3, top_k=1)
    assert generated.shape == (1, 7)
    with pytest.raises(ValueError):
        model.generate(inputs, max_new_tokens=1, temperature=0)


def test_sampling_and_dataset_validation() -> None:
    logits = torch.tensor([[0.0, 10.0, -2.0]])
    token = sample_next_token(logits, top_k=1)
    assert token.item() == 1

    train, validation = create_train_val_split(list(range(20)), val_ratio=0.2)
    assert len(train) == 16
    assert len(validation) == 4
    dataset = TextDataset(train, block_size=4)
    inputs, targets = dataset[0]
    assert inputs.tolist() == [0, 1, 2, 3]
    assert targets.tolist() == [1, 2, 3, 4]


def test_tiny_training_config_is_portable() -> None:
    config = load_training_config("configs/tiny_gpt.yaml")
    assert config.data_path.name == "input.txt"
    assert config.output_dir.name == "tiny_gpt_shakespeare"
    assert config.d_ff == 512
