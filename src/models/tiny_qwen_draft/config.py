"""Configuration for the custom Qwen-token-compatible draft model.

The class intentionally stays dependency-light.  It mirrors the subset of a
Hugging Face causal-LM config that the rest of this project needs, while the
model itself remains a plain PyTorch implementation written in this repository.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from src.utils.io import read_json, write_json


@dataclass
class TinyQwenDraftConfig:
    """Validated architecture and tokenizer contract for ``TinyQwenDraft``."""

    model_type: str = "tiny_qwen_draft"
    schema_version: int = 1

    vocab_size: int = 0
    hidden_size: int = 256
    intermediate_size: int = 768
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 4096

    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1.0e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02

    tie_word_embeddings: bool = True
    use_cache: bool = True

    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    unk_token_id: int | None = None

    tokenizer_name_or_path: str | None = None
    target_model_name_or_path: str | None = None
    tokenizer_sha256: str | None = None
    chat_template_sha256: str | None = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def validate(self) -> None:
        if self.model_type != "tiny_qwen_draft":
            raise ValueError(
                "model_type must be 'tiny_qwen_draft', got "
                f"{self.model_type!r}."
            )
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise ValueError("schema_version must be a positive integer.")
        integer_fields = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in integer_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads."
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by "
                "num_key_value_heads for grouped-query attention."
            )
        if self.head_dim % 2 != 0:
            raise ValueError("The attention head dimension must be even for RoPE.")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive.")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive.")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1).")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive.")

        for name in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or null.")
            if value is not None and value >= self.vocab_size:
                raise ValueError(
                    f"{name}={value} is outside vocab_size={self.vocab_size}."
                )

        for name in ("tokenizer_sha256", "chat_template_sha256"):
            value = getattr(self, name)
            if value is not None:
                normalized = str(value).strip().casefold()
                if len(normalized) != 64 or any(
                    character not in "0123456789abcdef" for character in normalized
                ):
                    raise ValueError(f"{name} must be a 64-character SHA-256 hex digest.")
                setattr(self, name, normalized)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TinyQwenDraftConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("TinyQwenDraft config must be a mapping.")
        allowed = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in payload.items() if key in allowed}
        config = cls(**filtered)
        config.validate()
        return config

    def save_pretrained(self, directory: str | Path) -> Path:
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        return write_json(
            directory / "config.json",
            self.to_dict(),
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "TinyQwenDraftConfig":
        path = Path(directory).expanduser().resolve() / "config.json"
        if not path.exists():
            raise FileNotFoundError(f"TinyQwenDraft config not found: {path}")
        payload = read_json(path)
        return cls.from_dict(payload)
