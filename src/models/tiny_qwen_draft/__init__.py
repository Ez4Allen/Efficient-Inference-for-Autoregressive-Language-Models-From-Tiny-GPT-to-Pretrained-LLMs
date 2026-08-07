"""Custom small draft model for Qwen speculative decoding."""

from .cache import TinyQwenDraftCache
from .config import TinyQwenDraftConfig
from .model import TinyCausalLMOutputWithPast, TinyQwenDraft

__all__ = [
    "TinyCausalLMOutputWithPast",
    "TinyQwenDraft",
    "TinyQwenDraftCache",
    "TinyQwenDraftConfig",
]
