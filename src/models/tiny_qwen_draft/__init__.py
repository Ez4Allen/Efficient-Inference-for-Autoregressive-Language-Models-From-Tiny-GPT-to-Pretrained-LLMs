"""Custom Qwen-token-compatible student/draft model.

``TinyQwenDraft`` remains the checkpoint-compatible class name used by the
speculative runtime.  ``TinyQwenStudent`` is an explicit study alias used when
the same architecture is evaluated as a pretrained/distilled standalone model.
"""

from .cache import TinyQwenDraftCache
from .config import TinyQwenDraftConfig
from .model import TinyCausalLMOutputWithPast, TinyQwenDraft

TinyQwenStudent = TinyQwenDraft
TinyQwenStudentConfig = TinyQwenDraftConfig

__all__ = [
    "TinyCausalLMOutputWithPast",
    "TinyQwenDraft",
    "TinyQwenStudent",
    "TinyQwenDraftCache",
    "TinyQwenDraftConfig",
    "TinyQwenStudentConfig",
]
