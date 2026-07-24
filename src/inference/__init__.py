"""Inference algorithms and paired chat runtimes."""

from .autoregressive import AutoregressiveOutput, greedy_decode
from .chat_runtime import ChatGenerationResult, QwenPairRuntime
from .speculative import SpeculativeOutput, greedy_speculative_decode

__all__ = [
    "AutoregressiveOutput",
    "ChatGenerationResult",
    "QwenPairRuntime",
    "SpeculativeOutput",
    "greedy_decode",
    "greedy_speculative_decode",
]
