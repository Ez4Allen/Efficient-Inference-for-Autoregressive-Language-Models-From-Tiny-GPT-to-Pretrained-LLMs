"""Inference algorithms and paired chat runtimes."""

from .autoregressive import AutoregressiveOutput, greedy_decode
from .chat_runtime import ChatGenerationResult, QwenPairRuntime
from .consistency import (
    BlockConsistencyReport,
    BlockMismatch,
    diagnose_target_block_consistency,
)
from .speculative import SpeculativeOutput, greedy_speculative_decode

__all__ = [
    "AutoregressiveOutput",
    "BlockConsistencyReport",
    "BlockMismatch",
    "ChatGenerationResult",
    "QwenPairRuntime",
    "SpeculativeOutput",
    "diagnose_target_block_consistency",
    "greedy_decode",
    "greedy_speculative_decode",
]
