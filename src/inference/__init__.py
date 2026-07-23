"""Autoregressive and speculative decoding APIs."""

from .autoregressive import AutoregressiveOutput, greedy_decode
from .speculative import SpeculativeOutput, greedy_speculative_decode

__all__ = [
    "AutoregressiveOutput",
    "SpeculativeOutput",
    "greedy_decode",
    "greedy_speculative_decode",
]
