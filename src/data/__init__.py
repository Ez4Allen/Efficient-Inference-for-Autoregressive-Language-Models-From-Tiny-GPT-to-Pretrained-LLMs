"""Dataset and controlled-prompt utilities."""

from .causal_pretraining import (
    CausalCorpusRecord,
    CausalDataCollator,
    CausalPackedDataset,
    load_causal_corpus,
)
from .prompt_builder import (
    PromptBatch,
    build_prompt_batch,
    make_prompt,
    supported_prompt_types,
)

__all__ = [
    "CausalCorpusRecord",
    "CausalDataCollator",
    "CausalPackedDataset",
    "PromptBatch",
    "build_prompt_batch",
    "load_causal_corpus",
    "make_prompt",
    "supported_prompt_types",
]
