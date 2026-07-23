from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from src.utils.device import resolve_device
from src.utils.paths import PROJECT_ROOT

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
else:
    PreTrainedModel = Any
    PreTrainedTokenizerBase = Any


@dataclass
class ModelBundle:
    model_name: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    dtype: torch.dtype


@dataclass
class SpeculativeModelBundle:
    draft: ModelBundle
    target: ModelBundle


def resolve_dtype(
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.dtype:
    """
    Select BF16/FP16 on CUDA and FP32 on CPU unless explicitly specified.
    """

    if dtype is not None:
        return dtype

    if device.type == "cuda":
        return (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

    return torch.float32


def load_causal_lm(
    model_name: str | Path = "openai-community/gpt2",
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> ModelBundle:
    """
    Load a causal language model from Hugging Face or a local directory.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw_model_name = str(model_name)
    direct_candidate = Path(raw_model_name).expanduser()
    project_candidate = (PROJECT_ROOT / direct_candidate).resolve()

    if direct_candidate.exists():
        local_candidate = direct_candidate.resolve()
    elif not direct_candidate.is_absolute() and project_candidate.exists():
        local_candidate = project_candidate
    else:
        local_candidate = None

    if local_candidate is not None:
        if not local_candidate.is_dir():
            raise FileNotFoundError(
                f"Local model path is not a directory: {local_candidate}"
            )
        model_name = str(local_candidate)
    else:
        model_name = raw_model_name

    effective_local_files_only = local_files_only or local_candidate is not None
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(resolved_device, dtype)

    print(f"Loading model: {model_name}")
    print(f"Device: {resolved_device}")
    print(f"Dtype: {resolved_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=effective_local_files_only,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither a PAD token nor an EOS token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": resolved_dtype,
        "trust_remote_code": trust_remote_code,
        "local_files_only": effective_local_files_only,
        "low_cpu_mem_usage": True,
    }

    if resolved_device.type == "cuda":
        device_index = (
            resolved_device.index
            if resolved_device.index is not None
            else torch.cuda.current_device()
        )

        model_kwargs["device_map"] = {
            "": device_index,
        }

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    if resolved_device.type != "cuda":
        model.to(resolved_device)

    model.eval()

    actual_parameter = next(model.parameters())

    return ModelBundle(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=actual_parameter.device,
        dtype=actual_parameter.dtype,
    )


def validate_tokenizer_compatibility(
    draft: ModelBundle,
    target: ModelBundle,
) -> None:
    """
    Verify that draft and target models use identical token IDs.
    """

    if draft.tokenizer.get_vocab() != target.tokenizer.get_vocab():
        raise ValueError(
            "Draft and target tokenizers have different vocabularies."
        )

    token_attributes = [
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
    ]

    for attribute in token_attributes:
        draft_id = getattr(draft.tokenizer, attribute)
        target_id = getattr(target.tokenizer, attribute)

        if draft_id != target_id:
            raise ValueError(
                f"Tokenizer mismatch for {attribute}: "
                f"{draft_id} != {target_id}"
            )


def load_speculative_models(
    draft_model_name: str | Path,
    target_model_name: str | Path,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> SpeculativeModelBundle:
    """
    Load draft and target models for speculative decoding.
    """

    draft = load_causal_lm(
        model_name=draft_model_name,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=effective_local_files_only,
    )

    target = load_causal_lm(
        model_name=target_model_name,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=effective_local_files_only,
    )

    validate_tokenizer_compatibility(
        draft=draft,
        target=target,
    )

    return SpeculativeModelBundle(
        draft=draft,
        target=target,
    )


def get_parameter_count(
    model: PreTrainedModel,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def print_model_info(
    bundle: ModelBundle,
) -> None:
    """
    Print a compact model summary.
    """

    parameter_count = get_parameter_count(bundle.model)

    print("\n=== Model Information ===")
    print(f"Model: {bundle.model_name}")
    print(f"Class: {bundle.model.__class__.__name__}")
    print(f"Device: {bundle.device}")
    print(f"Dtype: {bundle.dtype}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Vocabulary: {len(bundle.tokenizer):,}")