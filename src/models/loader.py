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
    adapter_name: str | None = None
    quantized: bool = False


@dataclass
class SpeculativeModelBundle:
    draft: ModelBundle
    target: ModelBundle


def resolve_dtype(
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.dtype:
    """Select BF16/FP16 on CUDA and FP32 on CPU unless specified."""

    if dtype is not None:
        return dtype

    if device.type == "cuda":
        return (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

    return torch.float32


def resolve_dtype_name(value: str | torch.dtype | None) -> torch.dtype | None:
    """Resolve a user-facing dtype name; ``auto`` returns ``None``."""

    if value is None or isinstance(value, torch.dtype):
        return value

    normalized = str(value).strip().casefold()
    if normalized in {"", "auto"}:
        return None

    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {value!r}")
    return mapping[normalized]


def _resolve_reference(value: str | Path) -> tuple[str, bool]:
    """Resolve a local path while preserving a Hugging Face model ID."""

    raw_value = str(value)
    direct_candidate = Path(raw_value).expanduser()
    project_candidate = (PROJECT_ROOT / direct_candidate).resolve()

    if direct_candidate.exists():
        local_candidate = direct_candidate.resolve()
    elif not direct_candidate.is_absolute() and project_candidate.exists():
        local_candidate = project_candidate
    else:
        return raw_value, False

    if not local_candidate.is_dir():
        raise FileNotFoundError(
            f"Local model reference is not a directory: {local_candidate}"
        )
    return str(local_candidate), True


def _load_adapter(
    model: Any,
    adapter_reference: str | Path,
    *,
    expected_base_model: str,
) -> tuple[Any, str]:
    try:
        from peft import PeftConfig, PeftModel
    except ImportError as error:
        raise RuntimeError(
            "Loading a LoRA/QLoRA adapter requires PEFT. Install it with "
            "`pip install -r requirements-training.txt`."
        ) from error

    resolved_adapter, adapter_is_local = _resolve_reference(adapter_reference)
    peft_config = PeftConfig.from_pretrained(
        resolved_adapter,
        local_files_only=adapter_is_local,
    )
    adapter_base = str(
        getattr(peft_config, "base_model_name_or_path", "") or ""
    ).rstrip("/")
    expected = str(expected_base_model).rstrip("/")
    expected_is_local = Path(expected).expanduser().exists()
    adapter_base_is_local = bool(adapter_base) and Path(adapter_base).expanduser().exists()
    if (
        adapter_base
        and not expected_is_local
        and not adapter_base_is_local
        and adapter_base != expected
    ):
        raise ValueError(
            "Adapter/base mismatch: the adapter was trained from "
            f"{adapter_base!r}, but the configured base model is {expected!r}."
        )

    model = PeftModel.from_pretrained(
        model,
        resolved_adapter,
        local_files_only=adapter_is_local,
        is_trainable=False,
    )
    return model, resolved_adapter


def load_causal_lm(
    model_name: str | Path = "openai-community/gpt2",
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    *,
    adapter_path: str | Path | None = None,
    load_in_4bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> ModelBundle:
    """Load a causal LM from Hugging Face or a local directory.

    The function is shared by assistant inference, benchmarking, training
    smoke tests, and speculative decoding. Optional PEFT adapters are attached
    after the base checkpoint is loaded.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_model_name, model_is_local = _resolve_reference(model_name)
    effective_local_files_only = local_files_only or model_is_local
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(resolved_device, dtype)

    if load_in_4bit and resolved_device.type != "cuda":
        raise RuntimeError("4-bit bitsandbytes inference requires a CUDA GPU.")

    print(f"Loading model: {resolved_model_name}")
    print(f"Device: {resolved_device}")
    print(f"Dtype: {resolved_dtype}")
    if load_in_4bit:
        print("Quantization: 4-bit NF4")

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=effective_local_files_only,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a PAD token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "dtype": resolved_dtype,
        "trust_remote_code": trust_remote_code,
        "local_files_only": effective_local_files_only,
        "low_cpu_mem_usage": True,
    }

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError(
                "4-bit inference requires a Transformers build with "
                "BitsAndBytesConfig and the bitsandbytes package."
            ) from error

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(bnb_4bit_quant_type),
            bnb_4bit_use_double_quant=bool(bnb_4bit_use_double_quant),
            bnb_4bit_compute_dtype=resolved_dtype,
        )

    if resolved_device.type == "cuda":
        device_index = (
            resolved_device.index
            if resolved_device.index is not None
            else torch.cuda.current_device()
        )
        model_kwargs["device_map"] = {"": device_index}

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_name,
        **model_kwargs,
    )

    if resolved_device.type != "cuda":
        model.to(resolved_device)

    resolved_adapter: str | None = None
    if adapter_path is not None:
        model, resolved_adapter = _load_adapter(
            model,
            adapter_path,
            expected_base_model=resolved_model_name,
        )

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    try:
        actual_parameter = next(model.parameters())
        actual_device = actual_parameter.device
    except StopIteration:
        actual_device = resolved_device

    return ModelBundle(
        model_name=resolved_model_name,
        model=model,
        tokenizer=tokenizer,
        device=actual_device,
        dtype=resolved_dtype,
        adapter_name=resolved_adapter,
        quantized=bool(load_in_4bit),
    )


def validate_tokenizer_compatibility(
    draft: ModelBundle,
    target: ModelBundle,
) -> None:
    """Verify that draft and target models use identical token IDs."""

    if draft.tokenizer.get_vocab() != target.tokenizer.get_vocab():
        raise ValueError("Draft and target tokenizers have different vocabularies.")

    draft_added = getattr(draft.tokenizer, "get_added_vocab", lambda: {})()
    target_added = getattr(target.tokenizer, "get_added_vocab", lambda: {})()
    if draft_added != target_added:
        raise ValueError("Draft and target tokenizers have different added tokens.")

    token_attributes = [
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
    ]
    for attribute in token_attributes:
        draft_id = getattr(draft.tokenizer, attribute, None)
        target_id = getattr(target.tokenizer, attribute, None)
        if draft_id != target_id:
            raise ValueError(
                f"Tokenizer mismatch for {attribute}: {draft_id} != {target_id}"
            )


def load_speculative_models(
    draft_model_name: str | Path,
    target_model_name: str | Path,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    *,
    draft_adapter_path: str | Path | None = None,
    target_adapter_path: str | Path | None = None,
    draft_load_in_4bit: bool = False,
    target_load_in_4bit: bool = False,
) -> SpeculativeModelBundle:
    """Load and validate a draft/target pair for speculative decoding."""

    draft = load_causal_lm(
        model_name=draft_model_name,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        adapter_path=draft_adapter_path,
        load_in_4bit=draft_load_in_4bit,
    )

    target = load_causal_lm(
        model_name=target_model_name,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        adapter_path=target_adapter_path,
        load_in_4bit=target_load_in_4bit,
    )

    validate_tokenizer_compatibility(draft=draft, target=target)

    if draft.device != target.device:
        raise ValueError(
            "The current speculative decoder requires draft and target models "
            f"on the same device, got {draft.device} and {target.device}."
        )

    return SpeculativeModelBundle(draft=draft, target=target)


def get_parameter_count(model: PreTrainedModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def print_model_info(bundle: ModelBundle) -> None:
    """Print a compact model summary."""

    parameter_count = get_parameter_count(bundle.model)

    print("\n=== Model Information ===")
    print(f"Model: {bundle.model_name}")
    if bundle.adapter_name:
        print(f"Adapter: {bundle.adapter_name}")
    print(f"Class: {bundle.model.__class__.__name__}")
    print(f"Device: {bundle.device}")
    print(f"Dtype: {bundle.dtype}")
    print(f"Quantized: {bundle.quantized}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Vocabulary: {len(bundle.tokenizer):,}")
