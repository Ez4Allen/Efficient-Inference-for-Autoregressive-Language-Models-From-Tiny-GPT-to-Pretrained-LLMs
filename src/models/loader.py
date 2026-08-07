from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from src.models.tokenizer_contract import validate_model_tokenizer_contract
from src.utils.device import resolve_device
from src.utils.io import read_json
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


def _contains_tokenizer_files(directory: str | Path) -> bool:
    root = Path(directory)
    markers = (
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "spiece.model",
        "tokenizer.model",
    )
    return any((root / marker).exists() for marker in markers)


def _local_model_type(reference: str, is_local: bool) -> str | None:
    if not is_local:
        return None
    config_path = Path(reference) / "config.json"
    if not config_path.exists():
        return None
    payload = read_json(config_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Model config must be a JSON object: {config_path}")
    value = payload.get("model_type")
    return str(value) if value is not None else None


def load_causal_lm(
    model_name: str | Path = "openai-community/gpt2",
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    *,
    tokenizer_name: str | Path | None = None,
    adapter_path: str | Path | None = None,
    load_in_4bit: bool = False,
    attn_implementation: str | None = None,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> ModelBundle:
    """Load a Hugging Face causal LM or a local ``TinyQwenDraft`` checkpoint.

    ``tokenizer_name`` lets a custom draft checkpoint use the exact tokenizer
    of its target model.  The tokenizer mapping and special IDs are validated
    against metadata stored in the draft config before inference starts.
    """

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Model loading requires Transformers. Install project dependencies "
            "with `pip install -r requirements.txt`."
        ) from error

    resolved_model_name, model_is_local = _resolve_reference(model_name)
    model_type = _local_model_type(resolved_model_name, model_is_local)
    is_tiny_qwen_draft = model_type == "tiny_qwen_draft"

    tokenizer_reference: str | Path
    if tokenizer_name is not None:
        tokenizer_reference = tokenizer_name
    elif is_tiny_qwen_draft:
        draft_config = read_json(Path(resolved_model_name) / "config.json")
        recorded_tokenizer = (
            draft_config.get("tokenizer_name_or_path")
            if isinstance(draft_config, dict)
            else None
        )
        # Final custom checkpoints save an exact copy of the target tokenizer.
        # Prefer that self-contained local copy; fall back to the recorded target
        # reference for intermediate checkpoints that contain only weights/config.
        tokenizer_reference = (
            resolved_model_name
            if _contains_tokenizer_files(resolved_model_name)
            else recorded_tokenizer or model_name
        )
    else:
        tokenizer_reference = model_name

    resolved_tokenizer_name, tokenizer_is_local = _resolve_reference(
        tokenizer_reference
    )
    tokenizer_local_only = local_files_only or tokenizer_is_local
    model_local_only = local_files_only or model_is_local

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(resolved_device, dtype)

    if load_in_4bit and resolved_device.type != "cuda":
        raise RuntimeError("4-bit bitsandbytes inference requires a CUDA GPU.")
    if is_tiny_qwen_draft and load_in_4bit:
        raise ValueError(
            "TinyQwenDraft does not use bitsandbytes 4-bit loading; keep the "
            "small draft in BF16/FP16 instead."
        )
    if is_tiny_qwen_draft and adapter_path is not None:
        raise ValueError("PEFT adapters are not supported for TinyQwenDraft checkpoints.")
    if (
        is_tiny_qwen_draft
        and attn_implementation is not None
        and str(attn_implementation).strip().casefold() != "sdpa"
    ):
        raise ValueError(
            "TinyQwenDraft uses PyTorch SDPA internally; its "
            "attn_implementation may be omitted or set to 'sdpa'."
        )

    print(f"Loading model: {resolved_model_name}")
    print(f"Tokenizer: {resolved_tokenizer_name}")
    print(f"Device: {resolved_device}")
    print(f"Dtype: {resolved_dtype}")
    if load_in_4bit:
        print("Quantization: 4-bit NF4")
    if attn_implementation is not None:
        print(f"Attention implementation: {attn_implementation}")

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_tokenizer_name,
        trust_remote_code=trust_remote_code,
        local_files_only=tokenizer_local_only,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a PAD token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    resolved_adapter: str | None = None
    if is_tiny_qwen_draft:
        from src.models.tiny_qwen_draft import TinyQwenDraft

        model = TinyQwenDraft.from_pretrained(
            resolved_model_name,
            map_location="cpu",
        )
        model.to(device=resolved_device, dtype=resolved_dtype)
        validate_model_tokenizer_contract(model, tokenizer)
    else:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise RuntimeError(
                "Model loading requires Transformers. Install project dependencies "
                "with `pip install -r requirements.txt`."
            ) from error

        model_kwargs: dict[str, Any] = {
            "dtype": resolved_dtype,
            "trust_remote_code": trust_remote_code,
            "local_files_only": model_local_only,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = str(attn_implementation)

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
    draft_tokenizer_name: str | Path | None = None,
    target_tokenizer_name: str | Path | None = None,
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
        tokenizer_name=draft_tokenizer_name,
        adapter_path=draft_adapter_path,
        load_in_4bit=draft_load_in_4bit,
    )

    target = load_causal_lm(
        model_name=target_model_name,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        tokenizer_name=target_tokenizer_name,
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
