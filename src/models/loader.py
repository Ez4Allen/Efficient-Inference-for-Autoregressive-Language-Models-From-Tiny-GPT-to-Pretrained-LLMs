
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class ModelBundle:
    """
    A container for the objects required during inference.
    """

    model_name: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    dtype: torch.dtype


def resolve_device(
    device: str | torch.device | None = None,
) -> torch.device:
    """
    Resolve the device used for inference.

    If no device is specified, CUDA is used when available;
    otherwise, CPU is used.
    """

    if device is not None:
        resolved_device = torch.device(device)

        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False."
            )

        return resolved_device

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def resolve_dtype(
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.dtype:
    """
    Resolve the numerical precision used to load the model.

    CUDA:
        BF16 when supported, otherwise FP16.

    CPU:
        FP32 for compatibility.
    """

    if dtype is not None:
        return dtype

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float16

    return torch.float32


def load_causal_lm(
    model_name: str = "openai-community/gpt2",
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = False,
) -> ModelBundle:
    """
    Load a Hugging Face tokenizer and causal language model.

    Args:
        model_name:
            Hugging Face model identifier.

        device:
            Device used for inference, such as "cuda" or "cpu".
            When omitted, the device is selected automatically.

        dtype:
            Model precision. When omitted, it is selected automatically.

        trust_remote_code:
            Whether custom code from the model repository may be executed.

    Returns:
        ModelBundle containing the model, tokenizer, device, and dtype.
    """

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(
        device=resolved_device,
        dtype=dtype,
    )

    print(f"Loading model: {model_name}")
    print(f"Using device: {resolved_device}")
    print(f"Using dtype: {resolved_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    # GPT-2 has no dedicated padding token.
    # Reusing EOS as PAD is sufficient for the current inference experiments.
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "The tokenizer has neither a pad token nor an EOS token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=resolved_dtype,
        trust_remote_code=trust_remote_code,
    )

    model.to(resolved_device)
    model.eval()

    return ModelBundle(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        dtype=resolved_dtype,
    )


def get_parameter_count(model: PreTrainedModel) -> int:
    """
    Return the total number of model parameters.
    """

    return sum(parameter.numel() for parameter in model.parameters())


def get_parameter_memory_bytes(model: PreTrainedModel) -> int:
    """
    Return the approximate memory occupied by model parameters.
    """

    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )


def print_model_info(bundle: ModelBundle) -> None:
    """
    Print basic information about the loaded model and tokenizer.
    """

    model = bundle.model
    tokenizer = bundle.tokenizer
    config = model.config

    total_parameters = get_parameter_count(model)
    parameter_memory = get_parameter_memory_bytes(model)

    number_of_layers = getattr(
        config,
        "num_hidden_layers",
        getattr(config, "n_layer", None),
    )

    number_of_heads = getattr(
        config,
        "num_attention_heads",
        getattr(config, "n_head", None),
    )

    hidden_size = getattr(
        config,
        "hidden_size",
        getattr(config, "n_embd", None),
    )

    max_positions = getattr(
        config,
        "max_position_embeddings",
        getattr(config, "n_positions", None),
    )

    print("\n=== Model Information ===")
    print(f"Model name: {bundle.model_name}")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Device: {bundle.device}")
    print(f"Dtype: {bundle.dtype}")
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"Total parameters: {total_parameters:,}")
    print(
        "Approximate parameter memory: "
        f"{parameter_memory / 1024**3:.3f} GiB"
    )

    if number_of_layers is not None:
        print(f"Number of layers: {number_of_layers}")

    if number_of_heads is not None:
        print(f"Number of attention heads: {number_of_heads}")

    if hidden_size is not None:
        print(f"Hidden size: {hidden_size}")

    if max_positions is not None:
        print(f"Maximum positions: {max_positions}")


def print_device_info() -> None:
    """
    Print the current CUDA environment.
    """

    print("=== Device Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("Device: CPU")
        return

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)

    print(f"GPU: {torch.cuda.get_device_name(device_index)}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"Total VRAM: {properties.total_memory / 1024**3:.2f} GiB")
