"""Load a TinyGPT checkpoint and generate text."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.models.tiny_lm.model import TinyGPT
from src.models.tiny_lm.tokenizer import CharTokenizer
from src.utils.paths import resolve_project_path
from src.utils.seed import set_global_seed


def load_model_and_tokenizer(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    device: str | torch.device,
) -> tuple[TinyGPT, CharTokenizer, dict[str, Any]]:
    tokenizer = CharTokenizer.load(resolve_project_path(tokenizer_path))
    checkpoint = torch.load(
        resolve_project_path(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]
    model = TinyGPT(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        d_ff=config.get("d_ff", 4 * config["n_embd"]),
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, config


def generate_text(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    device: str | torch.device,
    *,
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int | None = None,
    seed: int | None = None,
) -> str:
    if not prompt:
        raise ValueError("prompt cannot be empty.")
    input_ids = tokenizer.encode(prompt)
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    generator = None
    if seed is not None:
        set_global_seed(seed)
        generator = torch.Generator(device=torch.device(device).type)
        generator.manual_seed(seed)
    output_ids = model.generate(
        tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
    )
    return tokenizer.decode(output_ids[0].tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", default="First Citizen:")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint, args.tokenizer, device
    )
    print(f"Loaded model config: {config}")
    print(
        generate_text(
            model,
            tokenizer,
            args.prompt,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
