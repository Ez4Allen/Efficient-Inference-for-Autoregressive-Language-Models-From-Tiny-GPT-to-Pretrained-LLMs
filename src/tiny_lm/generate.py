import argparse
import torch

from src.tiny_lm.model import TinyGPT
from src.tiny_lm.tokenizer import CharTokenizer


def load_model_and_tokenizer(checkpoint_path, tokenizer_path, device):
    """
    Load a trained TinyGPT checkpoint and its tokenizer.
    """

    tokenizer = CharTokenizer.load(tokenizer_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = TinyGPT(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, config


def generate_text(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=500,
    temperature=0.8,
):
    """
    Generate text from a prompt using the trained TinyGPT model.
    """

    input_ids = tokenizer.encode(prompt)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    generated_text = tokenizer.decode(output_ids[0].tolist())
    return generated_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained tiny GPT-style language model."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/tiny_gpt_shakespeare/model.pt",
        help="Path to the trained model checkpoint.",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="results/tiny_gpt_shakespeare/tokenizer.json",
        help="Path to the saved tokenizer JSON file.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="First Citizen:",
        help="Prompt text used to start generation.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Number of new tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Lower is more conservative; higher is more random.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: cuda, cpu, or leave empty for auto-detection.",
    )

    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Using device:", device)

    model, tokenizer, config = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=device,
    )

    print("Loaded checkpoint:", args.checkpoint)
    print("Loaded tokenizer:", args.tokenizer)
    print("Model config:", config)
    print("-" * 80)

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(generated_text)


if __name__ == "__main__":
    main()