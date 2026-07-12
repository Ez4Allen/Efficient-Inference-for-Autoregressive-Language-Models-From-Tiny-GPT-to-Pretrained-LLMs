
from src.decoding.autoregressive import greedy_decode
from src.models.loader import (
    load_causal_lm,
    print_device_info,
    print_model_info,
)


def main() -> None:
    print_device_info()

    bundle = load_causal_lm(
        model_name="openai-community/gpt2",
    )

    print_model_info(bundle)

    prompt = "Speculative decoding improves inference by"

    encoded = bundle.tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(bundle.device)

    result = greedy_decode(
        model=bundle.model,
        input_ids=input_ids,
        max_new_tokens=20,
        eos_token_id=bundle.tokenizer.eos_token_id,
    )

    full_text = bundle.tokenizer.decode(
        result.output_ids[0],
        skip_special_tokens=True,
    )

    generated_text = bundle.tokenizer.decode(
        result.generated_token_ids[0],
        skip_special_tokens=True,
    )

    print("\n=== Generation Result ===")
    print(f"Prompt: {prompt}")
    print(f"Generated continuation: {generated_text!r}")
    print(f"Full text: {full_text}")
    print(
        "Generated token count: "
        f"{result.generated_token_ids.shape[1]}"
    )
    print(
        "Target forward calls: "
        f"{result.target_forward_calls}"
    )


if __name__ == "__main__":
    main()
