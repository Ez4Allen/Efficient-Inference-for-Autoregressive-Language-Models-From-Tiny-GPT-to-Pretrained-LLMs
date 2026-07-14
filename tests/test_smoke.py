from pathlib import Path

import torch

from src.models.loader import load_speculative_models
from src.inference.autoregressive import greedy_decode
from src.inference.speculative import greedy_speculative_decode


PROJECT_ROOT = Path("/content/llm_project")

DRAFT_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "gpt2"
TARGET_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "gpt2-medium"


def check_local_checkpoint(model_path: Path) -> None:
    """
    Confirm that a local Hugging Face checkpoint contains
    the minimum required configuration, tokenizer, and weight files.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {model_path}"
        )

    required_config_files = [
        model_path / "config.json",
        model_path / "tokenizer_config.json",
    ]

    for file_path in required_config_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing required checkpoint file: {file_path}"
            )

    weight_files = (
        list(model_path.glob("*.safetensors"))
        + list(model_path.glob("*.bin"))
    )

    if not weight_files:
        raise FileNotFoundError(
            f"No model weight file found in: {model_path}"
        )

    print(f"Checkpoint valid: {model_path}")

    for weight_file in weight_files:
        size_mb = weight_file.stat().st_size / 1024**2
        print(
            f"  Weight file: {weight_file.name} "
            f"({size_mb:.2f} MB)"
        )


def find_first_mismatch(
    baseline_ids: torch.Tensor,
    speculative_ids: torch.Tensor,
) -> int | None:
    """
    Return the first token position where the two outputs differ.
    """

    min_length = min(
        baseline_ids.shape[1],
        speculative_ids.shape[1],
    )

    for index in range(min_length):
        if baseline_ids[0, index].item() != speculative_ids[0, index].item():
            return index

    if baseline_ids.shape[1] != speculative_ids.shape[1]:
        return min_length

    return None


def main() -> None:
    print("=" * 70)
    print("LOCAL CHECKPOINT TEST")
    print("=" * 70)

    check_local_checkpoint(DRAFT_MODEL_PATH)
    check_local_checkpoint(TARGET_MODEL_PATH)

    print()
    print("Loading models from local checkpoints...")

    models = load_speculative_models(
        draft_model_name=str(DRAFT_MODEL_PATH),
        target_model_name=str(TARGET_MODEL_PATH),
        dtype="float32",
    )

    draft_model = models.draft.model
    target_model = models.target.model
    tokenizer = models.target.tokenizer
    device = models.target.device

    print("Draft loaded from:", models.draft.model_name)
    print("Target loaded from:", models.target.model_name)
    print("Device:", device)
    print("Dtype:", models.target.dtype)

    prompt = (
        "Speculative decoding improves autoregressive language model "
        "inference by"
    )

    max_new_tokens = 40
    draft_tokens_per_round = 4

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)

    print()
    print("Prompt:", prompt)
    print("Prompt token count:", input_ids.shape[1])
    print("Max new tokens:", max_new_tokens)
    print("Draft tokens per round:", draft_tokens_per_round)

    print()
    print("Running warm-up...")

    _ = greedy_decode(
        model=target_model,
        input_ids=input_ids,
        max_new_tokens=5,
        eos_token_id=tokenizer.eos_token_id,
    )

    _ = greedy_speculative_decode(
        draft_model=draft_model,
        target_model=target_model,
        input_ids=input_ids,
        max_new_tokens=5,
        draft_tokens_per_round=draft_tokens_per_round,
        eos_token_id=tokenizer.eos_token_id,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("Warm-up complete.")

    print()
    print("Running target-only autoregressive baseline...")

    baseline = greedy_decode(
        model=target_model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )

    print("Running speculative decoding...")

    speculative = greedy_speculative_decode(
        draft_model=draft_model,
        target_model=target_model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        draft_tokens_per_round=draft_tokens_per_round,
        eos_token_id=tokenizer.eos_token_id,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    outputs_match = torch.equal(
        baseline.generated_token_ids,
        speculative.generated_token_ids,
    )

    baseline_text = tokenizer.decode(
        baseline.generated_token_ids[0],
        skip_special_tokens=True,
    )

    speculative_text = tokenizer.decode(
        speculative.generated_token_ids[0],
        skip_special_tokens=True,
    )

    baseline_time = baseline.total_time_seconds
    speculative_time = speculative.total_time_seconds

    speedup = (
        baseline_time / speculative_time
        if speculative_time > 0
        else float("inf")
    )

    baseline_tokens = baseline.generated_token_ids.shape[1]
    speculative_tokens = speculative.generated_token_ids.shape[1]

    baseline_throughput = (
        baseline_tokens / baseline_time
        if baseline_time > 0
        else 0.0
    )

    speculative_throughput = (
        speculative_tokens / speculative_time
        if speculative_time > 0
        else 0.0
    )

    average_accepted_per_round = (
        speculative.accepted_draft_tokens
        / speculative.speculative_rounds
        if speculative.speculative_rounds > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("CORRECTNESS")
    print("=" * 70)

    print("Outputs match:", outputs_match)

    print()
    print("Baseline generated text:")
    print(baseline_text)

    print()
    print("Speculative generated text:")
    print(speculative_text)

    print()
    print("=" * 70)
    print("PRELIMINARY PERFORMANCE")
    print("=" * 70)

    print(f"Baseline total time:        {baseline_time:.6f} s")
    print(f"Speculative total time:     {speculative_time:.6f} s")
    print(f"Preliminary speedup:        {speedup:.3f}x")

    print()
    print(f"Baseline throughput:        {baseline_throughput:.2f} tok/s")
    print(f"Speculative throughput:     {speculative_throughput:.2f} tok/s")
    print(
        "Target prefill time:       "
        f"{speculative.target_prefill_time_seconds * 1000:.3f} ms"
    )

    print()
    print("=" * 70)
    print("SPECULATIVE STATISTICS")
    print("=" * 70)

    print(f"Generated tokens:           {speculative_tokens}")
    print(f"Draft forward calls:        {speculative.draft_forward_calls}")
    print(f"Target forward calls:       {speculative.target_forward_calls}")
    print(f"Speculative rounds:         {speculative.speculative_rounds}")
    print(f"Proposed tokens:            {speculative.proposed_tokens}")
    print(
        f"Accepted draft tokens:      "
        f"{speculative.accepted_draft_tokens}"
    )
    print(
        f"Acceptance rate:            "
        f"{speculative.acceptance_rate:.2%}"
    )
    print(
        f"Average accepted per round: "
        f"{average_accepted_per_round:.3f}"
    )

    if not outputs_match:
        mismatch_index = find_first_mismatch(
            baseline.generated_token_ids,
            speculative.generated_token_ids,
        )

        print()
        print("=" * 70)
        print("MISMATCH DEBUG")
        print("=" * 70)

        print("First mismatch index:", mismatch_index)

        baseline_ids = baseline.generated_token_ids[0].tolist()
        speculative_ids = speculative.generated_token_ids[0].tolist()

        print("Baseline token IDs:")
        print(baseline_ids)

        print("Speculative token IDs:")
        print(speculative_ids)

        if mismatch_index is not None:
            if mismatch_index < len(baseline_ids):
                baseline_token = tokenizer.decode(
                    [baseline_ids[mismatch_index]]
                )
                print(
                    "Baseline mismatch token:",
                    repr(baseline_token),
                )

            if mismatch_index < len(speculative_ids):
                speculative_token = tokenizer.decode(
                    [speculative_ids[mismatch_index]]
                )
                print(
                    "Speculative mismatch token:",
                    repr(speculative_token),
                )

        raise RuntimeError(
            "Speculative output does not match target-only baseline."
        )

    print()
    print("Local checkpoint loading passed.")
    print("Speculative correctness test passed.")


if __name__ == "__main__":
    main()