from src.benchmark.runner import (
    print_benchmark_summary,
    run_autoregressive_benchmark,
)
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

    summary = run_autoregressive_benchmark(
        bundle=bundle,
        prompt="Speculative decoding improves inference by",
        max_new_tokens=20,
        warmup_runs=5,
        benchmark_runs=20,
    )

    print_benchmark_summary(summary)


if __name__ == "__main__":
    main()