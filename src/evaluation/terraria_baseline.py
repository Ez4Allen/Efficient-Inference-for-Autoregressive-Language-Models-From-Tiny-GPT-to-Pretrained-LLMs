
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from src.models.loader import ModelBundle

from src.utils.paths import RESULTS_ROOT, TERRARIA_DATA_ROOT


DEFAULT_EVAL_PATH = TERRARIA_DATA_ROOT / "terraria_eval.jsonl"

DEFAULT_OUTPUT_PATH = (
    RESULTS_ROOT / "terraria" / "qwen3_4b_baseline.jsonl"
)


def load_jsonl(path: str | Path) -> list[dict]:
    """
    Load all records from a JSONL file.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {path}"
        )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {path}"
                ) from error

            records.append(record)

    return records


def load_completed_ids(path: str | Path) -> set[str]:
    """
    Read IDs that are already saved in the result file.
    """

    path = Path(path)

    if not path.exists():
        return set()

    records = load_jsonl(path)

    return {
        record["id"]
        for record in records
        if "id" in record
    }


@torch.inference_mode()
def generate_baseline_answer(
    bundle: ModelBundle,
    system_prompt: str,
    question: str,
    max_new_tokens: int = 512,
) -> dict:
    """
    Generate one deterministic answer and measure performance.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    inputs = bundle.tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(bundle.device)

    prompt_tokens = inputs["input_ids"].shape[1]

    if bundle.device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    output_ids = bundle.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=bundle.tokenizer.pad_token_id,
        eos_token_id=bundle.tokenizer.eos_token_id,
    )

    if bundle.device.type == "cuda":
        torch.cuda.synchronize()

    latency_seconds = time.perf_counter() - start_time

    generated_ids = output_ids[0, prompt_tokens:]
    output_tokens = generated_ids.shape[0]

    answer = bundle.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    tokens_per_second = (
        output_tokens / latency_seconds
        if latency_seconds > 0
        else 0.0
    )

    return {
        "model_answer": answer,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": latency_seconds,
        "tokens_per_second": tokens_per_second,
    }


def run_baseline_evaluation(
    bundle: ModelBundle,
    eval_path: str | Path = DEFAULT_EVAL_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    max_new_tokens: int = 512,
    resume: bool = True,
) -> None:
    """
    Run the model on all Terraria evaluation questions.

    Results are written after every question so progress is preserved
    if the Colab session is interrupted.
    """

    eval_path = Path(eval_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    eval_records = load_jsonl(eval_path)

    if not resume and output_path.exists():
        output_path.unlink()

    completed_ids = (
        load_completed_ids(output_path)
        if resume
        else set()
    )

    pending_records = [
        record
        for record in eval_records
        if record["id"] not in completed_ids
    ]

    print("=== Terraria Baseline Evaluation ===")
    print(f"Total questions: {len(eval_records)}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Pending questions: {len(pending_records)}")
    print(f"Output file: {output_path}")
    print()

    bundle.model.eval()

    for index, record in enumerate(
        pending_records,
        start=1,
    ):
        record_id = record["id"]
        question = record["question"]
        system_prompt = record["system_prompt"]

        print(
            f"[{index}/{len(pending_records)}] "
            f"{record_id}"
        )
        print(f"Question: {question}")

        generation = generate_baseline_answer(
            bundle=bundle,
            system_prompt=system_prompt,
            question=question,
            max_new_tokens=max_new_tokens,
        )

        result_record = {
            **record,
            "baseline_model": bundle.model_name,
            "baseline_device": str(bundle.device),
            "baseline_dtype": str(bundle.dtype),
            **generation,
        }

        with output_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    result_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        print(
            f"Output tokens: "
            f"{generation['output_tokens']}"
        )
        print(
            f"Latency: "
            f"{generation['latency_seconds']:.2f} seconds"
        )
        print(
            f"Generation speed: "
            f"{generation['tokens_per_second']:.2f} tokens/s"
        )
        print("-" * 80)

    print()
    print("Baseline evaluation completed.")
    print(f"Results saved to: {output_path}")
