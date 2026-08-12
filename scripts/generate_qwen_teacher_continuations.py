#!/usr/bin/env python3
"""Generate deterministic Qwen teacher continuations for a prompt pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.inference.autoregressive import greedy_decode
from src.inference.chat_runtime import _apply_chat_template
from src.models.loader import load_causal_lm
from src.models.tokenizer_contract import tokenizer_sha256
from src.utils.io import append_jsonl, read_jsonl, write_json


def eos_ids(bundle: Any) -> int | list[int] | None:
    value = getattr(bundle.model.generation_config, "eos_token_id", None)
    if value is None:
        value = bundle.tokenizer.eos_token_id
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from src.models.loader import resolve_dtype_name

    records = [
        row for row in read_jsonl(args.input) if str(row.get("split")) == args.split
    ]
    if args.limit is not None:
        records = records[: max(0, args.limit)]
    if not records:
        raise ValueError(f"No {args.split} prompt records found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.force:
        args.output.unlink(missing_ok=True)
    completed = {
        str(row.get("source_prompt_id"))
        for row in read_jsonl(args.output)
    } if args.output.exists() else set()

    bundle = load_causal_lm(
        args.model,
        tokenizer_name=args.tokenizer or args.model,
        device=args.device,
        dtype=resolve_dtype_name(args.dtype),
        local_files_only=args.local_files_only,
    )
    bundle.model.eval()
    teacher_fingerprint = tokenizer_sha256(bundle.tokenizer)
    generated = 0
    skipped = 0
    try:
        for index, record in enumerate(records, start=1):
            prompt_id = str(record["id"])
            if prompt_id in completed:
                skipped += 1
                continue
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages or messages[-1].get("role") != "user":
                raise ValueError(f"Prompt {prompt_id} must end with a user message.")
            input_ids = _apply_chat_template(
                bundle.tokenizer,
                messages,
                enable_thinking=False,
            ).to(bundle.device)
            output = greedy_decode(
                bundle.model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=eos_ids(bundle),
            )
            answer = bundle.tokenizer.decode(
                output.generated_token_ids[0],
                skip_special_tokens=True,
            ).strip()
            if not answer:
                skipped += 1
                continue
            output_record = {
                "id": f"teacher_{prompt_id}",
                "source_prompt_id": prompt_id,
                "split": args.split,
                "language": record.get("language"),
                "domain": record.get("domain"),
                "category": record.get("category"),
                "intent": record.get("intent"),
                "messages": [*messages, {"role": "assistant", "content": answer}],
                "target_source": "qwen3_0_6b_greedy_teacher",
                "teacher_model_name_or_path": args.model,
                "teacher_tokenizer_sha256": teacher_fingerprint,
                "prompt_tokens": int(input_ids.shape[1]),
                "generated_tokens": int(output.generated_token_ids.shape[1]),
                "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            }
            append_jsonl(args.output, output_record)
            completed.add(prompt_id)
            generated += 1
            print(
                json.dumps(
                    {
                        "index": index,
                        "id": prompt_id,
                        "prompt_tokens": input_ids.shape[1],
                        "generated_tokens": output.generated_token_ids.shape[1],
                    },
                    ensure_ascii=False,
                )
            )
    finally:
        del bundle.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_records = read_jsonl(args.output)
    manifest = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "split": args.split,
        "teacher_model_name_or_path": args.model,
        "teacher_tokenizer_sha256": teacher_fingerprint,
        "requested_records": len(records),
        "output_records": len(output_records),
        "generated_this_run": generated,
        "skipped_this_run": skipped,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
