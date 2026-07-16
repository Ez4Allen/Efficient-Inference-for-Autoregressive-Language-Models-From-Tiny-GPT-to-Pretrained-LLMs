from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ALLOWED_ROLES = {"system", "user", "assistant"}
ALLOWED_SPLITS = {"train", "validation", "eval"}
RECOMMENDED_FIELDS = {
    "split",
    "domain",
    "category",
    "language",
    "required_facts",
    "forbidden_errors",
    "source_urls",
    "verified",
    "dataset_version",
}


@dataclass
class Issue:
    severity: str
    message: str
    path: str | None = None
    line: int | None = None
    record_id: str | None = None

    def format(self) -> str:
        location_parts: list[str] = []

        if self.path:
            location_parts.append(self.path)

        if self.line is not None:
            location_parts.append(f"line {self.line}")

        if self.record_id:
            location_parts.append(f"id={self.record_id}")

        location = (
            f" ({', '.join(location_parts)})"
            if location_parts
            else ""
        )

        return f"[{self.severity.upper()}]{location} {self.message}"


@dataclass
class DatasetRecord:
    split_name: str
    path: Path
    line_number: int
    data: dict[str, Any]
    record_id: str
    user_questions: list[str]


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    records: list[DatasetRecord] = field(default_factory=list)

    def error(self, message: str, **location: Any) -> None:
        self.issues.append(
            Issue("error", message, **location)
        )

    def warning(self, message: str, **location: Any) -> None:
        self.issues.append(
            Issue("warning", message, **location)
        )

    @property
    def errors(self) -> list[Issue]:
        return [
            issue for issue in self.issues
            if issue.severity == "error"
        ]

    @property
    def warnings(self) -> list[Issue]:
        return [
            issue for issue in self.issues
            if issue.severity == "warning"
        ]


def normalize_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def is_valid_url(value: str) -> bool:
    parsed = urlparse(value)
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
    )


def validate_string_list(
    value: Any,
    field_name: str,
    report: ValidationReport,
    *,
    path: Path,
    line_number: int,
    record_id: str,
    required: bool,
) -> None:
    if value is None:
        if required:
            report.error(
                f"Missing required field {field_name!r}",
                path=str(path),
                line=line_number,
                record_id=record_id,
            )
        return

    if not isinstance(value, list):
        report.error(
            f"{field_name!r} must be a list",
            path=str(path),
            line=line_number,
            record_id=record_id,
        )
        return

    if required and not value:
        report.error(
            f"{field_name!r} cannot be empty in strict mode",
            path=str(path),
            line=line_number,
            record_id=record_id,
        )

    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            report.error(
                f"{field_name}[{index}] must be a non-empty string",
                path=str(path),
                line=line_number,
                record_id=record_id,
            )


def validate_messages(
    messages: Any,
    report: ValidationReport,
    *,
    path: Path,
    line_number: int,
    record_id: str,
) -> list[str]:
    if not isinstance(messages, list):
        report.error(
            "'messages' must be a list",
            path=str(path),
            line=line_number,
            record_id=record_id,
        )
        return []

    if not messages:
        report.error(
            "'messages' cannot be empty",
            path=str(path),
            line=line_number,
            record_id=record_id,
        )
        return []

    cleaned_roles: list[str] = []
    user_questions: list[str] = []

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            report.error(
                f"messages[{index}] must be an object",
                path=str(path),
                line=line_number,
                record_id=record_id,
            )
            continue

        role = message.get("role")
        content = message.get("content")

        if role not in ALLOWED_ROLES:
            report.error(
                f"messages[{index}].role is invalid: {role!r}",
                path=str(path),
                line=line_number,
                record_id=record_id,
            )
            continue

        if not isinstance(content, str) or not content.strip():
            report.error(
                f"messages[{index}].content must be a non-empty string",
                path=str(path),
                line=line_number,
                record_id=record_id,
            )
            continue

        cleaned_roles.append(role)

        if role == "user":
            user_questions.append(content.strip())

    if not cleaned_roles:
        return user_questions

    for index, role in enumerate(cleaned_roles):
        if role == "system" and index != 0:
            report.error(
                "A system message may only appear first",
                path=str(path),
                line=line_number,
                record_id=record_id,
            )

    conversation_roles = cleaned_roles[1:] if cleaned_roles[0] == "system" else cleaned_roles

    if not conversation_roles:
        report.error(
            "Conversation contains no user/assistant messages",
            path=str(path),
            line=line_number,
            record_id=record_id,
        )
        return user_questions

    expected = "user"

    for index, role in enumerate(conversation_roles):
        if role != expected:
            report.error(
                f"Expected role {expected!r} at conversation position "
                f"{index}, found {role!r}",
                path=str(path),
                line=line_number,
                record_id=record_id,
            )
            break

        expected = "assistant" if expected == "user" else "user"

    if conversation_roles[-1] != "assistant":
        report.error(
            "Final message must be from assistant",
            path=str(path),
            line=line_number,
            record_id=record_id,
        )

    return user_questions


def validate_record(
    data: Any,
    split_name: str,
    path: Path,
    line_number: int,
    strict: bool,
    report: ValidationReport,
) -> DatasetRecord | None:
    if not isinstance(data, dict):
        report.error(
            "Each JSONL line must contain a JSON object",
            path=str(path),
            line=line_number,
        )
        return None

    raw_id = data.get("id")

    if not isinstance(raw_id, str) or not raw_id.strip():
        report.error(
            "Missing or invalid non-empty string field 'id'",
            path=str(path),
            line=line_number,
        )
        record_id = f"<missing-id:{line_number}>"
    else:
        record_id = raw_id.strip()

    if "messages" in data:
        user_questions = validate_messages(
            data.get("messages"),
            report,
            path=path,
            line_number=line_number,
            record_id=record_id,
        )
    elif split_name == "eval":
        question = data.get("question")
        reference_answer = data.get("reference_answer")

        user_questions = []

        if not isinstance(question, str) or not question.strip():
            report.error(
                "Eval record must contain a non-empty 'question' "
                "when 'messages' is absent",
                path=str(path),
                line=line_number,
                record_id=record_id,
            )
        else:
            user_questions.append(question.strip())

        if (
            not isinstance(reference_answer, str)
            or not reference_answer.strip()
        ):
            report.error(
                "Eval record must contain a non-empty "
                "'reference_answer' when 'messages' is absent",
                path=str(path),
                line=line_number,
                record_id=record_id,
            )

        system_prompt = data.get("system_prompt")

        if system_prompt is not None and (
            not isinstance(system_prompt, str)
            or not system_prompt.strip()
        ):
            report.error(
                "'system_prompt' must be a non-empty string when present",
                path=str(path),
                line=line_number,
                record_id=record_id,
            )
    else:
        report.error(
            "Training and validation records must contain 'messages'",
            path=str(path),
            line=line_number,
            record_id=record_id,
        )
        user_questions = []

    declared_split = data.get("split")

    if declared_split is not None:
        if declared_split not in ALLOWED_SPLITS:
            report.error(
                f"Invalid split value: {declared_split!r}",
                path=str(path),
                line=line_number,
                record_id=record_id,
            )
        elif declared_split != split_name:
            report.error(
                f"Record declares split={declared_split!r}, but the file "
                f"was provided as {split_name!r}",
                path=str(path),
                line=line_number,
                record_id=record_id,
            )
    elif strict:
        report.error(
            "Missing field 'split' in strict mode",
            path=str(path),
            line=line_number,
            record_id=record_id,
        )

    if strict:
        for field_name in sorted(RECOMMENDED_FIELDS):
            if field_name not in data:
                report.error(
                    f"Missing field {field_name!r} in strict mode",
                    path=str(path),
                    line=line_number,
                    record_id=record_id,
                )

    for field_name in ("domain", "category", "language", "dataset_version"):
        value = data.get(field_name)

        if value is not None and (
            not isinstance(value, str)
            or not value.strip()
        ):
            report.error(
                f"{field_name!r} must be a non-empty string",
                path=str(path),
                line=line_number,
                record_id=record_id,
            )

    validate_string_list(
        data.get("required_facts"),
        "required_facts",
        report,
        path=path,
        line_number=line_number,
        record_id=record_id,
        required=strict,
    )

    validate_string_list(
        data.get("forbidden_errors"),
        "forbidden_errors",
        report,
        path=path,
        line_number=line_number,
        record_id=record_id,
        required=strict,
    )

    source_urls = data.get("source_urls")

    validate_string_list(
        source_urls,
        "source_urls",
        report,
        path=path,
        line_number=line_number,
        record_id=record_id,
        required=strict,
    )

    if isinstance(source_urls, list):
        for index, url in enumerate(source_urls):
            if isinstance(url, str) and url.strip() and not is_valid_url(url):
                report.error(
                    f"source_urls[{index}] is not a valid HTTP(S) URL",
                    path=str(path),
                    line=line_number,
                    record_id=record_id,
                )

    verified = data.get("verified")

    if verified is not None and not isinstance(verified, bool):
        report.error(
            "'verified' must be a boolean",
            path=str(path),
            line=line_number,
            record_id=record_id,
        )

    if strict and verified is not True:
        report.error(
            "'verified' must be true in strict mode",
            path=str(path),
            line=line_number,
            record_id=record_id,
        )

    generation_method = data.get("generation_method")

    if generation_method is not None and generation_method not in {
        "human",
        "ai_assisted",
        "mixed",
    }:
        report.warning(
            "generation_method should normally be 'human', "
            "'ai_assisted', or 'mixed'",
            path=str(path),
            line=line_number,
            record_id=record_id,
        )

    return DatasetRecord(
        split_name=split_name,
        path=path,
        line_number=line_number,
        data=data,
        record_id=record_id,
        user_questions=user_questions,
    )


def load_and_validate_file(
    split_name: str,
    path: Path,
    strict: bool,
    report: ValidationReport,
) -> None:
    if not path.exists():
        report.error(
            "File does not exist",
            path=str(path),
        )
        return

    if not path.is_file():
        report.error(
            "Path is not a file",
            path=str(path),
        )
        return

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line:
                report.warning(
                    "Blank line ignored",
                    path=str(path),
                    line=line_number,
                )
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                report.error(
                    f"Invalid JSON: {error.msg}",
                    path=str(path),
                    line=line_number,
                )
                continue

            record = validate_record(
                data=data,
                split_name=split_name,
                path=path,
                line_number=line_number,
                strict=strict,
                report=report,
            )

            if record is not None:
                report.records.append(record)


def check_duplicate_ids(report: ValidationReport) -> None:
    seen: dict[str, DatasetRecord] = {}

    for record in report.records:
        previous = seen.get(record.record_id)

        if previous is not None:
            report.error(
                "Duplicate record ID; first occurrence is "
                f"{previous.path}:{previous.line_number}",
                path=str(record.path),
                line=record.line_number,
                record_id=record.record_id,
            )
        else:
            seen[record.record_id] = record


def build_question_entries(
    report: ValidationReport,
) -> list[tuple[str, str, DatasetRecord]]:
    entries: list[tuple[str, str, DatasetRecord]] = []

    for record in report.records:
        for question in record.user_questions:
            normalized = normalize_text(question)

            if normalized:
                entries.append(
                    (normalized, question, record)
                )

    return entries


def check_exact_question_duplicates(
    report: ValidationReport,
) -> None:
    seen: dict[str, tuple[str, DatasetRecord]] = {}

    for normalized, original, record in build_question_entries(report):
        previous = seen.get(normalized)

        if previous is None:
            seen[normalized] = (original, record)
            continue

        previous_original, previous_record = previous

        if previous_record.split_name == record.split_name:
            report.warning(
                "Duplicate user question within the same split; "
                f"first occurrence is {previous_record.path}:"
                f"{previous_record.line_number}",
                path=str(record.path),
                line=record.line_number,
                record_id=record.record_id,
            )
        else:
            report.error(
                "Exact user-question overlap across splits: "
                f"{previous_record.split_name} and {record.split_name}. "
                f"First occurrence: {previous_record.path}:"
                f"{previous_record.line_number}",
                path=str(record.path),
                line=record.line_number,
                record_id=record.record_id,
            )


def check_fuzzy_cross_split_overlap(
    report: ValidationReport,
    threshold: float,
) -> None:
    entries = build_question_entries(report)

    for left_index in range(len(entries)):
        left_normalized, left_original, left_record = entries[left_index]

        for right_index in range(left_index + 1, len(entries)):
            right_normalized, right_original, right_record = entries[right_index]

            if left_record.split_name == right_record.split_name:
                continue

            if left_normalized == right_normalized:
                continue

            # Skip very short prompts because fuzzy ratios are noisy.
            if min(len(left_normalized), len(right_normalized)) < 20:
                continue

            ratio = SequenceMatcher(
                None,
                left_normalized,
                right_normalized,
            ).ratio()

            if ratio >= threshold:
                report.warning(
                    "Possible paraphrase leakage across splits "
                    f"(similarity={ratio:.3f}). "
                    f"Other question: {left_original!r} at "
                    f"{left_record.path}:{left_record.line_number}",
                    path=str(right_record.path),
                    line=right_record.line_number,
                    record_id=right_record.record_id,
                )


def print_summary(
    report: ValidationReport,
    provided_files: dict[str, Path],
) -> None:
    print("=== SFT Dataset Validation ===")

    for split_name, path in provided_files.items():
        count = sum(
            record.split_name == split_name
            for record in report.records
        )
        print(f"{split_name:10s}: {count:5d} records  ({path})")

    print(f"errors    : {len(report.errors)}")
    print(f"warnings  : {len(report.warnings)}")

    if report.issues:
        print("\n=== Issues ===")

        for issue in report.issues:
            print(issue.format())
    else:
        print("\nNo issues found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate generic chat-style SFT JSONL files and "
            "check train/validation/eval leakage."
        )
    )

    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--eval", dest="eval_path", type=Path)

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Require verification metadata, sources, and "
            "verified=true."
        ),
    )

    parser.add_argument(
        "--skip-fuzzy",
        action="store_true",
        help="Skip fuzzy cross-split paraphrase checks.",
    )

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.90,
        help=(
            "Fuzzy similarity threshold between 0 and 1 "
            "(default: 0.90)."
        ),
    )

    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Return a nonzero exit code when warnings exist.",
    )

    args = parser.parse_args()

    if not any(
        (args.train, args.validation, args.eval_path)
    ):
        parser.error(
            "Provide at least one of --train, --validation, or --eval"
        )

    if not 0.0 <= args.similarity_threshold <= 1.0:
        parser.error(
            "--similarity-threshold must be between 0 and 1"
        )

    return args


def main() -> int:
    args = parse_args()
    report = ValidationReport()

    provided_files: dict[str, Path] = {}

    if args.train:
        provided_files["train"] = args.train

    if args.validation:
        provided_files["validation"] = args.validation

    if args.eval_path:
        provided_files["eval"] = args.eval_path

    for split_name, path in provided_files.items():
        load_and_validate_file(
            split_name=split_name,
            path=path,
            strict=args.strict,
            report=report,
        )

    check_duplicate_ids(report)
    check_exact_question_duplicates(report)

    if not args.skip_fuzzy:
        check_fuzzy_cross_split_overlap(
            report,
            threshold=args.similarity_threshold,
        )

    print_summary(
        report=report,
        provided_files=provided_files,
    )

    if report.errors:
        return 1

    if args.fail_on_warnings and report.warnings:
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
