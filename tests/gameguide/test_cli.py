from __future__ import annotations

from src.gameguide.cli import build_parser, parse_args


def test_cli_parser_exposes_multi_game_and_model_modes() -> None:
    args = parse_args(
        [
            "--game",
            "stardew",
            "--llm",
            "--engine",
            "speculative",
            "--season",
            "spring",
            "--day",
            "24",
            "Can",
            "I",
            "still",
            "plant",
            "Parsnip?",
        ]
    )
    assert args.game == "stardew"
    assert args.llm is True
    assert args.engine == "speculative"
    assert args.season == "spring"
    assert args.day == 24
    assert " ".join(args.question) == "Can I still plant Parsnip?"


def test_cli_help_mentions_grounded_assistant() -> None:
    help_text = build_parser().format_help()
    assert "evidence-grounded Qwen assistant" in help_text
    assert "--ungrounded" in help_text
