#!/usr/bin/env python3
"""Convenience wrapper around the unified GameGuideLM CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
command = [sys.executable, str(ROOT / "scripts" / "chat_gameguide.py"), "--game", "stardew", *sys.argv[1:]]
raise SystemExit(subprocess.call(command, cwd=ROOT))
