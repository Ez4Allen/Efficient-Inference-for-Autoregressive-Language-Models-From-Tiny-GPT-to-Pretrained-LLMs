
"""Public multi-game grounded language-model interfaces."""

__version__ = "1.0.0"

from .assistant import GameGuideAssistant
from .baselines import UngroundedQwenGenerator, build_ungrounded_messages
from .evidence_selection import (
    EvidenceSelectionConfig,
    EvidenceSelectionReport,
    PreparedEvidence,
    prepare_evidence,
)
from .generator import GameGuideQwenGenerator
from .plugin import GamePlugin
from .prompting import (
    PreparedPrompt,
    build_gameguide_messages,
    prepare_gameguide_prompt,
)
from .schemas import GameEvidence, GameGuideResult
from .validation import GroundedValidation, validate_gameguide_answer

__all__ = [
    "EvidenceSelectionConfig",
    "EvidenceSelectionReport",
    "GameEvidence",
    "GameGuideAssistant",
    "GameGuideQwenGenerator",
    "GameGuideResult",
    "GamePlugin",
    "GroundedValidation",
    "PreparedEvidence",
    "PreparedPrompt",
    "UngroundedQwenGenerator",
    "build_gameguide_messages",
    "build_ungrounded_messages",
    "prepare_evidence",
    "prepare_gameguide_prompt",
    "validate_gameguide_answer",
    "__version__",
]
