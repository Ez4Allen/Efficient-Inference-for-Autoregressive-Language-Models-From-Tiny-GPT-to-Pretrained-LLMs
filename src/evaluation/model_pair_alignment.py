"""Token-distribution analysis for draft/target language-model pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class ModelPairAlignmentReport:
    positions: int
    top1_agreement: float
    mean_topk_overlap: float
    mean_draft_entropy: float
    mean_target_entropy: float
    mean_js_divergence: float
    target_token_logprob_draft: float
    target_token_logprob_target: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def js_divergence_from_logits(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    first_log = F.log_softmax(first.float(), dim=-1)
    second_log = F.log_softmax(second.float(), dim=-1)
    first_prob = first_log.exp()
    second_prob = second_log.exp()
    mixture = 0.5 * (first_prob + second_prob)
    mixture_log = mixture.clamp_min(1e-30).log()
    first_kl = (first_prob * (first_log - mixture_log)).sum(dim=-1)
    second_kl = (second_prob * (second_log - mixture_log)).sum(dim=-1)
    return 0.5 * (first_kl + second_kl)


def topk_overlap(first: torch.Tensor, second: torch.Tensor, *, k: int = 5) -> torch.Tensor:
    k = max(1, min(int(k), first.shape[-1], second.shape[-1]))
    first_top = torch.topk(first, k=k, dim=-1).indices
    second_top = torch.topk(second, k=k, dim=-1).indices
    matches = (first_top.unsqueeze(-1) == second_top.unsqueeze(-2)).any(dim=-1)
    return matches.float().sum(dim=-1) / float(k)


def analyze_model_pair_logits(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    *,
    target_token_ids: torch.Tensor | None = None,
    top_k: int = 5,
) -> ModelPairAlignmentReport:
    if draft_logits.shape != target_logits.shape:
        raise ValueError(
            f"Draft and target logits must have the same shape; got "
            f"{tuple(draft_logits.shape)} and {tuple(target_logits.shape)}."
        )
    if draft_logits.ndim != 3:
        raise ValueError("Expected logits with shape [batch, positions, vocabulary].")
    positions = draft_logits.shape[0] * draft_logits.shape[1]
    if positions == 0:
        raise ValueError("Cannot analyze an empty sequence.")
    draft_argmax = draft_logits.argmax(dim=-1)
    target_argmax = target_logits.argmax(dim=-1)
    top1 = (draft_argmax == target_argmax).float().mean().item()
    overlap = topk_overlap(draft_logits, target_logits, k=top_k).mean().item()
    draft_entropy = entropy_from_logits(draft_logits).mean().item()
    target_entropy = entropy_from_logits(target_logits).mean().item()
    js = js_divergence_from_logits(draft_logits, target_logits).mean().item()
    draft_token_logprob = float("nan")
    target_token_logprob = float("nan")
    if target_token_ids is not None:
        if target_token_ids.shape != draft_logits.shape[:2]:
            raise ValueError("target_token_ids must have shape [batch, positions].")
        gather_ids = target_token_ids.unsqueeze(-1)
        draft_token_logprob = (
            F.log_softmax(draft_logits.float(), dim=-1).gather(-1, gather_ids).squeeze(-1).mean().item()
        )
        target_token_logprob = (
            F.log_softmax(target_logits.float(), dim=-1).gather(-1, gather_ids).squeeze(-1).mean().item()
        )
    return ModelPairAlignmentReport(
        positions=int(positions),
        top1_agreement=float(top1),
        mean_topk_overlap=float(overlap),
        mean_draft_entropy=float(draft_entropy),
        mean_target_entropy=float(target_entropy),
        mean_js_divergence=float(js),
        target_token_logprob_draft=float(draft_token_logprob),
        target_token_logprob_target=float(target_token_logprob),
    )


@torch.inference_mode()
def analyze_model_pair_on_sequence(
    draft_model: Any,
    target_model: Any,
    input_ids: torch.Tensor,
    *,
    completion_start: int,
    top_k: int = 5,
) -> ModelPairAlignmentReport:
    if completion_start < 1 or completion_start >= input_ids.shape[1]:
        raise ValueError("completion_start must identify at least one completion token.")
    draft_output = draft_model(input_ids=input_ids, use_cache=False)
    target_output = target_model(input_ids=input_ids, use_cache=False)
    # The logit at position i predicts token i+1. To evaluate completion tokens
    # beginning at completion_start, use logits from completion_start-1 onward.
    draft_logits = draft_output.logits[:, completion_start - 1 : -1, :]
    target_logits = target_output.logits[:, completion_start - 1 : -1, :]
    token_ids = input_ids[:, completion_start:]
    return analyze_model_pair_logits(
        draft_logits,
        target_logits,
        target_token_ids=token_ids,
        top_k=top_k,
    )
