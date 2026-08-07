#!/usr/bin/env python3
"""Build a self-contained static HTML dashboard for the Stardew release."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render_cards(values: dict[str, int]) -> str:
    return "".join(
        f'<div class="metric"><span class="metric-value">{value:,}</span><span class="metric-label">{html.escape(label)}</span></div>'
        for label, value in values.items()
    )


def render_bars(values: dict[str, int], *, total: int) -> str:
    rows = []
    for label, value in values.items():
        width = 0 if total == 0 else value / total * 100
        rows.append(
            '<div class="bar-row">'
            f'<div class="bar-label">{html.escape(label)}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%"></div></div>'
            f'<div class="bar-value">{value}</div>'
            '</div>'
        )
    return "".join(rows)


def render_examples(examples: list[dict]) -> str:
    cards = []
    for item in examples:
        evidence_items = []
        for source in item.get("evidence") or []:
            section = f" · {html.escape(source.get('section_title') or '')}" if source.get("section_title") else ""
            evidence_items.append(
                f'<li><strong>{html.escape(source["source_id"])}</strong> · '
                f'{html.escape(source["label"])}{section}</li>'
            )
        evidence = "".join(evidence_items)
        entity = (
            ' · Entity: <code>' + html.escape(str(item.get("entity"))) + '</code>'
            if item.get("entity") else ""
        )
        cards.append(
            f"""<article class="example" data-status="{html.escape(item['status'])}">
              <div class="example-head"><span>{html.escape(item['label'])}</span><span class="pill {html.escape(item['status'])}">{html.escape(item['status'])}</span></div>
              <div class="question">{html.escape(item['question'])}</div>
              <div class="answer">{html.escape(item['answer']).replace(chr(10), '<br>')}</div>
              <div class="intent">Intent: <code>{html.escape(item['intent'])}</code>{entity}</div>
              <details><summary>Evidence</summary><ul>{evidence or '<li>No evidence required for this status.</li>'}</ul></details>
            </article>"""
        )
    return "".join(cards)


def build(output: Path) -> Path:
    catalog = load(PROJECT_ROOT / "data" / "stardew" / "catalog" / "snapshot_manifest.json")
    evaluation = load(PROJECT_ROOT / "results" / "stardew" / "evaluation_summary.json")
    eval_manifest = load(PROJECT_ROOT / "data" / "stardew" / "evaluation" / "manifest_v1.json")
    training = load(PROJECT_ROOT / "data" / "stardew" / "training" / "manifest_v1.json")
    guide = load(PROJECT_ROOT / "data" / "stardew" / "guides" / "reports" / "build_report.json")
    release = load(PROJECT_ROOT / "results" / "stardew" / "release_validation.json")
    demo = load(PROJECT_ROOT / "results" / "stardew" / "demo_outputs.json")
    sft = load(PROJECT_ROOT / "data" / "stardew" / "sft" / "audit_report.json")

    metrics = {
        "Structured records": catalog["record_count"],
        "Acquisition relations": catalog["acquisition_relation_count"],
        "Guide chunks": guide["counts"]["chunks"],
        "Regression cases": eval_manifest["total_count"],
        "Grounded train records": training["train_count"],
        "Repository tests": 171,
    }
    record_bars = {name.title(): value for name, value in catalog["record_type_counts"].items()}
    status_bars = {name: value for name, value in eval_manifest["status_distribution"].items()}
    review_note = "Engineering validation passed; independent human source review remains required before the benchmark is labelled approved."

    page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GameGuideLM · Stardew Valley Release</title>
<style>
:root{{--bg:#07110d;--panel:#102019;--panel2:#14291f;--text:#eef7f0;--muted:#a8b9ad;--accent:#e7c265;--accent2:#68b984;--danger:#d66b68;--line:#284436;}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 20% 0,#183425 0,#07110d 42%);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.55}}
a{{color:var(--accent)}} .wrap{{max-width:1180px;margin:auto;padding:44px 26px 70px}} .hero{{display:grid;grid-template-columns:1.3fr .7fr;gap:30px;align-items:center;padding:34px;border:1px solid var(--line);border-radius:24px;background:linear-gradient(145deg,rgba(20,41,31,.94),rgba(8,19,14,.94));box-shadow:0 24px 80px rgba(0,0,0,.28)}}
.eyebrow{{text-transform:uppercase;letter-spacing:.18em;font-size:.75rem;color:var(--accent)}} h1{{font-size:clamp(2.5rem,6vw,5.5rem);line-height:.95;margin:.35rem 0 1rem;letter-spacing:-.06em}} h2{{font-size:1.85rem;margin:0 0 18px}} p{{color:var(--muted)}} .badge{{display:inline-flex;padding:7px 11px;border:1px solid #4b775e;border-radius:999px;color:#cfe9d7;background:#153725;font-size:.82rem}}
.pixel{{min-height:260px;border-radius:20px;display:grid;place-items:center;background:linear-gradient(#96c9e8 0 45%,#6aa64f 45% 70%,#6a4b31 70%);position:relative;overflow:hidden;border:6px solid #2b4939;image-rendering:pixelated}} .pixel:before{{content:"";width:92px;height:92px;background:#d7a54d;box-shadow:0 0 0 16px #70502c,0 76px 0 -25px #e9d7a7;clip-path:polygon(15% 0,85% 0,100% 35%,80% 100%,20% 100%,0 35%);position:absolute;bottom:45px}} .pixel:after{{content:"Grounded QA";position:absolute;bottom:12px;background:#112419;padding:5px 12px;border-radius:7px;font-family:monospace;color:#f4d47b}}
.metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:22px 0 48px}} .metric{{padding:18px 16px;border:1px solid var(--line);background:rgba(16,32,25,.85);border-radius:16px}} .metric-value{{display:block;font-size:1.7rem;font-weight:750;color:var(--accent)}} .metric-label{{font-size:.76rem;color:var(--muted)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0 48px}} .panel{{border:1px solid var(--line);border-radius:20px;padding:24px;background:rgba(16,32,25,.83)}} .bar-row{{display:grid;grid-template-columns:125px 1fr 42px;gap:10px;align-items:center;margin:12px 0}} .bar-label,.bar-value{{font-size:.83rem;color:var(--muted)}} .bar-track{{height:10px;border-radius:9px;background:#223a2f;overflow:hidden}} .bar-fill{{height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent));border-radius:9px}}
.pipeline{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;align-items:stretch}} .node{{padding:18px 12px;border:1px solid var(--line);border-radius:14px;background:var(--panel2);min-height:116px}} .node strong{{display:block;color:var(--accent);margin-bottom:7px}} .node span{{color:var(--muted);font-size:.82rem}} .arrow{{display:none}}
.controls{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}} button{{border:1px solid var(--line);background:#13281e;color:var(--text);padding:8px 12px;border-radius:999px;cursor:pointer}} button.active{{background:var(--accent);color:#172015;border-color:var(--accent)}} .examples{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}} .example{{border:1px solid var(--line);background:rgba(13,28,21,.9);padding:20px;border-radius:17px}} .example-head{{display:flex;justify-content:space-between;gap:14px;font-size:.86rem;color:var(--muted)}} .pill{{padding:2px 8px;border-radius:999px;font-size:.72rem;background:#264436}} .pill.not_found{{background:#4a2928}} .pill.needs_context,.pill.partial{{background:#4a3e23}} .question{{font-weight:720;font-size:1.05rem;margin:14px 0 9px}} .answer{{color:#d8e6dc;font-size:.91rem}} .intent{{margin-top:13px;color:var(--muted);font-size:.78rem}} code{{color:#f0cf78}} details{{margin-top:12px;color:var(--muted);font-size:.8rem}} .notice{{border-left:4px solid var(--accent);padding:14px 17px;background:#192d22;color:#dfe9e2;border-radius:0 12px 12px 0;margin:20px 0}} footer{{margin-top:55px;border-top:1px solid var(--line);padding-top:20px;color:var(--muted);font-size:.82rem}}
@media(max-width:900px){{.hero,.grid2{{grid-template-columns:1fr}}.metrics{{grid-template-columns:repeat(2,1fr)}}.pipeline{{grid-template-columns:1fr 1fr}}.examples{{grid-template-columns:1fr}}}} @media(max-width:520px){{.metrics{{grid-template-columns:1fr 1fr}}.pipeline{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main class="wrap">
<section class="hero"><div><div class="eyebrow">GameGuideLM · Course Release</div><h1>Stardew Valley<br>Grounded Assistant</h1><p>A bilingual, evidence-first game guide combining structured facts, offline guide retrieval, deterministic safety behavior, Qwen adaptation hooks, and speculative-decoding research infrastructure.</p><span class="badge">{html.escape(release['release_readiness'])}</span></div><div class="pixel" aria-label="Abstract pixel farm illustration"></div></section>
<section class="metrics">{render_cards(metrics)}</section>
<section class="panel"><h2>System architecture</h2><div class="pipeline"><div class="node"><strong>Question + state</strong><span>English/Chinese query, season, day, weather, time, location, bundle mode.</span></div><div class="node"><strong>Intent router</strong><span>Fact lookup, deadline calculation, acquisition, guide retrieval, or safe refusal.</span></div><div class="node"><strong>Dual retrieval</strong><span>505 structured records plus 100 compact guide chunks with provenance.</span></div><div class="node"><strong>Grounded answer</strong><span>Deterministic rendering today; Qwen + QLoRA is an optional generation layer.</span></div><div class="node"><strong>Fast decoding</strong><span>0.6B or custom TinyQwenDraft proposes tokens for 4B target verification.</span></div></div></section>
<section class="grid2"><div class="panel"><h2>Catalog composition</h2>{render_bars(record_bars,total=catalog['record_count'])}</div><div class="panel"><h2>Regression status distribution</h2>{render_bars(status_bars,total=eval_manifest['total_count'])}<p>Deterministic regression pass rate: <strong>{evaluation['pass_rate']*100:.0f}%</strong>. English/Chinese split: 50/50.</p></div></section>
<div class="notice"><strong>Review boundary.</strong> {html.escape(review_note)} The 1,262 legacy AI-assisted SFT candidates are preserved as <code>pending</code> and <code>verified=false</code>; they are not silently treated as reviewed training data.</div>
<section><h2>Live-path demonstration outputs</h2><div class="controls"><button class="active" data-filter="all">All</button><button data-filter="found">Found</button><button data-filter="needs_context">Needs context</button><button data-filter="partial">Partial</button><button data-filter="not_found">Not found</button></div><div class="examples">{render_examples(demo['examples'])}</div></section>
<footer>Snapshot target: Stardew Valley {html.escape(catalog['game_version'])} · Structured-source attribution: {html.escape(catalog['source_name'])} · Guide seed: project-authored summaries with explicit provenance flags · Formal human approval pending.</footer>
</main>
<script>
for (const button of document.querySelectorAll('button[data-filter]')) {{button.addEventListener('click',()=>{{document.querySelectorAll('button[data-filter]').forEach(b=>b.classList.remove('active'));button.classList.add('active');const f=button.dataset.filter;document.querySelectorAll('.example').forEach(card=>{{card.style.display=(f==='all'||card.dataset.status===f)?'block':'none';}});}});}}
</script></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "demo" / "stardew_showcase.html")
    args = parser.parse_args()
    print(build(args.output.expanduser().resolve()))


if __name__ == "__main__":
    main()
