# Proposal: TypeSafe AI (Jev) integration

> **Status (2026-09-23): item 3 (autolabeling pre-filter) is scaffolded** in
> `training_project/src/decisions.py` / `autolabeler.py`, using a threshold-based stub decider
> until a `TYPESAFE_API_KEY` exists — see that section below for the corrected design. Items
> 1, 2, 4, 5 are still proposal-only.

**Important correction (2026-09-23):** Jev's `state` input is **text/JSON only** — TypeSafe's
own model docs are explicit: "No image, audio, or video input" (current version 1.13). Jev
cannot look at a cropped image directly; every integration point below has to feed it
*structured evidence about* an image/document, not the pixels themselves. Section 3 was
originally written assuming visual input and has been corrected.

## What TypeSafe AI is

[TypeSafe AI](https://docs.typesafe.ai/introduction) is built around **Jev**, a "System One"
model: instead of generating free text that then has to be parsed/coerced into a usable data
structure, it returns typed, confidence-scored judgments directly. Three primitives:

- **`Choice`** — pick one option from a set of labeled criteria; returns the choice plus a
  confidence and a probability distribution over all options.
- **`Score`** — rate something against an ordered rubric; returns a numeric score plus
  confidence.
- **`Noul`** — evaluate a binary/truth statement; returns 0.0–1.0.

Multiple questions can be batched into a single `system_one()` call and answered in parallel.
Because it's a small, purpose-built model rather than a general LLM, it's positioned as
meaningfully cheaper and faster for this class of "make a judgment my code will branch on"
problem than routing the same decision through a full LLM call.

Python SDK: `pip install typesafe-sdk` (or `uv add typesafe-sdk`), requires Python ≥3.10
(this repo is already on 3.12). Auth via `TYPESAFE_API_KEY` env var — this repo already uses
direnv (`.envrc`), which is a natural place to source it from a local secrets file.

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient()
response = client.system_one(
    state=some_text_or_context,
    questions={
        "department": Choice(instructions="...", criteria={...}),
        "quality": Score(instructions="...", criteria=[...]),
        "is_urgent": Noul(instructions="..."),
    },
)
```

## Why it fits this pipeline

Several points in the planned pipeline (see [`roadmap.md`](./roadmap.md)) are exactly
"make a structured judgment, then branch" problems, not free-text generation — a good match
for Jev instead of spending a full LLM call on them.

## Concrete integration points

### 1. OCR/conversion quality gating (`src/main.py`)

After the docling conversion step, use `Score` (and/or `Noul`) to judge whether the extracted
markdown looks complete/well-structured (e.g. "does this look like a garbled/incomplete OCR
output vs. a clean structured datasheet"). If the score is low, fall back to `ocrmypdf` +
`pytesseract` before re-running the conversion, instead of always running the same fixed
pipeline regardless of input quality.

### 2. Datasheet classification

Use `Choice` to classify an incoming PDF (e.g. VCO datasheet vs. schematic vs. generic
component datasheet vs. something else) right after ingestion, and route it to the
appropriate downstream processing branch. This turns "analyse the structure of PDFs" (root
README step 2) into a dispatch step instead of one fixed pipeline for every PDF type.

### 3. Autolabeling pre-filter for YOLO training data — **scaffolded**

The root README's open TODO — "build an autolabeling image pipeline from the current trained
model" — is implemented in `training_project/src/`:

- `features.py` extracts cheap, non-ML evidence per YOLO candidate: confidence, box geometry
  (aspect ratio, area fraction), and classical-CV descriptors (edge density, Hough line count)
  that help distinguish line-art (schematics) from noise/photos — no vision model or LLM call.
- `decisions.py` sends that evidence as JSON `state` to Jev with a `Choice` question
  (`accept` / `flag_for_review` / `reject`), falling back to a `ThresholdStubDecider` (plain
  YOLO-confidence thresholds) when `TYPESAFE_API_KEY` isn't set, so the rest of the pipeline
  is testable without a TypeSafe account.
- `autolabeler.py` routes `accept` straight into `training_data/train/` (YOLO label format),
  and `flag_for_review` into a Label Studio pre-annotated task queue
  (`training_data/review_queue/label_studio_tasks.json` — full image + pre-drawn box, not an
  isolated crop, so a human reviewer has page context and can drag-correct) plus an
  `evidence_log.jsonl` audit trail for later threshold tuning.
- `scripts/autolabel.py` is the CLI entry point.

Not yet done: wiring a real `TYPESAFE_API_KEY` (still using the stub), and there's still no
bulk source of new unlabeled images to run this over at scale — see roadmap.md's "connect
YOLO to the PDF pipeline" step, which is what would actually feed this.

### 4. Pre-LLM routing gate (before the planned MCP hand-off)

Root README step 5 plans to "use a MCP to feed a LLM with the generated context." Before that
expensive call, use cheap Jev judgments to decide *whether* and *how* to invoke the LLM — e.g.
`is_urgent`/`needs_human_review` `Noul` checks, or a `Choice` over which extraction path/prompt
template to use. This keeps the expensive LLM call reserved for cases that actually need it.

### 5. Extracted table/field validation

Once component tables/fields are parsed out of a datasheet, use `Score` to rate confidence in
the parse (e.g. "does this look like a complete, correctly-aligned component table") and
decide whether to retry extraction with different settings (different OCR pass, different
table-detection heuristic) rather than silently emitting a possibly-broken table.

## If/when this moves past proposal stage

- Add `typesafe-sdk` to `pyproject.toml` dependencies.
- Store `TYPESAFE_API_KEY` via the existing `.envrc`/direnv setup (keep the key out of git).
- Suggested module location: a small `src/decisions.py` wrapping the specific typed questions
  this project needs (e.g. `classify_datasheet()`, `score_ocr_quality()`,
  `validate_autolabel()`), rather than sprinkling raw `TypeSafeClient` calls through the
  pipeline — keeps the question definitions in one reviewable place and makes them easy to
  swap out later if this evaluation doesn't pan out.
- Worth a small spike first: run a handful of real datasheet/YOLO-crop examples through the
  primitives above and eyeball the confidence scores before wiring anything into the main
  pipeline.
