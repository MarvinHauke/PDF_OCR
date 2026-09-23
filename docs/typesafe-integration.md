# Proposal: TypeSafe AI (Jev) integration

> **Status: proposal only.** Nothing here has been implemented — no dependency added, no code
> written. This lays out concrete integration points to evaluate before committing to it.

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

### 3. Autolabeling pre-filter for YOLO training data

The root README's open TODO — "build an autolabeling image pipeline from the current trained
model" — is a natural fit for `Score`/`Noul` as a cheap sanity check: after the trained YOLO
model proposes a bounding box/label for a cropped region, ask Jev "does this crop plausibly
match the proposed label" before accepting it into the training set. This adds a
confidence-aware human-review queue (low-confidence proposals get flagged) without needing a
full LLM call per candidate label.

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
