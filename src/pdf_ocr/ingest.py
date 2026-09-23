"""Turn raw training material into page images waiting for autolabeling.

    training_data/sources/**   PDFs and images (manual drops, crawler output)
        -> training_data/unlabeled/<source path>__<stem>-pNNN.png

No detection happens here; training_project/scripts/autolabel.py picks the
pages up from unlabeled/. Every ingested file is recorded by content hash in
training_data/ingest_manifest.jsonl, so re-running only processes new files
and each page can be traced back to where it came from, including the
crawler's license tier (free/restricted) when the file came from a crawl.
"""

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pdf_ocr.sources import collect_sources, prepare_document

MANIFEST_NAME = "ingest_manifest.jsonl"


def run(sources_dir: Path, unlabeled_dir: Path, dpi: int = 200, max_pages: int = 30) -> dict:
    sources_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = unlabeled_dir.parent / MANIFEST_NAME
    seen = _load_seen_hashes(manifest_path)

    summary = {"ingested": 0, "already_ingested": 0, "pages": 0}
    for source in collect_sources(sources_dir, recursive=True):
        digest = _sha256(source)
        if digest in seen:
            summary["already_ingested"] += 1
            continue

        prefix = _prefix_for(source, sources_dir)
        pages = _render_to_unlabeled(source, unlabeled_dir, prefix, dpi, max_pages)

        entry = {
            "source": str(source.relative_to(sources_dir)),
            "sha256": digest,
            "tier": _crawl_tier(source),
            "pages": pages,
            "dpi": dpi if source.suffix.lower() == ".pdf" else None,
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        seen.add(digest)

        summary["ingested"] += 1
        summary["pages"] += len(pages)
        print(f"{entry['source']}: {len(pages)} page(s)")

    return summary


def _prefix_for(source: Path, sources_dir: Path) -> str:
    """sources/manual/foo/bar.pdf -> 'manual__foo__bar' (unique per source file)."""
    rel = source.relative_to(sources_dir).with_suffix("")
    return "__".join(rel.parts)


def _crawl_tier(source: Path) -> str | None:
    """License tier from the crawler's sources.jsonl next to the file (None for manual drops)."""
    log = source.parent / "sources.jsonl"
    if not log.exists():
        return None
    for line in log.read_text().splitlines():
        record = json.loads(line) if line.strip() else {}
        if record.get("file") == source.name:
            return record.get("tier")
    return None


def _render_to_unlabeled(
    source: Path, unlabeled_dir: Path, prefix: str, dpi: int, max_pages: int
) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        doc = prepare_document(source, Path(tmp), dpi, max_pages)
        names = []
        for page in doc.pages:
            name = f"{prefix}-p{page.number:03d}.png"
            shutil.move(page.image_path, unlabeled_dir / name)
            names.append(name)
    return names


def _load_seen_hashes(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    with manifest_path.open(encoding="utf-8") as f:
        return {json.loads(line)["sha256"] for line in f if line.strip()}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
