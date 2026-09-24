import json

import pypdfium2 as pdfium

from pdf_ocr import ingest


def _make_pdf(path, pages):
    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(100, 100)
    pdf.save(str(path))
    pdf.close()


def _manifest(tmp_path):
    return [json.loads(line) for line in (tmp_path / ingest.MANIFEST_NAME).read_text().splitlines()]


def test_refill_adds_missing_pages_but_not_deleted_ones(tmp_path):
    sources, unlabeled = tmp_path / "sources", tmp_path / "unlabeled"
    sources.mkdir()
    _make_pdf(sources / "doc.pdf", 10)

    first = ingest.run(sources, unlabeled, dpi=20, max_pages=3)
    assert first["pages"] == 3  # pages 1, 5, 10 (spread)
    (unlabeled / "doc-p005.png").unlink()  # removed by hand after review

    # without --refill, nothing happens
    assert ingest.run(sources, unlabeled, dpi=20, max_pages=0)["already_ingested"] == 1

    summary = ingest.run(sources, unlabeled, dpi=20, max_pages=0, refill=True)
    assert summary == {"ingested": 0, "already_ingested": 0, "refilled": 1, "pages": 7}
    assert not (unlabeled / "doc-p005.png").exists()
    [entry] = _manifest(tmp_path)
    assert entry["pages"] == [f"doc-p{n:03d}.png" for n in range(1, 11)]
    assert "refilled_at" in entry

    # a second refill finds nothing new
    again = ingest.run(sources, unlabeled, dpi=20, max_pages=0, refill=True)
    assert again["refilled"] == 0 and again["already_ingested"] == 1


def test_only_keeps_prefix_relative_to_sources(tmp_path):
    sources, unlabeled = tmp_path / "sources", tmp_path / "unlabeled"
    (sources / "manual").mkdir(parents=True)
    (sources / "crawled").mkdir()
    _make_pdf(sources / "manual" / "doc.pdf", 2)
    _make_pdf(sources / "crawled" / "other.pdf", 2)

    summary = ingest.run(sources, unlabeled, dpi=20, max_pages=0, only=sources / "manual")
    assert summary["ingested"] == 1
    assert sorted(p.name for p in unlabeled.iterdir()) == ["manual__doc-p001.png", "manual__doc-p002.png"]
    assert _manifest(tmp_path)[0]["source"] == "manual/doc.pdf"
