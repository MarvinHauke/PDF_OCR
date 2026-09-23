"""Curated URL list (e.g. datasheets): one URL per line, optionally followed by a license.

    https://example.com/part.pdf
    https://example.com/other.pdf CC-BY-4.0

Lines starting with # are ignored. A URL without a free license is only
downloaded if its host is in reviewed_hosts (crawl.yaml) -- add a host there
only after checking its terms. Blocked hosts are always refused.
"""

from pathlib import PurePosixPath
from urllib.parse import urlparse

from pdf_ocr.crawl.base import TRAINING_PROJECT, CrawlContext, host_of, safe_filename


def crawl(ctx: CrawlContext):
    list_path = TRAINING_PROJECT / ctx.settings["urls"]["file"]
    if not list_path.exists():
        print(f"  no URL list at {list_path}; create it with one URL per line")
        return
    max_mb = ctx.settings["limits"]["max_pdf_mb"]
    for line in list_path.read_text().splitlines():
        if ctx.done:
            return
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        url, _, license_name = line.partition(" ")
        name = PurePosixPath(urlparse(url).path).name or host_of(url)
        ctx.fetch_file(
            url,
            safe_filename(f"{host_of(url)}__{name}"),
            max_mb,
            license_name.strip() or None,
            {"source_page": url, "hint": None},
        )
