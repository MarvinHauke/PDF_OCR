"""Training-data crawler: `pdf-ocr crawl <source>`. See base.py for the rules all sources follow."""

from importlib import import_module

from pdf_ocr.crawl.base import MB, CrawlContext, CrawlStop, DiskBudget, Fetcher, LicensePolicy, load_settings
from pdf_ocr.training import DATASET_CONFIGS, dataset_config

SOURCES = {
    "wikimedia": "pdf_ocr.crawl.wikimedia",
    "wikimedia_blocks": "pdf_ocr.crawl.wikimedia",
    "archive_org": "pdf_ocr.crawl.archive_org",
    "kicad_github": "pdf_ocr.crawl.kicad_github",
    "urls": "pdf_ocr.crawl.url_list",
}


def run(
    source: str, limit: int | None = None, dry_run: bool = False, refresh_netlists: bool = False
) -> CrawlContext:
    settings = load_settings()
    dataset = settings[source]["dataset"]
    out_dir = dataset_config(dataset).SOURCES_PATH / "crawled" / source
    crawled_roots = [dataset_config(d).SOURCES_PATH / "crawled" for d in DATASET_CONFIGS]
    budget = DiskBudget(int(settings["limits"]["max_total_gb"] * 1024 * MB), crawled_roots)

    ctx = CrawlContext(
        name=source,
        settings=settings,
        fetcher=Fetcher(settings),
        policy=LicensePolicy(settings),
        budget=budget,
        out_dir=out_dir,
        limit=limit or settings["limits"]["per_run"][source],
        dry_run=dry_run,
    )
    print(
        f"{source} -> {out_dir} (dataset: {dataset}, limit {ctx.limit}, "
        f"disk {budget.used / MB:.0f}/{budget.max_bytes / MB:.0f} MB){' [dry run]' if dry_run else ''}"
    )
    try:
        module = import_module(SOURCES[source])
        if refresh_netlists:
            module.refresh_netlists(ctx)
        else:
            module.crawl(ctx)
    except CrawlStop as e:
        print(f"stopped: {e}")
        ctx.summary["stopped"] = 1
    return ctx
