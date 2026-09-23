"""archive.org: service manual PDFs found via the advanced search API.

Most items have no license metadata; they become `restricted` because
archive.org is in reviewed_hosts. A public-domain/CC item becomes `free`.
"""

import re

from pdf_ocr.crawl.base import CrawlContext, safe_filename

SEARCH = "https://archive.org/advancedsearch.php"
LICENSE_URLS = [  # licenseurl -> license name understood by LicensePolicy
    (r"publicdomain/mark", "PDM"),
    (r"publicdomain/zero", "CC0"),
    (r"licenses/(by(?:-[a-z]+)*)/([\d.]+)", None),  # -> cc-<parts>-<version>
]


def license_from_url(url: str | None) -> str | None:
    if not url:
        return None
    for pattern, name in LICENSE_URLS:
        match = re.search(pattern, url)
        if match:
            return name or f"cc-{match.group(1)}-{match.group(2)}"
    return url


def crawl(ctx: CrawlContext):
    limits = ctx.settings["limits"]
    for query in ctx.settings["archive_org"]["queries"]:
        print(f"- query: {query}")
        response = ctx.fetcher.get(
            SEARCH,
            api=True,
            params={"q": query, "fl[]": ["identifier", "title"], "rows": 200, "output": "json"},
        )
        for doc in response.json()["response"]["docs"]:
            if ctx.done:
                return
            identifier = doc["identifier"]
            metadata = ctx.fetcher.get(f"https://archive.org/metadata/{identifier}", api=True).json()
            pdfs = [f for f in metadata.get("files", []) if f.get("name", "").lower().endswith(".pdf")]
            if not pdfs:
                ctx.skip("no pdf", identifier)
                continue
            # Prefer the original upload over derived copies
            pdf = sorted(pdfs, key=lambda f: (f.get("source") != "original", int(f.get("size") or 0)))[0]
            size_mb = int(pdf.get("size") or 0) / 1024 / 1024
            if size_mb > limits["max_pdf_mb"]:
                ctx.skip(f"size: {size_mb:.0f} MB > {limits['max_pdf_mb']} MB", identifier)
                continue
            license_url = metadata.get("metadata", {}).get("licenseurl")
            ctx.fetch_file(
                f"https://archive.org/download/{identifier}/{pdf['name']}",
                safe_filename(f"{identifier}.pdf"),
                limits["max_pdf_mb"],
                license_from_url(license_url),
                {
                    "source_page": f"https://archive.org/details/{identifier}",
                    "title": doc.get("title"),
                    "license_url": license_url,
                    "hint": None,
                },
            )
