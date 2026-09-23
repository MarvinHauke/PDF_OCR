"""Wikimedia Commons: circuit diagrams from configured categories, license per file.

Uses the MediaWiki API (serial requests, maxlag, informative User-Agent, per
Wikimedia's API etiquette). SVGs are fetched as PNG renders.
"""

import re
import time
from urllib.parse import urlsplit, urlunsplit

from pdf_ocr.crawl.base import CrawlContext, safe_filename

API = "https://commons.wikimedia.org/w/api.php"
RASTER = {"image/png": "png", "image/jpeg": "jpg"}


def _api(ctx: CrawlContext, params: dict) -> dict:
    params = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
    for _ in range(5):
        data = ctx.fetcher.get(API, api=True, params=params).json()
        if data.get("error", {}).get("code") == "maxlag":
            time.sleep(10)  # servers are busy, back off as the etiquette asks
            continue
        return data
    return data


def _members(ctx: CrawlContext, category: str, member_type: str, extra: dict):
    """All members of a category, following API continuation."""
    params = {
        "action": "query",
        "generator" if member_type == "file" else "list": "categorymembers",
        **extra,
    }
    prefix = "gcm" if member_type == "file" else "cm"
    params.update({f"{prefix}title": f"Category:{category}", f"{prefix}type": member_type, f"{prefix}limit": 50})
    while True:
        data = _api(ctx, params)
        query = data.get("query", {})
        yield from query.get("pages", []) if member_type == "file" else query.get("categorymembers", [])
        if "continue" not in data:
            return
        params.update(data["continue"])


def _categories(ctx: CrawlContext, settings: dict):
    """(category, hint) for configured categories plus subcategories up to max_depth."""
    seen = set()
    queue = [(name, hint, 0) for name, hint in settings["categories"].items()]
    while queue:
        name, hint, depth = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        yield name, hint
        if depth < settings.get("max_depth", 0):
            for sub in _members(ctx, name, "subcat", {}):
                queue.append((sub["title"].split(":", 1)[1], hint, depth + 1))


def _clean(url: str | None) -> str:
    """Drop the tracking query (?utm_source=...) Commons appends to file URLs."""
    if not url:
        return ""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _plain(html: str | None) -> str | None:
    return re.sub(r"<[^>]+>", "", html).strip() if html else None


def crawl(ctx: CrawlContext):
    settings = ctx.settings[ctx.name]  # "wikimedia" or "wikimedia_blocks"
    limits = ctx.settings["limits"]
    if not ctx.fetcher.has_contact:
        print("  note: crawl.yaml `contact` is empty; Wikimedia asks for contact info in the User-Agent")

    image_info = {
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist",
        "iiurlwidth": settings.get("render_width", 2000),
    }
    for category, hint in _categories(ctx, settings):
        if ctx.done:
            return
        print(f"- Category:{category} (hint: {hint})")
        for page in _members(ctx, category, "file", image_info):
            if ctx.done:
                return
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata", {})
            title = page["title"].split(":", 1)[1]
            original, thumb = _clean(info.get("url")), _clean(info.get("thumburl"))
            if info.get("mime") in RASTER and info.get("size", 0) <= limits["max_image_mb"] * 1024 * 1024:
                url, ext = original, RASTER[info["mime"]]
            elif thumb.lower().endswith((".png", ".jpg", ".jpeg")):
                url, ext = thumb, thumb.rsplit(".", 1)[1].lower()
            else:
                ctx.skip(f"format {info.get('mime')}", title)
                continue
            ctx.fetch_file(
                url,
                safe_filename(f"{title.rsplit('.', 1)[0]}.{ext}"),
                limits["max_image_mb"],
                meta.get("LicenseShortName", {}).get("value"),
                {
                    "source_page": info.get("descriptionurl"),
                    "license_url": meta.get("LicenseUrl", {}).get("value"),
                    "author": _plain(meta.get("Artist", {}).get("value")),
                    "category": category,
                    "hint": hint,
                },
            )
