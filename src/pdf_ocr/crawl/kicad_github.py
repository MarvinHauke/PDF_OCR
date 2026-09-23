"""Open-hardware KiCad projects on GitHub, rendered to PDF with kicad-cli.

Repos are found by `topic:kicad license:<key>` for free license keys only; a
repo counts if its git tree contains a .kicad_sch. The repo tarball is
downloaded to a temp folder (size-capped), the root schematic (the one next
to the .kicad_pro, else the first) is exported with `kicad-cli sch export pdf`,
and only that PDF is kept.

Unauthenticated GitHub limits are tight (search 10/min, core 60/h); a valid
GITHUB_TOKEN in the environment raises them. An invalid one is ignored.
"""

import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

from pdf_ocr.crawl.base import MB, CrawlContext, CrawlStop, safe_filename, store_local

API = "https://api.github.com"


def _use_token_if_valid(ctx: CrawlContext):
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return
    response = ctx.fetcher.get(f"{API}/rate_limit", api=True, headers={"Authorization": f"Bearer {token}"})
    if response.ok:
        ctx.fetcher.session.headers["Authorization"] = f"Bearer {token}"
        print("  using GITHUB_TOKEN")
    else:
        print("  GITHUB_TOKEN is invalid, continuing without it (lower rate limits)")


def _github(ctx: CrawlContext, url: str, **kwargs):
    response = ctx.fetcher.get(url, api=True, **kwargs)
    if response.status_code in (403, 429) and response.headers.get("X-RateLimit-Remaining") == "0":
        raise CrawlStop(
            "GitHub rate limit reached (resets at unix time "
            f"{response.headers.get('X-RateLimit-Reset')}); set a valid GITHUB_TOKEN or try later"
        )
    return response


def _root_schematic(paths: list[str]) -> str | None:
    schematics = [p for p in paths if p.endswith(".kicad_sch")]
    projects = {p[: -len(".kicad_pro")] for p in paths if p.endswith(".kicad_pro")}
    for sch in schematics:
        if sch[: -len(".kicad_sch")] in projects:
            return sch
    return schematics[0] if schematics else None


def crawl(ctx: CrawlContext):
    settings = ctx.settings["kicad_github"]
    kicad_cli = settings["kicad_cli"]
    if not Path(kicad_cli).exists():
        raise CrawlStop(f"kicad-cli not found at {kicad_cli} (crawl.yaml kicad_github.kicad_cli)")
    max_tar_mb = ctx.settings["limits"]["max_tarball_mb"]
    ctx.fetcher.session.headers["Accept"] = "application/vnd.github+json"
    _use_token_if_valid(ctx)

    for license_key in settings["licenses"]:
        if ctx.done:
            return
        print(f"- topic:kicad license:{license_key}")
        search = _github(
            ctx,
            f"{API}/search/repositories",
            params={"q": f"topic:kicad license:{license_key}", "sort": "stars", "per_page": 30},
        ).json()
        for repo in search.get("items", []):
            if ctx.done:
                return
            name, branch = repo["full_name"], repo["default_branch"]
            url = repo["html_url"]
            if url in ctx.provenance.urls:
                ctx.skip("already downloaded")
                continue
            license_name = (repo.get("license") or {}).get("spdx_id")
            tier, why = ctx.policy.tier(license_name, url)
            if tier != "free":  # GitHub is not a reviewed host: only licensed repos
                ctx.skip(f"license: {why}", name)
                continue
            if repo.get("size", 0) / 1024 > max_tar_mb:  # size is in KB
                ctx.skip(f"size: repo > {max_tar_mb} MB", name)
                continue

            tree = _github(ctx, f"{API}/repos/{name}/git/trees/{branch}", params={"recursive": 1}).json()
            root = _root_schematic([t["path"] for t in tree.get("tree", []) if t.get("type") == "blob"])
            if root is None:
                ctx.skip("no .kicad_sch", name)
                continue
            if ctx.dry_run:
                print(f"  would render [free] {name}:{root}")
                ctx.summary["would download"] += 1
                ctx.downloaded += 1
                continue
            with tempfile.TemporaryDirectory(prefix="kicad_") as tmp:
                pdf = _render(ctx, Path(tmp), name, branch, root, kicad_cli, max_tar_mb)
                if pdf is None:
                    continue
                store_local(
                    ctx,
                    pdf,
                    safe_filename(f"{name.replace('/', '__')}.pdf"),
                    url,
                    license_name,
                    "free",
                    {
                        "source_page": url,
                        "schematic": root,
                        "author": repo.get("owner", {}).get("login"),
                        "hint": None,
                    },
                )


def _render(ctx, tmp: Path, name, branch, root, kicad_cli, max_tar_mb) -> Path | None:
    """Download the repo tarball into tmp and export the root schematic to PDF."""
    response = _github(ctx, f"{API}/repos/{name}/tarball/{branch}", stream=True)
    if not response.ok:
        ctx.skip(f"http {response.status_code}", name)
        return None
    tarball, size = tmp / "repo.tar.gz", 0
    with tarball.open("wb") as f:
        for chunk in response.iter_content(1 << 16):
            size += len(chunk)
            if size > max_tar_mb * MB:
                ctx.skip(f"size: tarball > {max_tar_mb} MB", name)
                return None
            f.write(chunk)
    with tarfile.open(tarball) as tar:
        tar.extractall(tmp / "src", filter="data")
    top = next((tmp / "src").iterdir())  # GitHub wraps the repo in one folder
    out = tmp / "sheet.pdf"
    result = subprocess.run(
        [kicad_cli, "sch", "export", "pdf", "-o", str(out), str(top / root)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0 or not out.exists():
        ctx.skip(f"kicad-cli failed ({result.stderr.strip()[:80]})", name)
        return None
    return out
