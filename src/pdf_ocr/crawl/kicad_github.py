"""Open-hardware KiCad projects on GitHub, rendered to PDF with kicad-cli.

Repos are found by `<query> license:<key>` (queries in crawl.yaml) for free license keys only; a
repo counts if its git tree contains a .kicad_sch. The repo tarball is
downloaded to a temp folder (size-capped), the root schematic (the one next
to the .kicad_pro, else the first) is exported with `kicad-cli sch export pdf`
and `sch export netlist --format kicadxml`. Kept per project:

    <owner>__<repo>.pdf            rendered sheets (-> pdf-ocr ingest)
    <owner>__<repo>.netlist.xml    true circuit graph (-> pdf-ocr circuit)
    <owner>__<repo>.kicad/         all .kicad_sch files, relative paths kept
                                   (symbol positions -> auto-labeled symbol boxes)

`--refresh-netlists` backfills netlist and schematic files for projects
crawled before they were kept.

Unauthenticated GitHub limits are tight (search 10/min, core 60/h); a valid
GITHUB_TOKEN in the environment raises them. An invalid one is ignored.
"""

import json
import os
import shutil
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


def _root_schematic(blobs: dict[str, int]) -> str | None:
    """The top-level schematic: prefer ones with a .kicad_pro next to them, then the
    largest (repos often hold a main board plus small side projects like a lid)."""
    schematics = [p for p in blobs if p.endswith(".kicad_sch")]
    projects = {p[: -len(".kicad_pro")] for p in blobs if p.endswith(".kicad_pro")}
    with_project = [s for s in schematics if s[: -len(".kicad_sch")] in projects]
    candidates = with_project or schematics
    return max(candidates, key=lambda p: blobs[p]) if candidates else None


def crawl(ctx: CrawlContext):
    settings = ctx.settings["kicad_github"]
    kicad_cli = settings["kicad_cli"]
    if not Path(kicad_cli).exists():
        raise CrawlStop(f"kicad-cli not found at {kicad_cli} (crawl.yaml kicad_github.kicad_cli)")
    max_tar_mb = ctx.settings["limits"]["max_tarball_mb"]
    ctx.fetcher.session.headers["Accept"] = "application/vnd.github+json"
    _use_token_if_valid(ctx)

    searches = [(q, lic) for q in settings.get("queries", ["topic:kicad"]) for lic in settings["licenses"]]
    for query, license_key in searches:
        if ctx.done:
            return
        print(f"- {query} license:{license_key}")
        search = _github(
            ctx,
            f"{API}/search/repositories",
            params={"q": f"{query} license:{license_key}", "sort": "stars", "per_page": 30},
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
            root = _root_schematic(
                {t["path"]: t.get("size", 0) for t in tree.get("tree", []) if t.get("type") == "blob"}
            )
            if root is None:
                ctx.skip("no .kicad_sch", name)
                continue
            if ctx.dry_run:
                print(f"  would render [free] {name}:{root}")
                ctx.summary["would download"] += 1
                ctx.downloaded += 1
                continue
            base = safe_filename(name.replace("/", "__"))
            with tempfile.TemporaryDirectory(prefix="kicad_") as tmp:
                top = _download(ctx, Path(tmp), name, branch, max_tar_mb)
                if top is None:
                    continue
                pdf = _export(ctx, kicad_cli, top / root, Path(tmp) / "sheet.pdf", "pdf", name)
                if pdf is None:
                    continue
                extras = _keep_project_files(ctx, kicad_cli, top, root, base, name)
                store_local(
                    ctx,
                    pdf,
                    f"{base}.pdf",
                    url,
                    license_name,
                    "free",
                    {
                        "source_page": url,
                        "schematic": root,
                        "branch": branch,
                        "author": repo.get("owner", {}).get("login"),
                        "hint": None,
                        **extras,
                    },
                )


def _export(ctx, kicad_cli, schematic: Path | None, out: Path, kind: str, name: str) -> Path | None:
    """kicad-cli sch export pdf|netlist; returns the output path or None."""
    if schematic is None:
        return None
    args = [kicad_cli, "sch", "export", kind, "-o", str(out)]
    if kind == "netlist":
        args += ["--format", "kicadxml"]
    result = subprocess.run(args + [str(schematic)], capture_output=True, text=True, timeout=300)
    if result.returncode != 0 or not out.exists():
        ctx.skip(f"kicad-cli {kind} failed ({result.stderr.strip()[:80]})", name)
        return None
    return out


def _keep_project_files(ctx, kicad_cli, top: Path, root: str, base: str, name: str) -> dict:
    """Store netlist + all .kicad_sch files next to the PDF; returns provenance fields."""
    fields = {}
    netlist = _export(ctx, kicad_cli, top / root, top.parent / "netlist.xml", "netlist", name)
    if netlist is not None:
        dest = ctx.out_dir / f"{base}.netlist.xml"
        ctx.budget.check(netlist.stat().st_size)
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(netlist, dest)
        ctx.budget.add(dest.stat().st_size)
        fields["netlist"] = dest.name
    sheets = sorted(top.rglob("*.kicad_sch"))
    size = sum(f.stat().st_size for f in sheets)
    ctx.budget.check(size)
    kicad_dir = ctx.out_dir / f"{base}.kicad"
    for sheet in sheets:
        dest = kicad_dir / sheet.relative_to(top)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sheet, dest)
    ctx.budget.add(size)
    fields["kicad_dir"] = kicad_dir.name
    fields["root_schematic"] = root
    return fields


def refresh_netlists(ctx: CrawlContext):
    """Backfill netlist + schematic files for projects crawled before they were kept."""
    settings = ctx.settings["kicad_github"]
    ctx.fetcher.session.headers["Accept"] = "application/vnd.github+json"
    _use_token_if_valid(ctx)
    records = ctx.provenance.records
    todo = [r for r in records if "netlist" not in r]
    print(f"- {len(todo)} project(s) without netlist")
    for record in todo:
        name = record["url"].removeprefix("https://github.com/")
        branch = record.get("branch") or _github(ctx, f"{API}/repos/{name}").json()["default_branch"]
        base = record["file"].removesuffix(".pdf")
        with tempfile.TemporaryDirectory(prefix="kicad_") as tmp:
            top = _download(ctx, Path(tmp), name, branch, ctx.settings["limits"]["max_tarball_mb"])
            if top is None:
                continue
            record.update(_keep_project_files(ctx, settings["kicad_cli"], top, record["schematic"], base, name))
            record["branch"] = branch
            print(f"  {name}: {record.get('netlist', 'no netlist')}")
            ctx.summary["netlists refreshed"] += 1
    ctx.provenance.path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def _download(ctx, tmp: Path, name, branch, max_tar_mb) -> Path | None:
    """Download and unpack the repo tarball into tmp; returns the repo folder."""
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
    tarball.unlink()
    return next((tmp / "src").iterdir())  # GitHub wraps the repo in one folder
