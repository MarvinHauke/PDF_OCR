"""Shared crawler infrastructure: settings, license policy, polite HTTP, budget, provenance.

Rules every source goes through:
- blocked_hosts (crawl.yaml) are never fetched.
- File downloads respect robots.txt. Documented APIs (Wikimedia, archive.org,
  GitHub) are called with api=True: they're governed by their own usage
  policies instead (e.g. Wikimedia's API etiquette: informative User-Agent,
  serial requests, maxlag), and Wikimedia's robots.txt disallows /w/ for
  crawlers, which includes api.php.
- Each file gets a tier: `free` if its license is on the allowlist,
  `restricted` if it has no usable license but its host was reviewed by hand
  (reviewed_hosts) and doesn't opt out of text and data mining in a
  machine-readable way (tdm-reservation header, /.well-known/tdmrep.json),
  otherwise it's skipped. NC/ND licenses are never free.
- Every download is checked against the per-file size limit and the total
  disk cap before and while streaming.
"""

import hashlib
import json
import re
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlparse

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_PROJECT = REPO_ROOT / "training_project"
CRAWL_CONFIG = TRAINING_PROJECT / "config" / "crawl.yaml"
USER_AGENT = "PDF_OCR-crawler/0.1 (training data for a schematic detector{contact})"
MB = 1024 * 1024


class CrawlStop(Exception):
    """Stop the whole run cleanly (disk cap reached, rate limit exhausted)."""


def load_settings(path: Path = CRAWL_CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def host_matches(host: str, domains) -> str | None:
    """The configured domain that host equals or is a subdomain of."""
    for domain in domains or {}:
        if host == domain or host.endswith("." + domain):
            return domain
    return None


class LicensePolicy:
    def __init__(self, settings: dict):
        self.free = [self.normalize(x) for x in settings["free_licenses"]]
        self.reviewed_hosts = settings.get("reviewed_hosts") or {}

    @staticmethod
    def normalize(license_name: str | None) -> str:
        text = (license_name or "").strip().lower()
        text = re.sub(r"[\s_]+", "-", text)
        return text.replace("public-domain-mark", "pdm")

    def is_free(self, license_name: str | None) -> bool:
        norm = self.normalize(license_name)
        if not norm or re.search(r"(^|-)(nc|nd)(-|$)", norm):
            return False
        return any(norm == f or norm.startswith(f + "-") for f in self.free)

    def tier(self, license_name: str | None, url: str) -> tuple[str | None, str]:
        """(tier, reason). tier None means skip."""
        if self.is_free(license_name):
            return "free", license_name
        if host_matches(host_of(url), self.reviewed_hosts):
            return "restricted", f"no free license ({license_name or 'none'}), reviewed host"
        return None, f"license {license_name or 'none'} not free and host not reviewed"


class Fetcher:
    """Serial, rate-limited HTTP with robots.txt, TDM opt-out and blocked-host checks."""

    def __init__(self, settings: dict):
        contact = settings.get("contact") or ""
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT.format(
            contact=f"; {contact}" if contact else ""
        )
        self.has_contact = bool(contact)
        self.delay = 1 / float(settings["limits"].get("requests_per_second", 1))
        self.blocked_hosts = settings.get("blocked_hosts") or {}
        self._last_request = {}
        self._robots = {}
        self._tdm_reserved = {}

    # -- checks -------------------------------------------------------------

    def blocked_reason(self, url: str) -> str | None:
        domain = host_matches(host_of(url), self.blocked_hosts)
        return f"blocked host {domain}" if domain else None

    def robots_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots:
            parser = robotparser.RobotFileParser()
            try:
                response = self._raw_get(f"{base}/robots.txt", timeout=15)
                parser.parse(response.text.splitlines() if response.ok else [])
            except requests.RequestException:
                parser.parse([])
            self._robots[base] = parser
        return self._robots[base].can_fetch(self.session.headers["User-Agent"], url)

    def tdm_reserved(self, url: str) -> bool:
        """Machine-readable TDM opt-out for the host (TDMRep /.well-known/tdmrep.json)."""
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._tdm_reserved:
            reserved = False
            try:
                response = self._raw_get(f"{base}/.well-known/tdmrep.json", timeout=15)
                if response.ok and "json" in response.headers.get("content-type", ""):
                    rules = response.json()
                    rules = rules if isinstance(rules, list) else [rules]
                    reserved = any(str(r.get("tdm-reservation")) == "1" for r in rules)
            except (requests.RequestException, ValueError):
                pass
            self._tdm_reserved[base] = reserved
        return self._tdm_reserved[base]

    # -- requests -----------------------------------------------------------

    def _wait(self, host: str):
        elapsed = time.monotonic() - self._last_request.get(host, 0)
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request[host] = time.monotonic()

    def _raw_get(self, url: str, **kwargs):
        self._wait(host_of(url))
        return self.session.get(url, **kwargs)

    def get(self, url: str, api: bool = False, **kwargs) -> requests.Response:
        """GET with blocked-host check, robots.txt (unless api), retries on 429/5xx."""
        reason = self.blocked_reason(url)
        if reason:
            raise PermissionError(reason)
        if not api and not self.robots_allowed(url):
            raise PermissionError("robots.txt disallows")
        kwargs.setdefault("timeout", 60)
        for attempt in range(4):
            response = self._raw_get(url, **kwargs)
            if response.status_code not in (429, 500, 502, 503, 504):
                return response
            wait = int(response.headers.get("Retry-After", 0) or 0) or 2 ** (attempt + 2)
            time.sleep(min(wait, 120))
        return response


class DiskBudget:
    """Total size cap across all sources/crawled/ folders of all datasets."""

    def __init__(self, max_bytes: int, roots: list[Path]):
        self.max_bytes = max_bytes
        self.used = sum(
            f.stat().st_size for root in roots if root.exists() for f in root.rglob("*") if f.is_file()
        )

    def check(self, size: int):
        if self.used + size > self.max_bytes:
            raise CrawlStop(
                f"disk cap reached: {self.used / MB:.0f} MB used + {size / MB:.1f} MB "
                f"> {self.max_bytes / MB:.0f} MB (limits.max_total_gb)"
            )

    def add(self, size: int):
        self.used += size


class ProvenanceLog:
    """sources.jsonl in a site folder: one record per downloaded file."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.path = folder / "sources.jsonl"
        self.records = []
        if self.path.exists():
            self.records = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
        self.urls = {r["url"] for r in self.records}
        self.hashes = {r["sha256"] for r in self.records}

    def add(self, record: dict):
        self.folder.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records.append(record)
        self.urls.add(record["url"])
        self.hashes.add(record["sha256"])


@dataclass
class CrawlContext:
    name: str
    settings: dict
    fetcher: Fetcher
    policy: LicensePolicy
    budget: DiskBudget
    out_dir: Path
    limit: int
    dry_run: bool = False
    provenance: ProvenanceLog = None
    summary: Counter = field(default_factory=Counter)
    downloaded: int = 0

    def __post_init__(self):
        self.provenance = ProvenanceLog(self.out_dir)

    @property
    def done(self) -> bool:
        return self.downloaded >= self.limit

    def skip(self, reason: str, what: str = ""):
        key = reason.split(":")[0].split(" (")[0]
        self.summary[f"skipped: {key}"] += 1
        if what:
            print(f"  skip {what}: {reason}")

    def fetch_file(
        self,
        url: str,
        filename: str,
        max_mb: float,
        license_name: str | None,
        meta: dict,
        tier: str | None = None,
    ) -> Path | None:
        """Download one file with all policy checks. Returns its path, or None if skipped."""
        if url in self.provenance.urls:
            self.skip("already downloaded")
            return None
        blocked = self.fetcher.blocked_reason(url)
        if blocked:
            self.skip(blocked, filename)
            return None
        if tier is None:
            tier, why = self.policy.tier(license_name, url)
            if tier is None:
                self.skip(f"license: {why}", filename)
                return None
        if tier == "restricted" and self.fetcher.tdm_reserved(url):
            self.skip("tdm reservation", filename)
            return None
        if self.dry_run:
            print(f"  would download [{tier}] {filename} <- {url}")
            self.summary["would download"] += 1
            self.downloaded += 1
            return None

        try:
            response = self.fetcher.get(url, stream=True)
        except PermissionError as e:
            self.skip(str(e), filename)
            return None
        if not response.ok:
            self.skip(f"http {response.status_code}", filename)
            return None
        if tier == "restricted" and response.headers.get("tdm-reservation", "").strip() == "1":
            self.skip("tdm reservation", filename)
            return None

        max_bytes = int(max_mb * MB)
        declared = int(response.headers.get("Content-Length") or 0)
        if declared > max_bytes:
            self.skip(f"size: {declared / MB:.1f} MB > {max_mb} MB", filename)
            return None
        self.budget.check(declared)

        digest = hashlib.sha256()
        size = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=self.out_dir, suffix=".part") as tmp:
            tmp_path = Path(tmp.name)
            try:
                for chunk in response.iter_content(1 << 16):
                    size += len(chunk)
                    if size > max_bytes:
                        break
                    self.budget.check(size)
                    digest.update(chunk)
                    tmp.write(chunk)
            except BaseException:
                tmp.close()
                tmp_path.unlink(missing_ok=True)
                raise
        if size > max_bytes:
            tmp_path.unlink()
            self.skip(f"size: > {max_mb} MB", filename)
            return None
        sha = digest.hexdigest()
        if sha in self.provenance.hashes:
            tmp_path.unlink()
            self.skip("duplicate content", filename)
            return None

        dest = self.out_dir / filename
        shutil.move(tmp_path, dest)
        self.budget.add(size)
        self.provenance.add(
            {
                "file": filename,
                "url": url,
                "license": license_name,
                "tier": tier,
                "sha256": sha,
                "bytes": size,
                "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **meta,
            }
        )
        self.downloaded += 1
        self.summary[f"downloaded ({tier})"] += 1
        print(f"  [{tier}] {filename} ({size / MB:.1f} MB)")
        return dest


def store_local(
    ctx: "CrawlContext", path: Path, filename: str, url: str, license_name: str, tier: str, meta: dict
) -> Path | None:
    """Add a file produced locally (e.g. a rendered KiCad sheet) with the same budget,
    dedup and provenance rules as a download."""
    size = path.stat().st_size
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha in ctx.provenance.hashes:
        ctx.skip("duplicate content", filename)
        return None
    ctx.budget.check(size)
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    dest = ctx.out_dir / filename
    shutil.move(path, dest)
    ctx.budget.add(size)
    ctx.provenance.add(
        {
            "file": filename,
            "url": url,
            "license": license_name,
            "tier": tier,
            "sha256": sha,
            "bytes": size,
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **meta,
        }
    )
    ctx.downloaded += 1
    ctx.summary[f"downloaded ({tier})"] += 1
    print(f"  [{tier}] {filename} ({size / MB:.1f} MB)")
    return dest


def safe_filename(text: str, max_len: int = 120) -> str:
    stem, dot, ext = text.rpartition(".")
    if not dot:
        stem, ext = text, ""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")[:max_len] or "file"
    ext = re.sub(r"[^A-Za-z0-9]+", "", ext)[:8]
    return f"{stem}.{ext}" if ext else stem
