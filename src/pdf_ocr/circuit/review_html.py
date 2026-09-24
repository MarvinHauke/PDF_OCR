"""Self-contained KiCanvas review pages for `pdf-ocr circuit --review`.

Converts analysis results / gold files into KiCanvas's generic "symbol groups"
format (v1) and writes one HTML page per project that inlines the schematic
files and the groups, so it opens straight from file:// with no server.

The page only references the locally built KiCanvas bundle (KICANVAS_JS or
--kicanvas-js); no KiCanvas code lives in this repo. Rules for the page, agreed
with the KiCanvas side:
- classic <script src=...> for the bundle (module scripts are blocked on file://),
- one <kicanvas-source name=...> per sheet file; name = the Sheetfile path as the
  parent sheet writes it, the root gets its plain file name,
- exactly one <kicanvas-groups> inside <kicanvas-embed>,
- all inlined text HTML-escaped.
"""

import html
import json
import re
from pathlib import Path

from pdf_ocr.circuit.kicad_sch import children, parse_sexpr
from pdf_ocr.circuit.review_import import is_confirmed, key


# -- symbol groups ------------------------------------------------------------------

def to_symbol_groups(result: dict, gold: dict | None = None) -> dict:
    """Analysis result -> symbol-groups v1 dict for reviewing the matches in KiCanvas.

    With a gold file, verdicts from earlier reviews are pre-filled (`review`), and
    entries that exist only in the gold file (added by hand) are appended as gold#n,
    so the export (review_import.py) round-trips them."""
    parts = {ref: {"kind": c["type"], "value": c["value"]} for ref, c in result["components"].items()}
    confirmed = {key(e) for e in gold["subcircuits"]} if is_confirmed(gold) else set()
    rejected = {key(e) for e in gold.get("rejected", [])} if gold else set()
    groups = []
    for s in result["subcircuits"]:
        group = {"id": s["id"], "label": s["id"], "kind": s["type"], "refs": list(s["components"])}
        if s.get("ambiguous"):
            group["status"] = "ambiguous"
        if s.get("part_of"):
            group["parent"] = s["part_of"]
        if s.get("notes"):
            group["description"] = s["notes"]
        if key(s) in confirmed:
            group["review"] = "correct"
        elif key(s) in rejected:
            group["review"] = "wrong"
        groups.append(group)
    match_keys = {key(s) for s in result["subcircuits"]}
    extra = [e for e in (gold or {}).get("subcircuits", []) if key(e) not in match_keys]
    for i, entry in enumerate(extra, start=1):
        group = {"id": f"gold#{i}", "label": f"gold#{i} {entry['type']}", "kind": entry["type"],
                 "refs": list(entry["components"]), "description": ["added by hand in the gold file"]}
        if confirmed:
            group["review"] = "correct"
        groups.append(group)
    return {"version": 1, "title": result["circuit"], "source": "analysis", "reviewed": False,
            "groups": groups, "parts": parts}


# -- schematic sources --------------------------------------------------------------

def _sheetfiles(sheet_path: Path) -> list[str]:
    """Sheetfile paths referenced by one schematic, as written."""
    root = parse_sexpr(sheet_path.read_text(encoding="utf-8"))
    out = []
    for sheet in children(root, "sheet"):
        for prop in children(sheet, "property"):
            if len(prop) > 2 and prop[1] in ("Sheetfile", "Sheet file"):
                out.append(prop[2])
    return out


def find_root(project_dir: Path, netlist_path: Path | None = None) -> Path | None:
    """Root sheet: named in the netlist's <design><sheet><source>, else the file no sheet refers to."""
    sheets = sorted(project_dir.rglob("*.kicad_sch"))
    if netlist_path is not None and netlist_path.exists():
        m = re.search(r"<sheet[^>]*name=\"/\"[\s\S]*?<source>([^<]+)</source>",
                      netlist_path.read_text(encoding="utf-8", errors="ignore"))
        if m:
            name = Path(m.group(1)).name
            named = [s for s in sheets if s.name == name]
            if named:
                return named[0]
    referenced = set()
    for s in sheets:
        referenced |= {(s.parent / ref).resolve() for ref in _sheetfiles(s)}
    candidates = [s for s in sheets if s.resolve() not in referenced]
    return max(candidates, key=lambda s: s.stat().st_size) if candidates else None


def collect_sources(root: Path) -> list[tuple[str, str]]:
    """(name, text) for the root and every sheet reachable from it, each file once.
    Names are Sheetfile paths as written, relative to the root's directory."""
    base = root.parent
    sources = [(root.name, root.read_text(encoding="utf-8"))]
    seen = {root.resolve()}
    queue = [root]
    while queue:
        current = queue.pop(0)
        for ref in _sheetfiles(current):
            path = (base / ref).resolve()
            if path in seen or not path.exists():
                continue
            seen.add(path)
            sources.append((ref, path.read_text(encoding="utf-8")))
            queue.append(path)
    return sources


# -- page ---------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title} – subcircuit review</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #1e1e1e; }}
  kicanvas-embed {{ display: block; width: 100vw; height: 100vh; }}
</style>
</head>
<body>
<kicanvas-embed controls="full">
{sources}
<kicanvas-groups>{groups}</kicanvas-groups>
</kicanvas-embed>
<script src="{bundle}"></script>
</body>
</html>
"""


def build_page(sources: list[tuple[str, str]], groups: dict, bundle: Path) -> str:
    source_tags = "\n".join(
        f'<kicanvas-source name="{html.escape(name, quote=True)}">{html.escape(text)}</kicanvas-source>'
        for name, text in sources
    )
    return PAGE.format(
        title=html.escape(groups.get("title", "")),
        sources=source_tags,
        groups=html.escape(json.dumps(groups)),
        bundle=html.escape(Path(bundle).resolve().as_uri(), quote=True),
    )


def write_review_page(netlist_path: Path, result: dict, gold: dict | None, bundle: Path,
                      out_dir: Path) -> Path | None:
    project = netlist_path.with_name(netlist_path.name.removesuffix(".netlist.xml") + ".kicad")
    if not project.is_dir():
        print(f"  no schematic files at {project}, cannot write a review page")
        return None
    root = find_root(project, netlist_path)
    if root is None:
        print(f"  no root schematic found in {project}")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{result['circuit']}.html"
    target.write_text(build_page(collect_sources(root), to_symbol_groups(result, gold), bundle),
                      encoding="utf-8")
    return target
