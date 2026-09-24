"""KiCanvas review exports -> gold files and the circuit graph dataset.

The review page (review_html.py) shows the analysis matches; in KiCanvas each
group is marked Correct or Wrong and exported as <circuit>.groups.json (the
page's own groups JSON with a `review` field per group). Importing it updates
training_data/circuits/gold/<circuit>.json:

- subcircuits: the confirmed entries -- matches marked correct, plus entries
  added by hand (anything that isn't one of the analysis matches)
  Groups created in KiCanvas during review (new#n) land here too; without a
  kind they get the type "unlabeled".
- rejected: matches marked wrong (kept so they're not asked again, and as
  negatives for the graph dataset)
- reviewed: true once every match that needs a verdict has one. A match needs
  one if it's top-level, or its parent was marked wrong. Children of a correct
  match are covered by it, the same way --evaluate only compares top-level
  matches.

Entries are keyed by (type, components), not by group id, so an export stays
valid when the analysis renumbers its matches. An untouched --init-gold draft
(reviewed false, no rejected list) only holds copies of the matches; those
count as unconfirmed and are replaced by the verdicts.

Each import also writes training_data/circuits/graphs/<circuit>.json: the
netlist graph (units, nets, pin edges) with the gold subcircuits as positive
and the rejected ones as negative labels.
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pdf_ocr.circuit.graph import CircuitGraph

VERDICTS = ("correct", "wrong")
UNLABELED = "unlabeled"  # type of a group created in KiCanvas without a kind


def key(entry: dict) -> tuple:
    """(type, components) of a gold entry or analysis match."""
    return entry["type"], frozenset(entry["components"])


def is_confirmed(gold: dict | None) -> bool:
    """True once the gold file's entries were checked (by hand or by an import)."""
    return gold is not None and (bool(gold.get("reviewed")) or "rejected" in gold)


def group_refs(group: dict) -> list[str]:
    refs = group.get("refs", [])
    return refs.split() if isinstance(refs, str) else list(refs)


def load_exports(path: Path) -> dict[str, tuple[Path, dict]]:
    """circuit name -> (file, groups JSON). A folder is searched for *.json (browsers
    rename repeated downloads to 'x.groups (1).json'); the newest export per circuit wins."""
    files = [path] if path.is_file() else sorted(path.glob("*.json"), key=lambda p: p.stat().st_mtime)
    exports = {}
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict) or data.get("version") != 1 or "groups" not in data:
            continue
        name = data.get("title")
        if not name:
            print(f"{file.name}: groups JSON without a title, skipped")
            continue
        if name in exports:
            print(f"{name}: several exports, using the newest ({file.name})")
        exports[name] = (file, data)
    return exports


def merge_review(export: dict, result: dict, gold: dict | None, source_name: str) -> tuple[dict, Counter]:
    """Apply the verdicts of one export to the circuit's gold file (new dict)."""
    if export.get("source") != "analysis":
        raise ValueError(f"export source is {export.get('source')!r}; review the analysis page "
                         "(pdf-ocr circuit ... --review), gold pages are read-only views")
    matches = result["subcircuits"]
    match_keys = {key(s) for s in matches}
    entries = list(gold["subcircuits"]) if gold else []
    if gold and not is_confirmed(gold):
        entries = [e for e in entries if key(e) not in match_keys]
    kept = {key(e): e for e in entries}
    rejected = {key(e): e for e in (gold or {}).get("rejected", [])}

    counts = Counter()
    for group in export["groups"]:
        verdict = group.get("review")
        if verdict not in VERDICTS or not group_refs(group):
            continue
        if not group.get("kind"):
            print(f"  {group.get('id')}: no kind given, imported as {UNLABELED!r} "
                  "(fix in the gold file, or copy the group in KiCanvas and name it)")
        entry = {"type": group.get("kind") or UNLABELED, "components": sorted(group_refs(group))}
        unknown = [r for r in entry["components"] if r not in result["components"]]
        if unknown:
            print(f"  {group.get('id')}: refs not in the netlist: {', '.join(unknown)}")
        k = key(entry)
        if verdict == "correct":
            kept[k] = entry
            rejected.pop(k, None)
        else:
            rejected[k] = entry
            kept.pop(k, None)
        counts[verdict] += 1

    # A child of a confirmed match is covered by it
    by_id = {s["id"]: s for s in matches}
    for s in matches:
        parent = by_id.get(s["part_of"])
        if parent is not None and key(parent) in kept:
            kept.pop(key(s), None)

    need = [s for s in matches
            if s["part_of"] is None or key(by_id[s["part_of"]]) in rejected]
    open_ = [s for s in need if key(s) not in kept and key(s) not in rejected]
    counts["open"] = len(open_)

    imports = list((gold or {}).get("review_imports", []))
    imports.append({"file": source_name,
                    "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    new_gold = {
        "circuit": result["circuit"],
        "reviewed": not open_,
        "subcircuits": sorted(kept.values(), key=lambda e: (e["type"], e["components"])),
        "rejected": sorted(rejected.values(), key=lambda e: (e["type"], e["components"])),
        "review_imports": imports,
    }
    return new_gold, counts


def graph_record(g: CircuitGraph, gold: dict, netlist: str) -> dict:
    """Graph dataset entry: bipartite unit/net graph with labeled subcircuits."""
    edges = []
    for a, b, data in g.g.edges(data=True):
        unit, net = (a, b) if a[0] == "unit" else (b, a)
        edges.append({"unit": unit[1], "net": net[1], "roles": sorted(data["roles"])})
    nets = sorted({e["net"] for e in edges})
    return {
        "version": 1,
        "circuit": g.name,
        "netlist": netlist,
        "reviewed": gold["reviewed"],
        "units": [{"id": u.id, "ref": u.ref, "type": u.ctype, "value": u.value}
                  for u in sorted(g.units.values(), key=lambda u: u.id)],
        "nets": [{"name": n, "rail": "ground" if n in g.ground else "supply" if n in g.supply else None}
                 for n in nets],
        "edges": sorted(edges, key=lambda e: (e["unit"], e["net"])),
        "subcircuits": [{"type": e["type"], "units": e["components"], "label": "positive"}
                        for e in gold["subcircuits"]]
                       + [{"type": e["type"], "units": e["components"], "label": "negative"}
                          for e in gold.get("rejected", [])],
    }


def import_review(export_file: Path, export: dict, g: CircuitGraph, result: dict, netlist: Path,
                  gold_dir: Path) -> dict:
    """Merge one export into the gold file and write the graph dataset entry."""
    gold_file = gold_dir / f"{result['circuit']}.json"
    gold = json.loads(gold_file.read_text()) if gold_file.exists() else None
    new_gold, counts = merge_review(export, result, gold, export_file.name)
    gold_dir.mkdir(parents=True, exist_ok=True)
    gold_file.write_text(json.dumps(new_gold, indent=2))

    graphs_dir = gold_dir.parent / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    try:
        netlist_name = str(netlist.resolve().relative_to(gold_dir.resolve().parents[2]))
    except ValueError:
        netlist_name = netlist.name
    (graphs_dir / f"{result['circuit']}.json").write_text(
        json.dumps(graph_record(g, new_gold, netlist_name), indent=2))

    print(f"  review imported from {export_file.name}: {counts['correct']} correct, "
          f"{counts['wrong']} wrong -> gold {len(new_gold['subcircuits'])} entries, "
          f"{len(new_gold['rejected'])} rejected"
          + (f", {counts['open']} match(es) still without a verdict" if counts["open"] else ", complete"))
    return new_gold
