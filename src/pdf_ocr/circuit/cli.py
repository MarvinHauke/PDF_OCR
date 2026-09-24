"""`pdf-ocr circuit`: subcircuits from KiCad netlists, plus evaluation against a gold set.

Gold files (training_data/circuits/gold/<circuit>.json) list the expected
subcircuits by component refs. `--init-gold` writes the current matches as a
starting point; review them by hand (delete wrong entries, add missing ones),
then `--evaluate` reports precision/recall per pattern type.

`--review` draws the entries (gold file if present, else the current matches)
into the rendered schematic sheets: one colored, numbered box per entry plus a
legend, numbered like the gold file. Needs the project's .kicad_sch files next
to the netlist (<name>.kicad/, kept by the crawler).
"""

import json
from collections import Counter
from pathlib import Path

from pdf_ocr.circuit import netlist
from pdf_ocr.circuit.graph import CircuitGraph
from pdf_ocr.circuit.kicad_sch import KICAD_CLI, placed_symbols, render_sheet
from pdf_ocr.circuit.patterns import analyze

REVIEW_DPI = 150
LABEL_ALPHA = 90  # label background opacity (0-255): the circuit stays visible behind it
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46a0a0", "#f032e6",
          "#808000", "#9a6324", "#800000", "#000075", "#008080"]


def collect(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.netlist.xml"))
    return [path]


def run(path: Path, out_dir: Path, gold_dir: Path, evaluate: bool = False, init_gold: bool = False,
        review: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for source in collect(path):
        circuit = netlist.load(source, kicad_cli=KICAD_CLI)
        if not circuit.components:
            print(f"{circuit.name}: empty netlist, skipped")
            continue
        result = analyze(CircuitGraph(circuit))
        results[circuit.name] = result
        (out_dir / f"{circuit.name}.circuit.json").write_text(json.dumps(result, indent=2))
        top = Counter(s["type"] for s in result["subcircuits"] if not s["part_of"])
        ambiguous = sum(1 for s in result["subcircuits"] if s["ambiguous"] and not s["part_of"])
        print(f"{circuit.name}: {len(result['components'])} parts, "
              + ", ".join(f"{t} {n}" for t, n in top.most_common())
              + (f" ({ambiguous} ambiguous)" if ambiguous else ""))

        if init_gold:
            gold_file = gold_dir / f"{circuit.name}.json"
            if gold_file.exists():
                print(f"  gold file exists, not overwritten: {gold_file}")
            else:
                gold_dir.mkdir(parents=True, exist_ok=True)
                gold_file.write_text(json.dumps({"circuit": circuit.name, "reviewed": False, "subcircuits": [
                    {"type": s["type"], "components": s["components"]}
                    for s in result["subcircuits"] if not s["part_of"]
                ]}, indent=2))
                print(f"  gold draft written: {gold_file} (review it, then set reviewed: true)")

        if review:
            gold_file = gold_dir / f"{circuit.name}.json"
            entries = (json.loads(gold_file.read_text())["subcircuits"] if gold_file.exists() else
                       [s for s in result["subcircuits"] if not s["part_of"]])
            source_label = "gold file" if gold_file.exists() else "current matches"
            for image in review_images(source, circuit.name, entries, source_label, out_dir / "review"):
                print(f"  review: {image}")

    print(f"\nResults in {out_dir}")
    if evaluate:
        report(results, gold_dir)


def _unit_ref(component: str) -> tuple[str, int | None]:
    """'U2.A' -> ('U2', 1); 'R3' -> ('R3', None)."""
    ref, _, unit = component.partition(".")
    return ref, (ord(unit.upper()) - 64 if unit else None)


def review_images(netlist_path: Path, name: str, entries: list[dict], source_label: str,
                  out_dir: Path) -> list[Path]:
    """One PNG per sheet: numbered, colored boxes around each entry's parts, plus a legend."""
    from PIL import Image, ImageDraw, ImageFont

    project = netlist_path.with_name(netlist_path.name.removesuffix(".netlist.xml") + ".kicad")
    sheets = sorted(project.rglob("*.kicad_sch")) if project.is_dir() else []
    if not sheets:
        print(f"  no schematic files at {project}, cannot draw a review")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=18)
    px = REVIEW_DPI / 25.4
    drawn_anywhere, written = set(), []
    for sheet in sheets:
        symbols, _, _ = placed_symbols(sheet)
        boxes = {}
        for sym in symbols:
            boxes.setdefault(sym.ref, []).append(sym)
        found = []
        for i, entry in enumerate(entries, start=1):
            parts = []
            for component in entry["components"]:
                ref, unit = _unit_ref(component)
                parts += [s.box for s in boxes.get(ref, []) if unit is None or s.unit == unit]
            if parts:
                found.append((i, entry, parts))
        if not found:
            continue
        image = render_sheet(sheet, REVIEW_DPI)
        if image is None:
            print(f"  render failed: {sheet.name}")
            continue
        legend_w = 560
        canvas = Image.new("RGBA", (image.width + legend_w, image.height), "white")
        canvas.paste(image, (0, 0))
        # boxes and labels go on a transparent layer, so the circuit shows through
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        for i, entry, parts in found:
            drawn_anywhere.add(i)
            color = COLORS[(i - 1) % len(COLORS)]
            rgb = tuple(int(color[k:k + 2], 16) for k in (1, 3, 5))
            x1 = min(b[0] for b in parts) * px - 8
            y1 = min(b[1] for b in parts) * px - 8
            x2 = max(b[2] for b in parts) * px + 8
            y2 = max(b[3] for b in parts) * px + 8
            odraw.rectangle([x1, y1, x2, y2], outline=rgb + (230,), width=3)
            label = f"#{i} {entry['type']}"
            tw = odraw.textlength(label, font=font)
            odraw.rectangle([x1, y1 - 28, x1 + tw + 8, y1], fill=rgb + (LABEL_ALPHA,))
            odraw.text((x1 + 4, y1 - 26), label, fill=(255, 255, 255, 255), font=font,
                       stroke_width=2, stroke_fill=rgb + (255,))
        canvas = Image.alpha_composite(canvas, overlay)
        draw = ImageDraw.Draw(canvas)
        x, y = image.width + 16, 16
        draw.text((x, y), f"{name}", fill="black", font=font)
        draw.text((x, y + 30), f"sheet {sheet.name} · entries from {source_label}", fill="gray", font=small)
        y += 70
        on_sheet = {i for i, _, _ in found}
        for i, entry in enumerate(entries, start=1):
            color = COLORS[(i - 1) % len(COLORS)] if i in on_sheet else "gray"
            text = f"#{i} {entry['type']}: {', '.join(entry['components'])}"
            draw.text((x, y), text[:60], fill=color, font=small)
            y += 24
        target = out_dir / f"{name}__{sheet.stem}.png"
        canvas.convert("RGB").save(target)
        written.append(target)
    missing = [i for i in range(1, len(entries) + 1) if i not in drawn_anywhere]
    if missing:
        print(f"  entries without parts in the schematic files: {missing} (refs differ, e.g. a reused sub-sheet)")
    return written


def _key(s: dict) -> tuple:
    return s["type"], frozenset(s["components"])


def report(results: dict, gold_dir: Path):
    tp, fp, fn = Counter(), Counter(), Counter()
    used = 0
    for name, result in results.items():
        gold_file = gold_dir / f"{name}.json"
        if not gold_file.exists():
            continue
        gold = json.loads(gold_file.read_text())
        if not gold.get("reviewed"):
            print(f"{name}: gold not reviewed yet, skipped")
            continue
        used += 1
        predicted = {_key(s) for s in result["subcircuits"] if not s["part_of"]}
        expected = {_key(s) for s in gold["subcircuits"]}
        for t, _ in predicted & expected:
            tp[t] += 1
        for t, _ in predicted - expected:
            fp[t] += 1
        for t, _ in expected - predicted:
            fn[t] += 1
    if not used:
        print("No reviewed gold files to evaluate against.")
        return
    print(f"\nEvaluation on {used} reviewed circuit(s):")
    print(f"{'type':22} {'TP':>4} {'FP':>4} {'FN':>4} {'precision':>10} {'recall':>7}")
    for t in sorted(set(tp) | set(fp) | set(fn)):
        p = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else 0
        r = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else 0
        print(f"{t:22} {tp[t]:4} {fp[t]:4} {fn[t]:4} {p:10.2f} {r:7.2f}")
