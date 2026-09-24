"""Read symbol placements from a KiCad 6+ schematic (.kicad_sch) and compute each
symbol's bounding box on the sheet, in mm (origin top-left, y down).

Used to generate symbol-detection labels from rendered KiCad sheets: the
schematic says exactly which symbol sits where, so no manual labeling is needed.

Library symbol graphics are stored with y pointing up; placed symbols carry
(at x y angle), optional (mirror x|y) and a unit number.
"""

import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

KICAD_CLI = "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"


# -- minimal s-expression parser -------------------------------------------------

TOKEN = re.compile(r'\s*(?:(\()|(\))|"((?:[^"\\]|\\.)*)"|([^\s()"]+))')


def parse_sexpr(text: str):
    stack, cur = [], []
    for m in TOKEN.finditer(text):
        open_, close, string, atom = m.groups()
        if open_:
            stack.append(cur)
            cur = []
        elif close:
            done, cur = cur, stack.pop()
            cur.append(done)
        elif string is not None:
            cur.append(string.replace('\\"', '"'))
        elif atom is not None:
            cur.append(atom)
    return cur[0] if cur else None


def children(node, tag):
    return [c for c in node if isinstance(c, list) and c and c[0] == tag]


def child(node, tag):
    found = children(node, tag)
    return found[0] if found else None


def num(x) -> float:
    return float(x)


# -- symbol geometry ---------------------------------------------------------------

def _points(item) -> list[tuple[float, float]]:
    """Points of one graphic item in library coordinates (y up)."""
    tag = item[0]
    if tag == "rectangle":
        s, e = child(item, "start"), child(item, "end")
        return [(num(s[1]), num(s[2])), (num(e[1]), num(e[2]))]
    if tag in ("polyline", "bezier"):
        pts = child(item, "pts") or []
        return [(num(p[1]), num(p[2])) for p in children(pts, "xy")]
    if tag == "circle":
        c, r = child(item, "center"), num(child(item, "radius")[1])
        cx, cy = num(c[1]), num(c[2])
        return [(cx - r, cy - r), (cx + r, cy + r)]
    if tag == "arc":
        return [(num(child(item, k)[1]), num(child(item, k)[2])) for k in ("start", "mid", "end")
                if child(item, k)]
    if tag == "pin":
        at, length = child(item, "at"), child(item, "length")
        x, y, ang = num(at[1]), num(at[2]), num(at[3]) if len(at) > 3 else 0.0
        ln = num(length[1]) if length else 0.0
        return [(x, y), (x + ln * math.cos(math.radians(ang)), y + ln * math.sin(math.radians(ang)))]
    return []


GRAPHICS = ("rectangle", "polyline", "bezier", "circle", "arc", "pin")


def library_extents(lib_symbols):
    """lib name -> ({unit: points}, {unit: has body graphics}, {property: value}).
    Unit 0 holds graphics shared by all units."""
    out = {}
    for sym in children(lib_symbols, "symbol"):
        name = sym[1]
        per_unit, body = {}, {}
        for sub in children(sym, "symbol"):
            m = re.search(r"_(\d+)_(\d+)$", sub[1])
            unit, style = (int(m.group(1)), int(m.group(2))) if m else (0, 1)
            if style > 1:  # De Morgan alternative body
                continue
            items = [i for i in sub if isinstance(i, list) and i and i[0] in GRAPHICS]
            per_unit.setdefault(unit, []).extend(p for i in items for p in _points(i))
            body[unit] = body.get(unit, False) or any(i[0] != "pin" for i in items)
        props = {p[1]: p[2] for p in children(sym, "property") if len(p) > 2}
        out[name] = (per_unit, body, props)
    return out


@dataclass
class PlacedSymbol:
    ref: str
    lib_id: str
    value: str
    unit: int
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in sheet mm
    has_body: bool = True  # False for pin-only units (e.g. an op-amp's power unit)
    description: str = ""
    keywords: str = ""


def _transform(pts, x0, y0, angle, mirror):
    """Library points (y up) -> sheet coordinates (mm, y down)."""
    out = []
    a = math.radians(angle)
    ca, sa = round(math.cos(a)), round(math.sin(a))  # KiCad rotates in 90° steps
    for x, y in pts:
        y = -y  # library y-up -> sheet y-down
        if mirror == "x":
            y = -y
        elif mirror == "y":
            x = -x
        xr, yr = x * ca + y * sa, -x * sa + y * ca
        out.append((x0 + xr, y0 + yr))
    return out


def placed_symbols(path: Path) -> tuple[list[PlacedSymbol], list[tuple[float, float]], tuple[float, float]]:
    """(symbols, junction points, paper size in mm) of one sheet file."""
    root = parse_sexpr(Path(path).read_text(encoding="utf-8"))
    extents = library_extents(child(root, "lib_symbols") or [])
    symbols = []
    for sym in children(root, "symbol"):
        lib_id = child(sym, "lib_id")[1]
        lib_name = child(sym, "lib_name")
        key = lib_name[1] if lib_name else lib_id
        at = child(sym, "at")
        x0, y0, angle = num(at[1]), num(at[2]), num(at[3]) if len(at) > 3 else 0.0
        mirror = child(sym, "mirror")[1] if child(sym, "mirror") else None
        unit = int(child(sym, "unit")[1]) if child(sym, "unit") else 1
        props = {p[1]: p[2] for p in children(sym, "property") if len(p) > 2}
        units, body, lib_props = extents.get(key, ({}, {}, {}))
        pts = units.get(0, []) + units.get(unit, [])
        if not pts:
            continue
        xs, ys = zip(*_transform(pts, x0, y0, angle, mirror))
        symbols.append(PlacedSymbol(
            props.get("Reference", "?"), lib_id, props.get("Value", ""), unit,
            (min(xs), min(ys), max(xs), max(ys)),
            has_body=body.get(0, False) or body.get(unit, False),
            description=lib_props.get("Description", lib_props.get("ki_description", "")),
            keywords=lib_props.get("ki_keywords", ""),
        ))
    junctions = [(num(j[1][1]), num(j[1][2])) for j in children(root, "junction")]
    paper = PAPER_MM.get((child(root, "paper") or ["", "A4"])[1], PAPER_MM["A4"])
    return symbols, junctions, paper


PAPER_MM = {"A4": (297, 210), "A3": (420, 297), "A2": (594, 420), "A1": (841, 594), "A0": (1189, 841),
            "A5": (210, 148), "A": (279.4, 215.9), "B": (431.8, 279.4), "C": (558.8, 431.8),
            "USLetter": (279.4, 215.9), "USLegal": (355.6, 215.9), "USLedger": (431.8, 279.4)}


def render_sheet(sheet: Path, dpi: int = 200, kicad_cli: str = KICAD_CLI):
    """Render one sheet (page 1 of its black-and-white PDF export) to a PIL image, or None.
    Sheet coordinates in mm map to pixels as mm * dpi / 25.4 (origin top-left)."""
    import pypdfium2 as pdfium

    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "sheet.pdf"
        result = subprocess.run([kicad_cli, "sch", "export", "pdf", "-b", "-o", str(pdf), str(sheet)],
                                capture_output=True, text=True, timeout=300)
        if result.returncode != 0 or not pdf.exists():
            return None
        document = pdfium.PdfDocument(str(pdf))
        try:
            return document[0].render(scale=dpi / 72).to_pil().convert("RGB")
        finally:
            document.close()
