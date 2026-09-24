"""Parse KiCad netlists (`kicad-cli sch export netlist --format kicadxml`)."""

import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Component:
    ref: str
    lib: str = ""
    part: str = ""
    value: str = ""
    description: str = ""
    keywords: str = ""


@dataclass
class Node:
    ref: str
    pin: str
    function: str = ""  # pin name, e.g. "B", "+", "VOUT"


@dataclass
class Net:
    name: str
    nodes: list[Node] = field(default_factory=list)


@dataclass
class Circuit:
    name: str
    components: dict[str, Component]
    nets: list[Net]


def parse_kicad_xml(path: Path) -> Circuit:
    root = ET.parse(path).getroot()
    components = {}
    for comp in root.find("components") or []:
        source = comp.find("libsource")
        props = {p.get("name"): p.get("value", "") for p in comp.findall("property")}
        components[comp.get("ref")] = Component(
            ref=comp.get("ref"),
            lib=source.get("lib", "") if source is not None else "",
            part=source.get("part", "") if source is not None else "",
            value=(comp.findtext("value") or "").strip(),
            description=(source.get("description", "") if source is not None else "")
            or props.get("ki_description", ""),
            keywords=props.get("ki_keywords", ""),
        )
    nets = []
    for net in root.find("nets") or []:
        nodes = [
            Node(n.get("ref"), n.get("pin"), n.get("pinfunction", ""))
            for n in net.findall("node")
        ]
        nets.append(Net(net.get("name", ""), nodes))
    return Circuit(Path(path).name.removesuffix(".xml").removesuffix(".netlist"), components, nets)


def load(path: Path, kicad_cli: str | None = None) -> Circuit:
    """A kicadxml netlist, or a .kicad_sch that is exported to one first."""
    path = Path(path)
    if path.suffix == ".kicad_sch":
        if not kicad_cli:
            raise ValueError("kicad-cli path needed to export a .kicad_sch")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "netlist.xml"
            subprocess.run(
                [kicad_cli, "sch", "export", "netlist", "--format", "kicadxml", "-o", str(out), str(path)],
                check=True, capture_output=True, text=True, timeout=300,
            )
            circuit = parse_kicad_xml(out)
            circuit.name = path.stem
            return circuit
    return parse_kicad_xml(path)
