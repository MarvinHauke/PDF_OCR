"""Circuit -> graph of functional units and nets.

A unit is one functional part: a resistor, a transistor, or one amplifier of
a dual/quad op-amp (U1.A, U1.B). The networkx graph is bipartite: unit nodes
and net nodes, edges carry the pin roles (e.g. {"base"}, {"in_n"}).
"""

import re
from dataclasses import dataclass, field

import networkx as nx

from pdf_ocr.circuit.components import IGNORED, TWO_TERMINAL, classify, pin_role, refine_transistor
from pdf_ocr.circuit.netlist import Circuit

GROUND = re.compile(r"^(a|d|p|s)?gnd[a-z0-9_]*$|^v?ss[a-z]?$|^0v$|^com$|^earth$|^agnd|^dgnd", re.I)
SUPPLY = re.compile(
    r"^[+-]?\d+(\.\d+)?v\d*$|^\d+v\d+$|^v(cc|dd|ee|bat|bus|pp|nn|in|\+|-)[a-z0-9_]*$|^[+-]v[a-z0-9]*$",
    re.I,
)

# Standard pinouts of dual/quad op-amps: unit -> {pin: role}
OPAMP_UNITS = {
    8: {"A": {"1": "out", "2": "in_n", "3": "in_p"}, "B": {"7": "out", "6": "in_n", "5": "in_p"}},
    14: {
        "A": {"1": "out", "2": "in_n", "3": "in_p"},
        "B": {"7": "out", "6": "in_n", "5": "in_p"},
        "C": {"8": "out", "9": "in_n", "10": "in_p"},
        "D": {"14": "out", "13": "in_n", "12": "in_p"},
    },
}


def base_name(net: str) -> str:
    """'/BM1366/0V8' -> '0V8', 'Net-(R1-Pad2)' stays."""
    return net.rstrip("/").split("/")[-1]


@dataclass
class Unit:
    id: str
    ref: str
    ctype: str
    value: str
    pins: dict[str, list[str]] = field(default_factory=dict)  # role -> nets

    def nets(self) -> list[str]:
        return sorted({n for nets in self.pins.values() for n in nets})


class CircuitGraph:
    def __init__(self, circuit: Circuit):
        self.name = circuit.name
        self.units: dict[str, Unit] = {}
        self.ground: set[str] = set()
        self.supply: set[str] = set()
        self.g = nx.Graph()

        ctypes = {ref: classify(c) for ref, c in circuit.components.items()}
        pins_by_ref: dict[str, list[tuple[str, str, str]]] = {}
        for net in circuit.nets:
            if net.name.startswith("unconnected-"):  # KiCad pseudo-net for a no-connect pin
                continue
            for node in net.nodes:
                pins_by_ref.setdefault(node.ref, []).append((node.pin, node.function, net.name))
            name = base_name(net.name)
            if GROUND.match(name):
                self.ground.add(net.name)
            elif SUPPLY.match(name):
                self.supply.add(net.name)

        for ref, pins in pins_by_ref.items():
            ctype = ctypes.get(ref, "other")
            if ctype == "transistor":
                ctype = refine_transistor([fn for _, fn, _ in pins])
            if ctype in IGNORED:
                continue
            value = circuit.components[ref].value if ref in circuit.components else ""
            for unit_id, unit_pins in self._split_units(ref, ctype, pins).items():
                unit = Unit(unit_id, ref, ctype, value)
                for role, net in unit_pins:
                    unit.pins.setdefault(role, []).append(net)
                self.units[unit_id] = unit

        for net in {n for u in self.units.values() for n in u.nets()}:
            self.g.add_node(("net", net), kind="net", ground=net in self.ground,
                            supply=net in self.supply)
        for unit in self.units.values():
            self.g.add_node(("unit", unit.id), kind="unit", ctype=unit.ctype)
            for role, nets in unit.pins.items():
                for net in nets:
                    edge = self.g.get_edge_data(("unit", unit.id), ("net", net))
                    if edge:
                        edge["roles"].add(role)
                    else:
                        self.g.add_edge(("unit", unit.id), ("net", net), roles={role})

    @staticmethod
    def _split_units(ref, ctype, pins):
        """Dual/quad op-amps -> one unit per amplifier by standard pinout."""
        roles = [(pin_role(ctype, pin, fn), pin, net) for pin, fn, net in pins]
        if ctype in ("opamp", "comparator_ic"):
            max_pin = max((int(p) for _, p, _ in roles if p.isdigit()), default=0)
            layout = OPAMP_UNITS.get(max_pin)
            n_inputs = sum(1 for r, _, _ in roles if r == "in_n")
            if layout and (n_inputs > 1 or n_inputs == 0):
                units = {}
                for name, mapping in layout.items():
                    units[f"{ref}.{name}"] = [(mapping[p], net) for _, p, net in roles if p in mapping]
                return units
        return {ref: [(role, net) for role, _, net in roles]}

    # -- queries used by the patterns ------------------------------------------

    def members(self, net: str) -> list[tuple[str, set[str]]]:
        """(unit id, roles) of everything on a net."""
        return [(u[1], d["roles"]) for _, u, d in self.g.edges(("net", net), data=True)]

    def degree(self, net: str) -> int:
        return self.g.degree(("net", net)) if self.g.has_node(("net", net)) else 0

    def is_rail(self, net: str) -> bool:
        return net in self.ground or net in self.supply

    def two_terminal(self, unit: Unit) -> tuple[str, str] | None:
        """The two nets of a two-terminal part (None if shorted/unconnected)."""
        nets = unit.pins.get("p", [])
        if unit.ctype in TWO_TERMINAL and len(nets) == 2 and nets[0] != nets[1]:
            return nets[0], nets[1]
        return None

    def of_type(self, *ctypes: str) -> list[Unit]:
        return [u for u in self.units.values() if u.ctype in ctypes]

    def net(self, unit: Unit, role: str) -> str | None:
        nets = unit.pins.get(role)
        return nets[0] if nets else None
