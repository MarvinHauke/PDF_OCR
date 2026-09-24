"""Pattern library tests on tiny hand-written circuits.

Negative cases are the false positives seen on real KiCad projects: series
resistors without a tap, pull-up buses, 555 timing resistors, feedback through
a potentiometer, clamp diodes to the rails.
"""

from pdf_ocr.circuit.graph import CircuitGraph
from pdf_ocr.circuit.netlist import Circuit, Component, Net, Node
from pdf_ocr.circuit.patterns import find_subcircuits

R = ("Device", "R")
C = ("Device", "C")
D = ("Device", "D")
NPN = ("Device", "Q_NPN_BCE")
PNP = ("Device", "Q_PNP_BCE")
OPAMP = ("Amplifier_Operational", "TL071")
TIMER = ("Timer", "NE555P")
POT = ("Device", "R_Potentiometer")


def build(*parts) -> CircuitGraph:
    """parts: (ref, (lib, part), {pin_or_function: net}). Two-terminal parts use pins "1"/"2"."""
    components, nets = {}, {}
    for ref, (lib, part), pins in parts:
        components[ref] = Component(ref, lib, part, part)
        for i, (fn, net) in enumerate(pins.items(), start=1):
            pin, function = (fn, "") if fn.isdigit() else (str(i), fn)
            nets.setdefault(net, []).append(Node(ref, pin, function))
    return CircuitGraph(Circuit("test", components, [Net(n, nodes) for n, nodes in nets.items()]))


def top(g):
    """Top-level matches as {(type, frozenset(units))}."""
    return {(m.type, frozenset(m.units)) for m in find_subcircuits(g) if not m.part_of}


def types(g):
    return [t for t, _ in top(g)]


def load(net, ref="J1"):
    """Something using a net (a connector pin), so a tap isn't a dead end."""
    return (ref, ("Connector", "Conn_01x01"), {"1": net})


# -- voltage divider -------------------------------------------------------------

def test_voltage_divider():
    g = build(("R1", R, {"1": "+5V", "2": "MID"}), ("R2", R, {"1": "MID", "2": "GND"}), load("MID"))
    assert ("voltage_divider", frozenset({"R1", "R2"})) in top(g)


def test_series_resistors_without_tap_are_no_divider():
    g = build(("R1", R, {"1": "A", "2": "MID"}), ("R2", R, {"1": "MID", "2": "B"}), load("A"), load("B", "J2"))
    assert "voltage_divider" not in types(g)


def test_pullup_bus_is_no_divider():
    g = build(("R1", R, {"1": "+5V", "2": "BUS"}), ("R2", R, {"1": "BUS", "2": "GND"}),
              ("R3", R, {"1": "BUS", "2": "X"}), load("BUS"))
    assert "voltage_divider" not in types(g)


def test_decoupling_cap_and_rc_lowpass():
    g = build(("C1", C, {"1": "+5V", "2": "GND"}),
              ("R1", R, {"1": "IN", "2": "OUT"}), ("C2", C, {"1": "OUT", "2": "GND"}),
              load("IN"), load("OUT", "J2"))
    assert ("decoupling_cap", frozenset({"C1"})) in top(g)
    assert ("rc_lowpass", frozenset({"R1", "C2"})) in top(g)


# -- 555 ------------------------------------------------------------------------

def test_555_astable_and_timing_resistors_are_not_a_divider():
    g = build(
        ("U1", TIMER, {"TR": "T", "THR": "T", "DIS": "D", "Q": "OUT", "VCC": "+9V", "GND": "GND", "R": "+9V"}),
        ("R1", R, {"1": "+9V", "2": "D"}), ("R2", R, {"1": "D", "2": "T"}), ("C1", C, {"1": "T", "2": "GND"}),
        load("OUT"),
    )
    found = top(g)
    assert ("timer_555_astable", frozenset({"U1", "R1", "R2", "C1"})) in found
    assert "voltage_divider" not in types(g)


# -- transistors ----------------------------------------------------------------

def test_current_mirror():
    g = build(("Q1", NPN, {"B": "B", "C": "B", "E": "GND"}), ("Q2", NPN, {"B": "B", "C": "OUT", "E": "GND"}),
              ("R1", R, {"1": "+12V", "2": "B"}), load("OUT"))
    assert ("current_mirror", frozenset({"Q1", "Q2"})) in top(g)


def test_differential_pair():
    g = build(("Q1", NPN, {"B": "INP", "C": "C1", "E": "TAIL"}), ("Q2", NPN, {"B": "INN", "C": "C2", "E": "TAIL"}),
              ("R1", R, {"1": "TAIL", "2": "-12V"}), ("R2", R, {"1": "+12V", "2": "C1"}),
              ("R3", R, {"1": "+12V", "2": "C2"}), load("INP"), load("INN", "J2"))
    assert ("differential_pair", frozenset({"Q1", "Q2"})) in top(g)
    assert "current_mirror" not in types(g)


def test_push_pull_output_stage():
    g = build(("Q1", NPN, {"B": "BN", "C": "+12V", "E": "EN"}), ("Q2", PNP, {"B": "BP", "C": "-12V", "E": "EP"}),
              ("R1", R, {"1": "EN", "2": "OUT"}), ("R2", R, {"1": "EP", "2": "OUT"}),
              load("OUT"), load("BN", "J2"), load("BP", "J3"))
    assert ("push_pull", frozenset({"Q1", "Q2", "R1", "R2"})) in top(g)


# -- op-amps --------------------------------------------------------------------

def opamp(inp, inn, out, ref="U1"):
    return (ref, OPAMP, {"+": inp, "-": inn, "OUT": out, "V+": "+12V", "V-": "-12V"})


def test_inverting_amp_absorbs_its_resistor_network():
    g = build(opamp("GND", "INN", "OUT"), ("R1", R, {"1": "IN", "2": "INN"}), ("R2", R, {"1": "OUT", "2": "INN"}),
              load("IN"), load("OUT", "J2"))
    assert ("inverting_amp", frozenset({"U1", "R1", "R2"})) in top(g)
    assert "voltage_divider" not in types(g)


def test_non_inverting_amp():
    g = build(opamp("IN", "INN", "OUT"), ("R1", R, {"1": "INN", "2": "GND"}), ("R2", R, {"1": "OUT", "2": "INN"}),
              load("IN"), load("OUT", "J2"))
    assert ("non_inverting_amp", frozenset({"U1", "R1", "R2"})) in top(g)


def test_comparator_without_feedback():
    g = build(opamp("IN", "REF", "OUT"), load("IN"), load("REF", "J2"), load("OUT", "J3"))
    assert ("comparator", frozenset({"U1"})) in top(g)


def test_feedback_through_pot_is_not_a_comparator():
    g = build(opamp("IN", "INN", "OUT"), ("RV1", POT, {"1": "OUT", "2": "INN", "3": "GND"}),
              load("IN"), load("OUT", "J2"))
    assert "comparator" not in types(g)
    assert "opamp_stage" in types(g)


def test_feedback_behind_output_stage_is_not_a_comparator():
    g = build(opamp("GND", "INN", "DRV"), ("Q1", NPN, {"B": "DRV", "C": "+12V", "E": "OUT"}),
              ("R1", R, {"1": "OUT", "2": "INN"}), ("R2", R, {"1": "IN", "2": "INN"}),
              ("R3", R, {"1": "OUT", "2": "GND"}), load("IN"))
    assert "comparator" not in types(g)


# -- diodes ---------------------------------------------------------------------

def test_rectifier_bridge():
    g = build(("D1", D, {"K": "DCP", "A": "AC1"}), ("D2", D, {"K": "DCP", "A": "AC2"}),
              ("D3", D, {"K": "AC1", "A": "DCN"}), ("D4", D, {"K": "AC2", "A": "DCN"}),
              load("AC1"), load("AC2", "J2"), load("DCP", "J3"), load("DCN", "J4"))
    assert ("rectifier_bridge", frozenset({"D1", "D2", "D3", "D4"})) in top(g)


def test_clamp_diodes_on_many_signals_are_no_bridge():
    parts = []
    for i in range(3):  # three signals, each clamped to both rails
        parts += [(f"DA{i}", D, {"K": "+12V", "A": f"S{i}"}), (f"DB{i}", D, {"K": f"S{i}", "A": "-12V"}),
                  load(f"S{i}", f"J{i}")]
    g = build(*parts)
    assert "rectifier_bridge" not in types(g)
