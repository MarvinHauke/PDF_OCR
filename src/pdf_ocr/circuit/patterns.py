"""Subcircuit patterns over a CircuitGraph.

Each pattern is an explicit, explainable rule over units (parts) and nets,
e.g. a current mirror is two transistors of the same type whose bases share a
net, whose emitters share a net, and one of which is diode-connected.

After matching, weak patterns that lie completely inside a stronger one are
marked `part_of` (the timing resistors of a 555 astable are not a separate
voltage divider; the feedback network of an op-amp stage isn't either).
Matches whose reading depends on context get `ambiguous: true` -- input for a
later arbitration step (Jev) and for review.
"""

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

from pdf_ocr.circuit.graph import CircuitGraph, Unit

TRANSISTOR_TYPES = ("bjt_npn", "bjt_pnp", "mosfet_n", "mosfet_p", "mosfet", "bjt", "jfet")
OPAMP_TYPES = ("opamp", "comparator_ic")
# A potentiometer with only two pins connected is a variable resistor
RESISTORS = ("resistor", "potentiometer")
CAPACITORS = ("capacitor", "capacitor_polarized")

# Higher rank wins the part_of resolution
RANK = {
    "timer_555_astable": 5, "timer_555_monostable": 5,
    "inverting_amp": 4, "non_inverting_amp": 4, "voltage_follower": 4, "integrator": 4,
    "opamp_stage": 4,
    "comparator": 4, "linear_regulator": 4, "rectifier_bridge": 4,
    "push_pull": 3.5,
    "current_mirror": 3, "differential_pair": 3, "emitter_follower": 3,
    "rc_lowpass": 2, "rc_highpass": 2, "voltage_divider": 2,
    "decoupling_network": 1.5,
    "decoupling_cap": 1,
}


@dataclass
class Match:
    type: str
    units: list[str]
    nets: list[str]
    ambiguous: bool = False
    notes: list[str] = field(default_factory=list)
    part_of: str | None = None
    id: str = ""

    def to_dict(self, graph: CircuitGraph) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "components": sorted(self.units),
            "nets": sorted(self.nets),
            "part_of": self.part_of,
            "ambiguous": self.ambiguous,
            "notes": self.notes,
        }


def _control(u: Unit, g: CircuitGraph):
    """(control, output, common) nets of a transistor: base/collector/emitter or gate/drain/source."""
    if u.ctype.startswith("bjt"):
        return g.net(u, "base"), g.net(u, "collector"), g.net(u, "emitter")
    return g.net(u, "gate"), g.net(u, "drain"), g.net(u, "source")


def _two_terminals_between(g: CircuitGraph, a: str, b: str, types) -> list[Unit]:
    out = []
    for u in g.of_type(*types):
        nets = g.two_terminal(u)
        if nets and set(nets) == {a, b}:
            out.append(u)
    return out


def _two_terminals_on(g: CircuitGraph, net: str, types) -> list[tuple[Unit, str]]:
    """(part, other net) for two-terminal parts of the given types touching net."""
    out = []
    for u in g.of_type(*types):
        nets = g.two_terminal(u)
        if nets and net in nets:
            out.append((u, nets[1] if nets[0] == net else nets[0]))
    return out


# -- passive patterns ----------------------------------------------------------


def decoupling_caps(g: CircuitGraph) -> list[Match]:
    out = []
    for u in g.of_type(*CAPACITORS):
        nets = g.two_terminal(u)
        if nets and {n in g.ground for n in nets} == {True, False} and all(g.is_rail(n) for n in nets):
            out.append(Match("decoupling_cap", [u.id], list(nets)))
    return out


def decoupling_networks(g: CircuitGraph) -> list[Match]:
    """All decoupling caps of one supply rail as a network (bulk and ceramic together).

    The netlist can't tell which cap sits next to which IC -- every cap between
    +12V and GND is on the same two nets -- so the rail is the grouping it supports.
    The single caps become children of their network (part_of, lower rank)."""
    by_rail: dict[str, list[Match]] = {}
    for m in decoupling_caps(g):
        rail = next(n for n in m.nets if n not in g.ground)
        by_rail.setdefault(rail, []).append(m)
    out = []
    for rail, caps in sorted(by_rail.items()):
        units = sorted(u for m in caps for u in m.units)
        grounds = sorted({n for m in caps for n in m.nets if n in g.ground})
        values = Counter(g.units[u].value or "?" for u in units)
        summary = ", ".join(f"{n}× {v}" for v, n in sorted(values.items(), key=lambda x: (-x[1], x[0])))
        out.append(Match("decoupling_network", units, [rail, *grounds], notes=[summary]))
    return out


def voltage_dividers(g: CircuitGraph) -> list[Match]:
    """Two resistors in series whose middle node (tap) is used by something else.

    The tap must carry exactly two resistors: nets with more (pull-up buses,
    summing nodes) would otherwise yield every pair as a "divider"."""
    out, seen = [], set()
    tap_nets = {n for u in g.of_type(*RESISTORS) for n in (g.two_terminal(u) or ())}
    for tap in tap_nets:
        if g.is_rail(tap):
            continue
        rs = _two_terminals_on(g, tap, RESISTORS)
        if len(rs) != 2 or g.degree(tap) < 3:  # only the two resistors: plain series, no tap
            continue
        for (r1, a), (r2, b) in combinations(rs, 2):
            key = frozenset((r1.id, r2.id))
            if a == b or key in seen:
                continue
            seen.add(key)
            rail_end = g.is_rail(a) or g.is_rail(b)
            out.append(Match("voltage_divider", [r1.id, r2.id], [a, tap, b],
                             ambiguous=not rail_end,
                             notes=[] if rail_end else ["no end on a supply/ground rail (attenuator or feedback network?)"]))
    return out


def rc_filters(g: CircuitGraph) -> list[Match]:
    """First-order RC: R in series + C to ground (low-pass) or C in series + R to ground (high-pass)."""
    out = []
    nodes = {n for u in g.of_type(*RESISTORS, *CAPACITORS) for n in (g.two_terminal(u) or ())}
    for mid in nodes:
        if g.is_rail(mid) or g.degree(mid) < 3:  # output must go somewhere
            continue
        rs = _two_terminals_on(g, mid, RESISTORS)
        cs = _two_terminals_on(g, mid, CAPACITORS)
        for r, r_other in rs:
            for c, c_other in cs:
                if r_other in g.ground and not g.is_rail(c_other):
                    out.append(Match("rc_highpass", [c.id, r.id], [c_other, mid, r_other]))
                elif c_other in g.ground and not g.is_rail(r_other):
                    out.append(Match("rc_lowpass", [r.id, c.id], [r_other, mid, c_other]))
    return out


# -- transistor patterns -------------------------------------------------------


def transistor_pairs(g: CircuitGraph) -> list[Match]:
    out = []
    ts = g.of_type(*TRANSISTOR_TYPES)
    for q1, q2 in combinations(ts, 2):
        if q1.ctype != q2.ctype:
            continue
        b1, c1, e1 = _control(q1, g)
        b2, c2, e2 = _control(q2, g)
        if None in (b1, c1, e1, b2, c2, e2):
            continue
        if b1 == b2 and (c1 == b1) != (c2 == b2):
            common = e1 == e2 or _via_resistors_to_same_net(g, e1, e2)
            if common:
                notes = [] if e1 == e2 else ["emitter/source degeneration resistors"]
                out.append(Match("current_mirror", [q1.id, q2.id], [b1, c1, c2, e1, e2], notes=notes))
        elif e1 == e2 and not g.is_rail(e1) and b1 != b2 and c1 != c2:
            out.append(Match("differential_pair", [q1.id, q2.id], [b1, b2, c1, c2, e1]))
    return out


def _via_resistors_to_same_net(g: CircuitGraph, e1: str, e2: str) -> bool:
    far1 = {other for _, other in _two_terminals_on(g, e1, RESISTORS)}
    far2 = {other for _, other in _two_terminals_on(g, e2, RESISTORS)}
    return bool(far1 & far2 & (g.ground | g.supply))


def followers(g: CircuitGraph) -> list[Match]:
    """Emitter/source follower: collector/drain on a rail, output at the emitter/source."""
    out = []
    for q in g.of_type(*TRANSISTOR_TYPES):
        b, c, e = _control(q, g)
        if None in (b, c, e) or b == e or g.is_rail(e) or g.is_rail(b):
            continue
        if g.is_rail(c) and g.degree(e) >= 2:
            mos = not q.ctype.startswith("bjt")
            out.append(Match("emitter_follower", [q.id], [b, c, e], ambiguous=mos,
                             notes=["source follower (or high-side switch)"] if mos else []))
    return out


def _polarity(q: Unit, g: CircuitGraph) -> str:
    """Known type, or for an unknown follower: collector/drain on a negative rail -> PNP/P-channel."""
    if q.ctype not in ("bjt", "mosfet"):
        return q.ctype
    c = _control(q, g)[1] or ""
    negative = c.split("/")[-1].startswith("-") or "vee" in c.lower() or "vss" in c.lower()
    return f"{q.ctype}_{'p' if negative else 'n'}" if q.ctype == "mosfet" else \
        ("bjt_pnp" if negative else "bjt_npn")


def push_pull(g: CircuitGraph, followers_found: list[Match]) -> list[Match]:
    """Complementary output stage: an NPN and a PNP follower whose emitters meet
    (directly or via small resistors on a common output net)."""
    out = []
    by_type = {}
    for m in followers_found:
        q = g.units[m.units[0]]
        by_type.setdefault(_polarity(q, g), []).append(q)
    pairs = [("bjt_npn", "bjt_pnp"), ("mosfet_n", "mosfet_p")]
    for hi, lo in pairs:
        for qn in by_type.get(hi, []):
            for qp in by_type.get(lo, []):
                en, ep = _control(qn, g)[2], _control(qp, g)[2]
                shared = en == ep
                rs = []
                if not shared:
                    far_n = {o: r for r, o in _two_terminals_on(g, en, RESISTORS)}
                    far_p = {o: r for r, o in _two_terminals_on(g, ep, RESISTORS)}
                    common = set(far_n) & set(far_p)
                    if common:
                        n = common.pop()
                        rs = [far_n[n].id, far_p[n].id]
                        shared = True
                if shared:
                    out.append(Match("push_pull", [qn.id, qp.id] + rs, [en, ep]))
    return out


# -- op-amp patterns -----------------------------------------------------------

FEEDBACK_TYPES = ("resistor", "potentiometer", "capacitor", "capacitor_polarized", "inductor",
                  "diode", "zener")


def _feedback_parts(g: CircuitGraph, opamp: Unit, a: str, b: str) -> list[Unit]:
    """Passive parts touching both nets a and b (any pin), e.g. a 3-pin gain pot."""
    return [u for u in g.of_type(*FEEDBACK_TYPES)
            if u.id != opamp.id and a in u.nets() and b in u.nets()]


def _feedback_path(g: CircuitGraph, opamp: Unit, start: str, goal: str, max_parts: int = 4) -> list[str] | None:
    """Parts on a path from start to goal through passives, diodes and followers
    (transistor base/gate -> emitter/source), never through rails or the op-amp itself.
    Finds loops closed behind an output stage."""
    frontier, seen = [(start, [])], {start}
    while frontier:
        net, path = frontier.pop(0)
        if len(path) >= max_parts:
            continue
        for uid, roles in g.members(net):
            u = g.units[uid]
            if uid == opamp.id or uid in path:
                continue
            if u.ctype in FEEDBACK_TYPES:
                nexts = [n for n in u.nets() if n != net]
            elif u.ctype in TRANSISTOR_TYPES and roles & {"base", "gate"}:
                _, _, e = _control(u, g)
                nexts = [e] if e else []
            else:
                continue
            for n in nexts:
                if n == goal:
                    return path + [uid]
                if n not in seen and not g.is_rail(n):
                    seen.add(n)
                    frontier.append((n, path + [uid]))
    return None



def opamp_stages(g: CircuitGraph) -> list[Match]:
    out = []
    for u in g.of_type(*OPAMP_TYPES):
        inp, inn, o = g.net(u, "in_p"), g.net(u, "in_n"), g.net(u, "out")
        if None in (inp, inn, o):
            continue
        if u.ctype == "comparator_ic":
            out.append(Match("comparator", [u.id], [inp, inn, o]))
            continue
        if o == inn:
            out.append(Match("voltage_follower", [u.id], [inp, o]))
            continue
        rf = _two_terminals_between(g, o, inn, RESISTORS)
        cf = _two_terminals_between(g, o, inn, CAPACITORS)
        inputs = [(r, s) for r, s in _two_terminals_on(g, inn, RESISTORS) if r not in rf and s != o]
        feedback = _feedback_parts(g, u, o, inn)
        loop = None if feedback else _feedback_path(g, u, o, inn)
        if loop:
            out.append(Match("opamp_stage", [u.id] + loop, [inp, inn, o], ambiguous=True,
                             notes=["negative feedback closed through " + ", ".join(loop)
                                    + " (e.g. behind an output stage)"]))
            continue
        if not feedback:
            hyst = _two_terminals_between(g, o, inp, RESISTORS)
            out.append(Match("comparator", [u.id] + [r.id for r in hyst], [inp, inn, o],
                             notes=["positive feedback: Schmitt trigger"] if hyst else []))
            continue
        if not rf and not cf:
            # negative feedback through a pot, diodes, ... -- not one of the patterns yet
            kinds = sorted({f.ctype for f in feedback})
            out.append(Match("opamp_stage", [u.id] + [f.id for f in feedback], [inp, inn, o],
                             ambiguous=True, notes=[f"feedback via {', '.join(kinds)}"]))
            continue
        signal_inputs = [(r, s) for r, s in inputs if not g.is_rail(s)]
        if rf and signal_inputs:
            parts = [u.id] + [r.id for r in rf] + [r.id for r, _ in signal_inputs]
            notes = ["summing amplifier"] if len(signal_inputs) > 1 else []
            out.append(Match("inverting_amp", parts, [inp, inn, o] + [s for _, s in signal_inputs], notes=notes))
        elif rf and inputs and not g.is_rail(inp):
            parts = [u.id] + [r.id for r in rf] + [r.id for r, _ in inputs]
            out.append(Match("non_inverting_amp", parts, [inp, inn, o] + [s for _, s in inputs]))
        elif cf and not rf and signal_inputs:
            parts = [u.id] + [c.id for c in cf] + [r.id for r, _ in signal_inputs]
            out.append(Match("integrator", parts, [inp, inn, o]))
        # other feedback topologies (active filters, etc.) are left for later patterns
    return out


# -- power / timing ------------------------------------------------------------


def rectifier_bridges(g: CircuitGraph) -> list[Match]:
    out, seen = [], set()
    diodes = g.of_type("diode")
    ends = {d.id: (g.net(d, "anode"), g.net(d, "cathode")) for d in diodes}
    cathodes_on, anodes_on = {}, {}
    for a, k in ends.values():
        anodes_on[a] = anodes_on.get(a, 0) + 1
        cathodes_on[k] = cathodes_on.get(k, 0) + 1
    for combo in combinations(diodes, 4):
        key = frozenset(d.id for d in combo)
        if key in seen:
            continue
        anodes = [ends[d.id][0] for d in combo]
        cathodes = [ends[d.id][1] for d in combo]
        pos = {n for n in cathodes if cathodes.count(n) == 2}
        neg = {n for n in anodes if anodes.count(n) == 2}
        if len(pos) != 1 or len(neg) != 1:
            continue
        ac = [n for n in set(anodes + cathodes) if n not in pos | neg]
        (p,), (n,) = pos, neg
        # Clamp diodes from many signals to the rails form the same topology; a real
        # bridge has exactly two diodes on each DC net
        if cathodes_on[p] != 2 or anodes_on[n] != 2:
            continue
        if len(ac) == 2 and all(anodes.count(x) == 1 and cathodes.count(x) == 1 for x in ac):
            seen.add(key)
            rails = p in g.supply and (n in g.supply or n in g.ground)
            out.append(Match("rectifier_bridge", sorted(key), ac + [p, n], ambiguous=rails,
                             notes=["DC nets are named supply rails: could be clamp diodes"] if rails else []))
    return out


def regulators(g: CircuitGraph) -> list[Match]:
    out = []
    for u in g.of_type("linear_regulator"):
        vin, vout = g.net(u, "vin"), g.net(u, "vout")
        caps = []
        for net in (vin, vout):
            if net:
                caps += [c.id for c, other in _two_terminals_on(g, net, CAPACITORS) if other in g.ground]
        out.append(Match("linear_regulator", [u.id] + caps, [n for n in (vin, vout) if n]))
    return out


def timers_555(g: CircuitGraph) -> list[Match]:
    out = []
    for u in g.of_type("timer_555"):
        trig, thr, dis = g.net(u, "trig"), g.net(u, "thr"), g.net(u, "dis")
        if None in (trig, thr):
            continue
        cap = [c for c, other in _two_terminals_on(g, thr, CAPACITORS) if other in g.ground]
        if trig == thr and cap:
            rb = _two_terminals_between(g, dis, thr, RESISTORS) if dis else []
            ra = [r for r, other in _two_terminals_on(g, dis, RESISTORS) if other in g.supply] if dis else []
            out_net = g.net(u, "out")
            rout = _two_terminals_between(g, out_net, thr, RESISTORS) if out_net else []
            if rb and ra:
                parts = [u.id, ra[0].id, rb[0].id, cap[0].id]
                out.append(Match("timer_555_astable", parts, [trig, dis]))
            elif rout:
                out.append(Match("timer_555_astable", [u.id, rout[0].id, cap[0].id], [trig, out_net],
                                 notes=["50% duty variant (timing resistor from OUT)"]))
        elif trig != thr and dis == thr and cap:
            r = [r for r, other in _two_terminals_on(g, thr, RESISTORS) if other in g.supply]
            if r:
                out.append(Match("timer_555_monostable", [u.id, r[0].id, cap[0].id], [trig, thr]))
    return out


PATTERNS = [timers_555, opamp_stages, regulators, rectifier_bridges, transistor_pairs, followers,
            rc_filters, voltage_dividers, decoupling_networks, decoupling_caps]


def find_subcircuits(g: CircuitGraph) -> list[Match]:
    matches = [m for pattern in PATTERNS for m in pattern(g)]
    matches += push_pull(g, [m for m in matches if m.type == "emitter_follower"])
    counters: dict[str, int] = {}
    for m in matches:
        counters[m.type] = counters.get(m.type, 0) + 1
        m.id = f"{m.type}#{counters[m.type]}"
    # part_of: weak matches fully contained in a stronger one
    for weak in matches:
        for strong in matches:
            if RANK[strong.type] > RANK[weak.type] and set(weak.units) <= set(strong.units):
                weak.part_of = strong.id
                break
    return matches


def analyze(g: CircuitGraph) -> dict:
    matches = find_subcircuits(g)
    return {
        "circuit": g.name,
        "components": {u.id: {"type": u.ctype, "value": u.value} for u in g.units.values()},
        "nets": {n: sorted(uid for uid, _ in g.members(n)) for n in
                 sorted({n for u in g.units.values() for n in u.nets()})},
        "subcircuits": [m.to_dict(g) for m in matches],
    }
