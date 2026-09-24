"""Component types and normalized pin roles.

Classification looks at the symbol library and part name first (Device:R,
Amplifier_Operational:*), then keywords/description, then the reference
prefix -- many projects use their own libraries (e.g. `bitaxe:Q_NMOS_...`).
"""

import re

from pdf_ocr.circuit.netlist import Component

IGNORED = {"mechanical", "testpoint"}  # not part of the circuit's function

# (type, regex over "lib:part description keywords"), first match wins
RULES = [
    ("mechanical", r"^mechanical:|fiducial|mountinghole|^graphic:|logo|nettie|^jumper:solderjumper"),
    ("testpoint", r"testpoint"),
    ("potentiometer", r"^device:r_pot|potentiometer|trimmer"),
    ("resistor", r"^device:r(_|$)|^device:r_us|^device:r_small"),
    ("capacitor_polarized", r"^device:(c_polarized|cp)(_|\s|$)|electrolytic|(^|[^n])polarized capacitor"),
    ("capacitor", r"^device:c(_|$)|^device:c_small"),
    ("inductor", r"^device:(l|l_core|l_small|ferritebead)(_|$)|inductor"),
    ("led", r"^device:led|:led(_|$)"),
    ("zener", r"d_zener|zener"),
    ("diode", r"^device:d(_|$)|^device:d_schottky|diode|schottky"),
    ("timer_555", r"(^|[^0-9])(ne|lm|tlc|icm7|na|sa)?555|(^|[^0-9])556([^0-9]|$)|^timer:"),
    ("comparator_ic", r"^comparator:|comparator"),
    ("opamp", r"^amplifier_operational:|operational amplifier|op-?amp|(^|:)(tl07|tl08|lm358|lm324|ne5532|opa)"),
    ("linear_regulator", r"^regulator_linear:|(^|[^a-z])(78\d\d|79\d\d|lm317|lm337|ams1117)|ldo|linear regulator"),
    ("bjt_npn", r"q_npn|npn|(^|[^0-9a-z])(bc54[6-9]|bc337|bc817|bd13[579]|2n3904|2n2222|2n4401|mmbt3904|tip3[13]|tip41)"),
    ("bjt_pnp", r"q_pnp|pnp|(^|[^0-9a-z])(bc55[6-9]|bc327|bc807|bd1(36|38|40)|2n3906|2n2907|2n4403|mmbt3906|tip3[24]|tip42)"),
    ("mosfet_n", r"q_nmos|nmos|n-channel|n-mos"),
    ("mosfet_p", r"q_pmos|pmos|p-channel|p-mos"),
    ("jfet", r"q_njfet|q_pjfet|jfet"),
    ("crystal", r"^device:crystal|crystal|resonator"),
    ("connector", r"^connector|conn_"),
    ("switch", r"^switch:|push button|(^|[^a-z])switch"),
    ("fuse", r"fuse"),
    ("thermistor", r"thermistor"),
]

PREFIX = {"R": "resistor", "C": "capacitor", "L": "inductor", "D": "diode", "Y": "crystal",
          "J": "connector", "P": "connector", "U": "ic", "IC": "ic", "Q": "transistor"}


DISCRETE = {"bjt_npn", "bjt_pnp", "mosfet_n", "mosfet_p", "jfet", "diode", "zener", "led",
            "resistor", "capacitor", "capacitor_polarized", "inductor"}


def classify(comp: Component) -> str:
    text = f"{comp.lib}:{comp.part} {comp.description} {comp.keywords} {comp.value}".lower()
    prefix = re.match(r"[A-Za-z]+", comp.ref)
    prefix = prefix.group(0).upper() if prefix else ""
    for ctype, pattern in RULES:
        if re.search(pattern, text):
            # "NPN darlington array", "LED driver": an IC, not the discrete part
            if ctype in DISCRETE and prefix in ("U", "IC"):
                return "ic"
            return ctype
    return PREFIX.get(prefix, "other")


def refine_transistor(pin_functions: list[str]) -> str:
    """Custom-library transistors: tell MOSFET from BJT by pin names (polarity unknown)."""
    names = {re.sub(r"_\d+$", "", f.strip().lower()) for f in pin_functions}
    if names & {"g", "gate"} and names & {"d", "drain", "s", "source"}:
        return "mosfet"
    if names & {"b", "base"} and names & {"c", "collector", "e", "emitter"}:
        return "bjt"
    return "transistor"


TWO_TERMINAL = {"resistor", "capacitor", "capacitor_polarized", "inductor", "potentiometer",
                "crystal"}
DIODES = {"diode", "zener", "led"}

ROLE_NAMES = {
    "bjt": {"b": "base", "base": "base", "c": "collector", "collector": "collector",
            "e": "emitter", "emitter": "emitter"},
    "mosfet": {"g": "gate", "gate": "gate", "d": "drain", "drain": "drain",
               "s": "source", "source": "source"},
    "opamp": {"+": "in_p", "in+": "in_p", "+in": "in_p", "non-inverting": "in_p", "ni": "in_p",
              "-": "in_n", "in-": "in_n", "-in": "in_n", "inverting": "in_n", "inv": "in_n",
              "out": "out", "~": "out", "output": "out",
              "v+": "supply", "v-": "supply", "vcc": "supply", "vee": "supply",
              "vdd": "supply", "vss": "supply"},
    "timer_555": {"tr": "trig", "trig": "trig", "trigger": "trig", "thr": "thr", "thresh": "thr",
                  "threshold": "thr", "dis": "dis", "disch": "dis", "discharge": "dis",
                  "q": "out", "out": "out", "cv": "cv", "cont": "cv", "control": "cv",
                  "r": "reset", "reset": "reset", "vcc": "supply", "gnd": "gnd"},
    "linear_regulator": {"vi": "vin", "vin": "vin", "in": "vin", "input": "vin",
                         "vo": "vout", "vout": "vout", "out": "vout", "output": "vout",
                         "gnd": "gnd", "adj": "adj", "adj/gnd": "adj"},
    "diode": {"k": "cathode", "a": "anode"},
}

# KiCad convention for Device:D / LED symbols without pin names
DIODE_PIN_NUMBERS = {"1": "cathode", "2": "anode"}


def family(ctype: str) -> str:
    if ctype.startswith("bjt"):
        return "bjt"
    if ctype.startswith("mosfet") or ctype == "jfet":
        return "mosfet"
    if ctype in ("opamp", "comparator_ic"):
        return "opamp"
    if ctype in DIODES:
        return "diode"
    return ctype


def pin_role(ctype: str, pin: str, function: str) -> str:
    """Normalized role of a pin, e.g. 'base', 'in_n', 'thr'. Two-terminal parts: 'p'."""
    if ctype in TWO_TERMINAL:
        return "p"
    fam = family(ctype)
    name = re.sub(r"_\d+$", "", (function or "").strip().lower())  # "VOUT_5" -> "vout"
    roles = ROLE_NAMES.get(fam, {})
    if name in roles:
        return roles[name]
    if fam == "diode" and not name:
        return DIODE_PIN_NUMBERS.get(pin, f"pin{pin}")
    return name or f"pin{pin}"
