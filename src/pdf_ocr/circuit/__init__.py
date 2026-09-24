"""Circuit graphs and graph-based subcircuit detection.

netlist.py    KiCad netlist (kicadxml) -> Circuit (components + nets)
components.py component type and normalized pin roles from symbol/part names
graph.py      Circuit -> networkx graph (component nodes, net nodes, role-labeled edges)
patterns.py   subcircuit patterns (voltage divider, current mirror, 555 astable, ...)
"""
