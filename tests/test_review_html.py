"""Converter from circuit analysis / gold files to KiCanvas review pages."""

import html
import json
import re

from pdf_ocr.circuit.review_html import build_page, collect_sources, find_root, to_symbol_groups

RESULT = {
    "circuit": "demo",
    "components": {"R1": {"type": "resistor", "value": "10k"}, "Q1": {"type": "bjt_npn", "value": "BC547"}},
    "nets": {},
    "subcircuits": [
        {"id": "push_pull#1", "type": "push_pull", "components": ["Q1", "Q2"], "nets": [],
         "part_of": None, "ambiguous": False, "notes": []},
        {"id": "emitter_follower#1", "type": "emitter_follower", "components": ["Q1"], "nets": [],
         "part_of": "push_pull#1", "ambiguous": True, "notes": ["source follower", "or switch"]},
    ],
}


def test_analysis_groups():
    data = to_symbol_groups(RESULT)
    assert data["version"] == 1 and data["source"] == "analysis" and data["title"] == "demo"
    top, child = data["groups"]
    assert top == {"id": "push_pull#1", "label": "push_pull#1", "kind": "push_pull", "refs": ["Q1", "Q2"]}
    assert child["status"] == "ambiguous"
    assert child["parent"] == "push_pull#1"
    assert child["description"] == ["source follower", "or switch"]
    assert data["parts"]["R1"] == {"kind": "resistor", "value": "10k"}
    assert "decoupling_network" in data["kinds"] and data["kinds"] == sorted(data["kinds"])


def test_gold_prefills_matches_and_appends_hand_added_entries():
    gold = {"circuit": "demo", "reviewed": True,
            "subcircuits": [{"type": "push_pull", "components": ["Q2", "Q1"]},
                            {"type": "voltage_divider", "components": ["R1", "R2"]},
                            {"type": "inverting_amp", "components": ["U1.A", "R3"]}]}
    data = to_symbol_groups(RESULT, gold)
    assert data["source"] == "analysis"
    assert [g["id"] for g in data["groups"]] == ["push_pull#1", "emitter_follower#1", "gold#1", "gold#2"]
    assert data["groups"][0]["review"] == "correct"
    assert "review" not in data["groups"][1]
    assert data["groups"][3]["label"] == "gold#2 inverting_amp"
    assert data["groups"][3]["refs"] == ["U1.A", "R3"] and data["groups"][3]["review"] == "correct"
    assert data["groups"][3]["added"] is True and "added" not in data["groups"][0]
    assert "R1" in data["parts"]  # classification still comes from the analysis


def test_unreviewed_draft_prefills_nothing():
    draft = {"circuit": "demo", "reviewed": False,
             "subcircuits": [{"type": "push_pull", "components": ["Q1", "Q2"]}]}
    assert all("review" not in g for g in to_symbol_groups(RESULT, draft)["groups"])


def sheet(sheetfiles=()):
    refs = "".join(f'(sheet (property "Sheetname" "s{i}") (property "Sheetfile" "{f}"))'
                   for i, f in enumerate(sheetfiles))
    return f'(kicad_sch (version 20231120) (text "a < b & c") {refs})'


def test_sources_of_a_hierarchy_with_a_reused_subsheet(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "root.kicad_sch").write_text(sheet(["sub/amp.kicad_sch", "sub/amp.kicad_sch", "psu.kicad_sch"]))
    (tmp_path / "sub" / "amp.kicad_sch").write_text(sheet())
    (tmp_path / "psu.kicad_sch").write_text(sheet())
    root = find_root(tmp_path)
    assert root.name == "root.kicad_sch"
    names = [name for name, _ in collect_sources(root)]
    assert names == ["root.kicad_sch", "sub/amp.kicad_sch", "psu.kicad_sch"]  # reused sheet once


def test_page_rules(tmp_path):
    bundle = tmp_path / "kicanvas.js"
    bundle.write_text("")
    page = build_page([("root.kicad_sch", sheet())], to_symbol_groups(RESULT), bundle)
    assert "a &lt; b &amp; c" in page  # schematic text escaped
    assert page.count("<kicanvas-groups>") == 1
    assert page.index("<kicanvas-embed") < page.index("<kicanvas-groups>") < page.index("</kicanvas-embed>")
    script = re.search(r"<script[^>]*>", page).group(0)
    assert 'type="module"' not in script and bundle.resolve().as_uri() in script
    groups = re.search(r"<kicanvas-groups>(.*)</kicanvas-groups>", page, re.S).group(1)
    assert json.loads(html.unescape(groups))["groups"][0]["id"] == "push_pull#1"
