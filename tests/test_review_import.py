"""KiCanvas review export -> gold file and graph dataset; round trip through the review page."""

import json

import pytest

from pdf_ocr.circuit.patterns import analyze
from pdf_ocr.circuit.review_html import to_symbol_groups
from pdf_ocr.circuit.review_import import graph_record, import_review, load_exports, merge_review
from test_patterns import R, build, load

RESULT = {
    "circuit": "demo",
    "components": {r: {"type": "x", "value": ""} for r in ["Q1", "Q2", "R1", "R2", "R3", "R4", "C1"]},
    "subcircuits": [
        {"id": "push_pull#1", "type": "push_pull", "components": ["Q1", "Q2"], "part_of": None},
        {"id": "emitter_follower#1", "type": "emitter_follower", "components": ["Q1"],
         "part_of": "push_pull#1"},
        {"id": "voltage_divider#1", "type": "voltage_divider", "components": ["R1", "R2"], "part_of": None},
        {"id": "rc_lowpass#1", "type": "rc_lowpass", "components": ["C1", "R3"], "part_of": None},
    ],
}


def exported(result=RESULT, gold=None, **verdicts):
    """The review page's groups JSON with the user's verdicts applied, as KiCanvas exports it."""
    data = to_symbol_groups(result, gold)
    for group in data["groups"]:
        if group["id"] in verdicts:
            if verdicts[group["id"]] is None:
                group.pop("review", None)
            else:
                group["review"] = verdicts[group["id"]]
    return data


def test_partial_review_replaces_the_init_gold_draft():
    draft = {"circuit": "demo", "reviewed": False, "subcircuits": [
        {"type": "push_pull", "components": ["Q1", "Q2"]},
        {"type": "voltage_divider", "components": ["R1", "R2"]},
        {"type": "rc_lowpass", "components": ["C1", "R3"]},
        {"type": "voltage_divider", "components": ["R4", "R1"]},  # added by hand
    ]}
    gold, counts = merge_review(exported(**{"push_pull#1": "correct", "voltage_divider#1": "wrong"}),
                                RESULT, draft, "demo.groups.json")
    assert counts["correct"] == 1 and counts["wrong"] == 1 and counts["open"] == 1
    assert gold["subcircuits"] == [{"type": "push_pull", "components": ["Q1", "Q2"]},
                                   {"type": "voltage_divider", "components": ["R4", "R1"]}]  # as written
    assert gold["rejected"] == [{"type": "voltage_divider", "components": ["R1", "R2"]}]
    assert gold["reviewed"] is False  # rc_lowpass#1 has no verdict yet
    assert gold["review_imports"][0]["file"] == "demo.groups.json"


def test_second_round_prefills_and_completes():
    first, _ = merge_review(exported(**{"push_pull#1": "correct", "voltage_divider#1": "wrong"}),
                            RESULT, None, "a.json")
    page = to_symbol_groups(RESULT, first)
    reviews = {g["id"]: g.get("review") for g in page["groups"]}
    assert reviews == {"push_pull#1": "correct", "emitter_follower#1": None,
                       "voltage_divider#1": "wrong", "rc_lowpass#1": None}

    second, counts = merge_review(exported(gold=first, **{"rc_lowpass#1": "correct"}), RESULT, first, "b.json")
    assert second["reviewed"] is True and counts["open"] == 0
    assert [e["type"] for e in second["subcircuits"]] == ["push_pull", "rc_lowpass"]
    assert len(second["review_imports"]) == 2


def test_children_count_only_when_their_parent_is_wrong():
    # parent correct: its child is covered, even if marked correct too
    gold, _ = merge_review(exported(**{"push_pull#1": "correct", "emitter_follower#1": "correct",
                                       "voltage_divider#1": "wrong", "rc_lowpass#1": "wrong"}),
                           RESULT, None, "x.json")
    assert gold["reviewed"] is True
    assert [e["type"] for e in gold["subcircuits"]] == ["push_pull"]
    # parent wrong: the child needs its own verdict
    gold, counts = merge_review(exported(**{"push_pull#1": "wrong", "voltage_divider#1": "wrong",
                                            "rc_lowpass#1": "wrong"}), RESULT, None, "x.json")
    assert gold["reviewed"] is False and counts["open"] == 1
    gold, _ = merge_review(exported(**{"push_pull#1": "wrong", "emitter_follower#1": "correct",
                                       "voltage_divider#1": "wrong", "rc_lowpass#1": "wrong"}),
                           RESULT, None, "x.json")
    assert gold["reviewed"] is True
    assert gold["subcircuits"] == [{"type": "emitter_follower", "components": ["Q1"]}]


def test_hand_added_entries_round_trip_and_can_be_rejected():
    gold = {"circuit": "demo", "reviewed": True, "rejected": [],
            "subcircuits": [{"type": "push_pull", "components": ["Q1", "Q2"]},
                            {"type": "voltage_divider", "components": ["R4", "R1"]}]}
    page = to_symbol_groups(RESULT, gold)
    extra = page["groups"][-1]
    assert extra["id"] == "gold#1" and extra["review"] == "correct" and extra["refs"] == ["R4", "R1"]
    new, _ = merge_review(exported(gold=gold, **{"gold#1": "wrong"}), RESULT, gold, "x.json")
    assert {"type": "voltage_divider", "components": ["R1", "R4"]} in new["rejected"]
    assert all(e["components"] != ["R1", "R4"] for e in new["subcircuits"])


def test_gold_page_exports_are_refused():
    with pytest.raises(ValueError, match="analysis page"):
        merge_review({"version": 1, "title": "demo", "source": "gold", "groups": []}, RESULT, None, "x")


def test_load_exports_picks_newest_per_circuit(tmp_path):
    old = tmp_path / "demo.groups.json"
    old.write_text(json.dumps({"version": 1, "title": "demo", "groups": [], "n": 1}))
    newer = tmp_path / "demo.groups (1).json"
    newer.write_text(json.dumps({"version": 1, "title": "demo", "groups": [], "n": 2}))
    (tmp_path / "other.json").write_text("{\"not\": \"groups\"}")
    import os
    os.utime(old, (1, 1))
    exports = load_exports(tmp_path)
    assert list(exports) == ["demo"] and exports["demo"][1]["n"] == 2


def test_import_writes_gold_and_graph(tmp_path):
    g = build(("R1", R, {"1": "+5V", "2": "TAP"}), ("R2", R, {"1": "TAP", "2": "GND"}), load("TAP"))
    result = analyze(g)
    [divider] = [s for s in result["subcircuits"] if s["type"] == "voltage_divider"]
    export = exported(result, **{divider["id"]: "correct"})
    gold_dir = tmp_path / "training_data" / "circuits" / "gold"
    gold = import_review(tmp_path / "test.groups.json", export, g, result, tmp_path / "test.netlist.xml",
                         gold_dir)
    assert json.loads((gold_dir / "test.json").read_text()) == gold
    graph = json.loads((gold_dir.parent / "graphs" / "test.json").read_text())
    assert graph["reviewed"] is True
    assert {u["id"] for u in graph["units"]} >= {"R1", "R2"}
    assert {"name": "GND", "rail": "ground"} in graph["nets"]
    assert {"unit": "R1", "net": "TAP", "roles": ["p"]} in graph["edges"]
    assert graph["subcircuits"] == [{"type": "voltage_divider", "units": ["R1", "R2"], "label": "positive"}]
    assert graph == graph_record(g, gold, graph["netlist"])


def test_groups_created_in_kicanvas_become_hand_added_entries():
    export = exported(**{"push_pull#1": "wrong"})
    export["groups"] += [{"id": "new#1", "refs": ["Q2", "R4"], "kind": "push_pull", "review": "correct"},
                         {"id": "new#2", "refs": ["R3", "C1"], "review": "correct"}]
    gold, _ = merge_review(export, RESULT, None, "x.json")
    assert {"type": "push_pull", "components": ["Q2", "R4"]} in gold["subcircuits"]
    assert {"type": "unlabeled", "components": ["C1", "R3"]} in gold["subcircuits"]
    page = to_symbol_groups(RESULT, gold)  # next round: shown as gold#n, marked correct
    extras = [g for g in page["groups"] if g["id"].startswith("gold#")]
    assert [(g["kind"], g["review"]) for g in extras] == [("push_pull", "correct"), ("unlabeled", "correct")]


def test_unknown_kind_warns(capsys):
    export = exported()
    export["groups"].append({"id": "new#1", "refs": ["R1", "R2"], "kind": "volt_divider", "review": "correct"})
    merge_review(export, RESULT, None, "x.json")
    assert "unknown kind 'volt_divider'" in capsys.readouterr().out
