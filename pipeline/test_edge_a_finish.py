"""Final selection of edge-pack clips (edge_a_finish). python3 test_edge_a_finish.py  (controls; red-on-base: control)"""
import edge_a_finish as F


def row(i, ctc=True, basic=True, strict=None, cons="en.fin.bps", group="clone:US"):
    return {"id": i, "construction": cons, "group": group, "ctc_ok": ctc, "basic_ok": basic, "strict": strict or [], "family": "fin", "text": "t",
            "duration": 5.0, "voice": "v", "lang": "en"}


rows = [row("a"), row("b"), row("c", ctc=False, strict=[{"ok": False, "why": "wdel", "m": {"del": 200, "wdel": -9, "wsub": 20}, "low": [["wdel", "points", -9.0]]}]),
        row("i", ctc=False, strict=[{"ok": False, "why": "wdel", "m": {"del": 200, "wdel": -9, "wsub": 1.0}, "low": [["wdel", "points", -9.0], ["wsub", "a>b", 1.0]]}]),
        row("d", ctc=False, strict=[{"ok": False, "why": "wdel", "low": [["wdel", "seventy", -9.0]]}]),
        row("e", ctc=False, strict=[{"ok": False, "why": "wsub", "low": [["wsub", "points>joints", 1.0]]}]),
        row("f", cons="en.fin.money_scale"), row("g", group="chirp:en-GB"), row("h", ctc=False, basic=False)]
ver = {"i": {"a": True, "b": True}, "a": {"a": True, "b": True}, "b": {"a": False, "b": False}, "c": {"a": True, "b": True}, "d": {"a": True, "b": True},
       "e": {"a": True, "b": True}, "f": {"a": True, "b": False}, "g": {"a": True, "b": True}, "h": {"a": True, "b": True}}
keep, second, st = F.finalize(rows, ver, drop_groups=("chirp:en-GB",), unanimous=("en.fin.money_scale",), numkeys={"seventy", "five", "hundred"})
ids = {r["id"] for r in keep}
assert ids == {"a", "c"}, ids                       # c: only a unit noun was not heard, both transcripts agree -> reader weak, kept
assert [r["id"] for r in keep if r.get("reader_weak")] == ["c"]
assert {r["id"] for r in second} == {"d", "e", "i"}      # number word / swap probes failed: never weak, listed for a second look only
assert st["dropped_by_pilot_gate"] == 1 and st["kept"] == 2 and st["blind_fail"] >= 1, st
# without the number keys nothing is relaxed
keep2, _, _ = F.finalize(rows, ver, drop_groups=("chirp:en-GB",), unanimous=("en.fin.money_scale",))
assert {r["id"] for r in keep2} == {"a"}
print("test_edge_a_finish ok")
