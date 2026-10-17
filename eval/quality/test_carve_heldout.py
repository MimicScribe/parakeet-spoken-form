"""carve_heldout.plan: --kinds mode (common classes, cap = share of training, richest groups first, --avoid).
python3 test_carve_heldout.py"""
import unittest

import carve_heldout as C


def row(day, i, kinds):
    return {"id": f"{day}-0900-PLENARY_es_{i}", "src": "vox", "spans": [{"kind": k} for k in kinds]}


def day_rows(day, n_unit=0, n_dec=0, extra=(), n_rows=1):
    rs = [row(day, 0, ["unit"] * n_unit + ["dec"] * n_dec + list(extra))]
    rs += [row(day, i, []) for i in range(1, n_rows)]
    return rs


class KindsMode(unittest.TestCase):
    def test_richest_first_within_ten_percent_of_training(self):
        rows = day_rows("20190101", n_unit=12) + day_rows("20190102", n_unit=3) + day_rows("20190103", n_unit=5)
        rows += [r for d in range(10, 30) for r in day_rows(f"201902{d}", n_unit=10)]  # 200 more unit spans
        add, have, total = C.plan("es", rows, set(), 30, 0.10, only=["unit"])
        self.assertEqual(total["unit"], 220)
        self.assertLessEqual(have["unit"], 22)  # 10% of 220 training spans
        self.assertIn("vox:20190101", add)  # the 12-span day first
        self.assertEqual(have["unit"], 22)  # 12 + 10, then nothing else fits under the cap

    def test_avoid_and_cap_checks_every_listed_kind(self):
        rows = day_rows("20190101", n_unit=20, extra=["phone"])  # holds a phone: never carved
        rows += day_rows("20190102", n_unit=8, n_dec=30)  # would take dec past its cap
        rows += day_rows("20190103", n_unit=6)
        rows += [r for d in range(10, 30) for r in day_rows(f"201902{d}", n_unit=4, n_dec=2)]
        add, have, total = C.plan("es", rows, set(), 30, 0.10, only=["unit", "dec"], avoid={"phone"})
        self.assertNotIn("vox:20190101", add)
        self.assertNotIn("vox:20190102", add)
        self.assertIn("vox:20190103", add)
        self.assertLessEqual(have["dec"], int(total["dec"] * 0.10))

    def test_held_group_counts_whole_and_is_not_recarved(self):
        rows = day_rows("20190101", n_unit=4, n_rows=3)
        rows[1]["spans"] = [{"kind": "unit"}]  # a second unit span in another row of the held-out day
        rows += [r for d in range(10, 20) for r in day_rows(f"201902{d}", n_unit=10)]
        held = {rows[2]["id"]}  # one id of the day is listed: build4 holds out the whole day
        add, have, _ = C.plan("es", rows, held, 30, 0.10, only=["unit"])
        self.assertNotIn("vox:20190101", add)
        self.assertEqual(have["unit"] - sum(1 for g in add for _ in range(10)), 5)

    def test_counts_in_gate_classes(self):
        price = {"id": "20190101-0900-PLENARY_es_0", "src": "vox", "text": "cuesta tres coma cinco euros",
                 "spans": [{"kind": "dec", "text": "3,5", "reading": "tres coma cinco"}] * 3}
        rows = [price] + [r for d in range(10, 30) for r in day_rows(f"201902{d}", n_dec=2)]
        for r in rows[1:]:
            r["text"] = "son dos coma cinco"
            r["spans"] = [{"kind": "dec", "text": "2,5", "reading": "dos coma cinco"}] * 2
        _, _, raw = C.plan("es", rows, set(), 30, 0.10, only=["dec"])
        add, _, gate = C.plan("es", rows, set(), 30, 0.10, only=["dec"], cls=C.gate_class("es"))
        self.assertEqual(raw["dec"], 43)
        self.assertEqual(gate["dec"], 40)  # "tres coma cinco euros" is a price in the census
        self.assertNotIn("vox:20190101", add)

    def test_rare_mode_unchanged_by_default(self):
        rows = day_rows("20190101", extra=["roman"] * 3) + [r for d in range(10, 30) for r in day_rows(f"201902{d}",
                                                                                                      extra=["roman"])]
        add, have, _ = C.plan("es", rows, set(), 20, 0.25)
        self.assertLessEqual(have["roman"], int(23 * 0.25))
        self.assertTrue(add)
        add2, _, _ = C.plan("es", rows, set(), 20, 0.25, only=["unit"])
        self.assertFalse(add2)  # nothing of the kind: nothing carved


if __name__ == "__main__":
    unittest.main()
