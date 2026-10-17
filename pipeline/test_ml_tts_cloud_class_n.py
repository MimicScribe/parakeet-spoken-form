"""ml_tts_cloud per-class planning (--class-n) and the natural-date filter. python3 test_ml_tts_cloud_class_n.py"""
import collections
import unittest

import ml_tts as T
import ml_tts_cloud as CL


def _row(*readings):
    return {"spans": [{"kind": "date", "reading": x} for x in readings]}


class NaturalDate(unittest.TestCase):
    def test_month_name_cardinal_day_kept(self):
        self.assertTrue(CL.natural_date(_row("zeventien", "maart", "tweeduizendzeventien"), "nl"))
        self.assertTrue(CL.natural_date(_row("twaalf", "maart"), "nl"))

    def test_numeric_month_dropped(self):
        self.assertFalse(CL.natural_date(_row("zeventien", "drie", "tweeduizendzeventien"), "nl"))

    def test_ordinal_day_dropped(self):
        self.assertFalse(CL.natural_date(_row("zesde", "maart"), "nl"))
        self.assertFalse(CL.natural_date(_row("drieëntwintigste", "mei"), "nl"))

    def test_rows_without_dates_pass(self):
        self.assertTrue(CL.natural_date({"spans": [{"kind": "phone", "reading": "nul zes"}]}, "nl"))

    def test_other_language_month_only(self):  # no ORD_DAY entry: only the month name is required
        self.assertTrue(CL.natural_date(_row("dodici", "marzo"), "it"))
        self.assertFalse(CL.natural_date(_row("dodici", "tre"), "it"))


class PlanPerClass(unittest.TestCase):
    def test_counts_per_class_and_unique(self):
        rows = CL.plan_per_class("nl", {"phone": 6, "date_num": 6}, 5, {}, "t", natural_dates=True)
        c = collections.Counter(r["cls"] for r in rows)
        self.assertEqual(c["phone"], 6)
        self.assertGreaterEqual(c["date_num"], 5)
        self.assertEqual(len({r["text"] for r in rows}), len(rows))
        self.assertTrue(all(CL.natural_date(r, "nl") for r in rows))

    def test_count_quota_values(self):
        T.COUNT_QUOTA = {73: 1, 88: 1}
        try:
            rows = CL.plan_per_class("nl", {"count": 8}, 5, {}, "t")
        finally:
            T.COUNT_QUOTA = {}
        vals = {s["text"].split()[0] for r in rows for s in r["spans"] if s["kind"] == "int"}
        self.assertTrue(vals and vals <= {"73", "88"}, vals)


class DutchSetListed(unittest.TestCase):
    def test_nl_set_listed_and_targeted(self):
        # red-on-base: fails before tts9nl was added (build4 would never read the Dutch fill)
        import mix_build as MB

        self.assertIn(("tts9nl", "tts9nl_judge"), MB.TTS_SETS)
        self.assertIn("tts9nl", MB.TTS_TARGETED)
        self.assertEqual(len({t for t, _ in MB.TTS_SETS}), len(MB.TTS_SETS))


if __name__ == "__main__":
    unittest.main()
