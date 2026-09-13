import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LegacyWithdrawalTests(unittest.TestCase):
    def test_published_rankings_are_withdrawn_during_v2_audit(self):
        page = (ROOT / "index.html").read_text()
        latest = json.loads((ROOT / "data/results/latest.json").read_text())
        self.assertIn('data-benchmark-status="withdrawn"', page)
        self.assertIn("Rankings withdrawn", page)
        self.assertEqual(latest["benchmark_status"], "withdrawn_unreliable")
        self.assertIs(latest["ranking_valid"], False)
        self.assertTrue(latest["withdrawal_reason"])


if __name__ == "__main__":
    unittest.main()
