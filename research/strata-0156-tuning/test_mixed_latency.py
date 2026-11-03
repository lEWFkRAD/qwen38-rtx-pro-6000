#!/usr/bin/env python3
"""CPU-only interval/protocol audit; mocked native events never start an engine."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mixed_latency_cpu_audit", ROOT / "mixed_latency.py")
mixed = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mixed)


class IntervalTests(unittest.TestCase):
    def test_counts_only_inside_and_distinguishes_censored_gaps(self):
        result = mixed.interval_metrics([.5, 1.1, 1.4, 2.8, 3.5], 1, 3)
        self.assertEqual(result["observed_tokens"], 3)
        self.assertEqual(result["complete_intertoken_gaps"], 2)
        self.assertAlmostEqual(result["output_tokens_per_second"], 1.5)
        self.assertAlmostEqual(result["maximum_complete_intertoken_gap_ms"], 1400)
        self.assertAlmostEqual(result["maximum_silent_interval_ms"], 1400)

    def test_no_arrival_reports_whole_interval_silence(self):
        result = mixed.interval_metrics([.5, 3.5], 1, 3)
        self.assertEqual(result["observed_tokens"], 0)
        self.assertIsNone(result["maximum_complete_intertoken_gap_ms"])
        self.assertEqual(result["maximum_silent_interval_ms"], 2000)

    def test_single_arrival_does_not_invent_complete_gap(self):
        result = mixed.interval_metrics([1.2], 1, 3)
        self.assertEqual(result["observed_tokens"], 1)
        self.assertEqual(result["complete_intertoken_gaps"], 0)
        self.assertIsNone(result["maximum_complete_intertoken_gap_ms"])
        self.assertAlmostEqual(result["maximum_silent_interval_ms"], 1800)

    def test_observed_cross_boundary_pair_is_separate_from_clipped_silence(self):
        result = mixed.interval_metrics([.1, 1.9, 2.0], 1, 2.1)
        self.assertAlmostEqual(result["maximum_complete_intertoken_gap_ms"], 100)
        self.assertAlmostEqual(result["maximum_silent_interval_ms"], 900)
        self.assertAlmostEqual(result["maximum_observed_intertoken_gap_ending_in_interval_ms"], 1800)

    def test_invalid_intervals_and_clocks_refused(self):
        for arrivals, start, end in [([], 1, 1), ([], 2, 1), ([], 0, float("inf")),
                                     ([float("nan")], 0, 1), ([-1], 0, 1), ([.9, .8], 0, 1)]:
            with self.subTest(arrivals=arrivals, start=start, end=end), self.assertRaises(mixed.B.Refusal):
                mixed.interval_metrics(arrivals, start, end)


class Clock:
    now = 0.0


class FakeEngine:
    def __init__(self, events, clock):
        self.events = iter(events)
        self.clock = clock
        self.sent = []

    def send(self, line):
        self.sent.append((self.clock.now, line))

    def next(self, deadline):
        when, line = next(self.events)
        if when > deadline:
            raise mixed.B.Refusal("fake native deadline")
        self.clock.now = when
        return when, line


class FakeTokenizer:
    def decode(self, ids):
        return "MARKER" if ids == [99, 100] else "unexpected"


def successful_events():
    rows = []
    for slot in range(4):
        at = .1 * (slot + 1)
        rows.extend([(at, "T 42"), (at + .01, "DONE 1 100 5 1 length 0 0 0"),
                     (at + .02, f"BADM {slot} 1")])
    rows.extend((.8 + slot * .01, f"BT {slot} 42") for slot in range(4))
    rows.extend((1 + slot * .01, f"BT {slot} 42") for slot in range(4))
    rows.extend([(2.0, "T 99"), (2.01, "DONE 1 32076 1000 1 length 0 0 0"),
                 (2.02, "BADM 4 1"), (2.2, "BT 4 100"), (2.5, "BDONE 4 2 stop 400")])
    # A queued row may legitimately arrive between BSTOP submission and its
    # cancellation completion. The parser must count it, not assume instant stop.
    rows.extend((2.55 + slot * .01, f"BT {slot} 42") for slot in range(4))
    rows.extend((2.7 + slot * .01, f"BDONE {slot} 4 cancel 2000") for slot in range(4))
    return sorted(rows, key=lambda item: item[0])


class ProtocolTests(unittest.TestCase):
    def run_events(self, events=None, expected="MARKER"):
        clock = Clock()
        clock.now = 0
        engine = FakeEngine(events if events is not None else successful_events(), clock)
        backgrounds = [{"max_new": 1024, "tokens": [1, 2]} for _ in range(4)]
        joiner = {"id": "fresh32k", "tokens": [5, 6], "max_new": 64,
                  "prompt_tokens": 32076, "expected": expected}
        with patch.object(mixed.time, "monotonic", side_effect=lambda: clock.now):
            result = mixed.exercise(engine, backgrounds, joiner, FakeTokenizer(), 10, 2)
        return result, engine

    def test_success_waits_for_all_four_then_joins_and_drains_only_owned_slots(self):
        result, engine = self.run_events()
        self.assertEqual(result["infrastructure_status"], "passed")
        self.assertEqual(result["quality_status"], "passed")
        self.assertEqual(result["active_established_slots_at_join"], [0, 1, 2, 3])
        self.assertTrue(result["all_four_survived_until_joiner_completion"])
        self.assertTrue(result["all_owned_streams_cancelled_and_drained"])
        self.assertEqual([s["generated"] for s in result["established_streams"]], [4] * 4)
        joins = [(at, text) for at, text in engine.sent if text.startswith("BGEN 4 ")]
        self.assertEqual(len(joins), 1)
        self.assertAlmostEqual(joins[0][0], .83)
        self.assertAlmostEqual(result["joiner"]["ttft_ms"], 1170)
        self.assertEqual([line for _, line in engine.sent if line.startswith("BSTOP")],
                         ["BSTOP 0", "BSTOP 1", "BSTOP 2", "BSTOP 3"])

    def test_wrong_slot_admission_refused(self):
        events = successful_events()
        events = [(at, "BADM 3 1" if line == "BADM 0 1" else line) for at, line in events]
        with self.assertRaises(mixed.B.Refusal):
            self.run_events(events)

    def test_unadmitted_slot_token_refused(self):
        events = successful_events()
        events.insert(0, (.001, "BT 7 42"))
        with self.assertRaises(mixed.B.Refusal):
            self.run_events(events)

    def test_background_early_completion_refused(self):
        events = successful_events()
        events.append((1.5, "BDONE 0 3 stop 1000"))
        events.sort(key=lambda row: row[0])
        with self.assertRaises(mixed.B.Refusal):
            self.run_events(events)

    def test_no_joiner_continuation_refused(self):
        events = [(at, "BADM 4 0" if line == "BADM 4 1" else line) for at, line in successful_events()]
        with self.assertRaises(mixed.B.Refusal):
            self.run_events(events)

    def test_joiner_output_count_mismatch_refused(self):
        events = [(at, "BDONE 4 3 stop 400" if line == "BDONE 4 2 stop 400" else line)
                  for at, line in successful_events()]
        with self.assertRaises(mixed.B.Refusal):
            self.run_events(events)

    def test_background_natural_finish_after_join_is_still_unqualified(self):
        events = [(at, "BDONE 0 4 length 2000" if line == "BDONE 0 4 cancel 2000" else line)
                  for at, line in successful_events()]
        with self.assertRaises(mixed.B.Refusal):
            self.run_events(events)

    def test_wrong_marker_stays_quality_failure_with_clean_protocol(self):
        result, _ = self.run_events(expected="OTHER")
        self.assertEqual(result["infrastructure_status"], "passed")
        self.assertEqual(result["quality_status"], "failed")
        self.assertFalse(result["joiner"]["correct"])
        self.assertTrue(result["all_owned_streams_cancelled_and_drained"])


class ComparisonTests(unittest.TestCase):
    @staticmethod
    def arm(name, latency):
        return {"profile": name, "infrastructure_status": "passed", "owned_process_group_empty": True,
                "mixed": {"joiner": {"correct": True, "ttft_ms": latency,
                                      "admission_wall_ms": latency * 2, "completion_wall_ms": latency * 3},
                          "maximum_silent_interval_during_join_ms": latency / 2,
                          "all_four_survived_until_joiner_completion": True,
                          "all_owned_streams_cancelled_and_drained": True}}

    def test_lower_is_better_ratio_uses_observed_latency(self):
        result = mixed.comparisons([self.arm("old", 100), self.arm("new", 80)])[0]
        self.assertTrue(result["eligible_diagnostic_comparison"])
        self.assertAlmostEqual(result["ttft_ms"]["candidate_over_baseline"], .8)

    def test_marker_failure_or_incomplete_cleanup_blocks_comparison(self):
        for field in ["marker", "cleanup", "survival", "infra"]:
            candidate = self.arm("new", 80)
            if field == "marker":
                candidate["mixed"]["joiner"]["correct"] = False
            elif field == "cleanup":
                candidate["owned_process_group_empty"] = False
            elif field == "survival":
                candidate["mixed"]["all_four_survived_until_joiner_completion"] = False
            else:
                candidate["infrastructure_status"] = "failed"
            with self.subTest(field=field):
                result = mixed.comparisons([self.arm("old", 100), candidate])[0]
                self.assertFalse(result["eligible_diagnostic_comparison"])
                self.assertNotIn("ttft_ms", result)


if __name__ == "__main__":
    unittest.main()
