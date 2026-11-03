#!/usr/bin/env python3
"""CPU-only audit of benchmark accounting and safety helpers; never starts engines.

The gaps found during the initial comparison are fixed and required to pass.
Every case here uses synthetic data and no GPU or engine subprocess.
Run with PYTHONDONTWRITEBYTECODE=1 python3 test_benchmark_strata.py -v.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("benchmark_under_cpu_audit", ROOT / "benchmark_strata.py")
bench = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(bench)


class MetricsTests(unittest.TestCase):
    def test_aggregate_uses_every_observed_stream_count(self):
        rows = [[1 + k / 10 for k in range(8 + slot)] for slot in range(8)]
        result = bench.output_metrics(rows, 0, [4] * 8, trim_seconds=0)
        self.assertEqual(result["observed_output_tokens"], sum(range(8, 16)))
        self.assertEqual(result["whole_request_output_tokens_per_second"], sum(range(8, 16)) / 4)
        common = result["common_window"]
        self.assertTrue(common["valid"])
        self.assertEqual(common["per_stream_output_tokens"], list(range(8, 16)))
        self.assertAlmostEqual(common["aggregate_output_tokens_per_second"], sum(range(8, 16)) / 3)

    def test_common_window_excludes_admission_and_completion_tails(self):
        rows = [[.1, .4, 1.0, 1.4, 1.8, 2.0, 2.5], [.6, 1.2, 1.6, 2.2]]
        result = bench.output_metrics(rows, 0, [3, 2.5], trim_seconds=.25, live_from=1.0)
        common = result["common_window"]
        self.assertAlmostEqual(common["start_ms"], 1250)
        self.assertAlmostEqual(common["end_ms"], 2250)
        self.assertEqual(common["per_stream_output_tokens"], [3, 2])
        self.assertAlmostEqual(common["aggregate_output_tokens_per_second"], 5)

    def test_empty_stream_has_no_valid_common_window(self):
        result = bench.output_metrics([[1, 2], []], 0, [3, 3])
        self.assertFalse(result["common_window"]["valid"])
        self.assertEqual(result["observed_output_tokens"], 2)

    def test_arrival_time_reversal_is_refused(self):
        with self.assertRaises(bench.Refusal):
            bench.output_metrics([[1, .9]], 0, [2])

    def test_no_shared_interval_is_not_valid(self):
        result = bench.output_metrics([[.1], [1.9]], 0, [1.5, 2.5])
        self.assertFalse(result["common_window"]["valid"])

    def test_meaningful_common_window_is_speed_eligible(self):
        rows = [[.1 + i * .02 for i in range(80)] for _ in range(8)]
        common = bench.output_metrics(rows, 0, [2.0] * 8, live_from=.1)["common_window"]
        self.assertTrue(common["valid"])
        self.assertTrue(common["speed_eligible"])
        self.assertGreaterEqual(common["seconds"], .5)
        self.assertTrue(all(count >= 32 for count in common["per_stream_output_tokens"]))

    def test_sub_half_second_window_cannot_claim_speed(self):
        rows = [[.01 + i * .003 for i in range(128)] for _ in range(8)]
        common = bench.output_metrics(rows, 0, [.49] * 8, trim_seconds=0)["common_window"]
        self.assertTrue(common["valid"])
        self.assertGreaterEqual(min(common["per_stream_output_tokens"]), 32)
        self.assertFalse(common["speed_eligible"])

    def test_nonfinite_or_outside_lifetime_event_clock_refused(self):
        examples = [([[float("nan")]], 0, [2]), ([[1]], 0, [float("inf")]),
                    ([[1]], -1, [2]), ([[.5]], 1, [2]), ([[3]], 0, [2])]
        for rows, born, ends in examples:
            with self.subTest(rows=rows, born=born, ends=ends), self.assertRaises(bench.Refusal):
                bench.output_metrics(rows, born, ends)

    def test_shared_window_with_one_token_each_is_not_speed_eligible(self):
        # All eight emitted 256 tokens, but only ONE per slot happened after the
        # last admission and before the first completion, with trimmed margins.
        # This is possible when earlier streams advance during later prefill.
        events = []
        for slot in range(7):
            t = .01 * (slot + 1)
            events.extend([(t, "T 42"), (t + .0001, "DONE 1 10 1 1 length 0 0 0"),
                           (t + .0002, f"BADM {slot} 1")])
            events.extend((.5 + i * .001, f"BT {slot} 42") for i in range(254))
            events.append((1.7, f"BT {slot} 42"))
        events.extend([(1.399, "T 42"), (1.3995, "DONE 1 10 1 1 length 0 0 0"),
                       (1.4, "BADM 7 1")])
        events.extend((1.4001 + i * .0007, "BT 7 42") for i in range(254))
        events.append((1.7, "BT 7 42"))
        events.extend((2.0, f"BDONE {slot} 256 length 1000") for slot in range(8))
        events.sort(key=lambda item: item[0])

        class FakeEngine:
            def __init__(self):
                self.records = iter(events)
                self.sent = []

            def send(self, line):
                self.sent.append(line)

            def next(self, deadline):
                return next(self.records)

        class FakeTokenizer:
            def decode(self, ids):
                return ""

        engine = FakeEngine()
        cases = [{"tokens": [1], "max_new": 256} for _ in range(8)]
        with patch.object(bench.time, "monotonic", return_value=0):
            result = bench.NativeEngine.throughput(engine, cases, FakeTokenizer(), seconds=10)
        self.assertEqual(result["peak_live_slots"], 8)
        self.assertTrue(result["all_output_caps_reached"])
        self.assertEqual(result["metrics"]["common_window"]["per_stream_output_tokens"], [1] * 8)
        self.assertFalse(result["eligible_for_speed_comparison"])


class AnswerTests(unittest.TestCase):
    case = {"expected": "a JSON object", "checker": "json",
            "expected_json": {"ready": True, "count": 23}}

    def test_json_spacing_and_key_order_do_not_matter(self):
        self.assertTrue(bench.check_answer(self.case, '{ "count": 23, "ready": true }'))

    def test_json_missing_extra_or_wrong_value_fails(self):
        for text in ['{"ready":true}', '{"ready":true,"count":23,"extra":0}',
                     '{"ready":false,"count":23}', '```json\n{}\n```']:
            with self.subTest(text=text):
                self.assertFalse(bench.check_answer(self.case, text))

    def test_json_integer_does_not_match_boolean(self):
        self.assertFalse(bench.check_answer(self.case, '{"ready":1,"count":23}'))

    def test_duplicate_json_keys_fail(self):
        self.assertFalse(bench.check_answer(self.case, '{"ready":false,"ready":true,"count":23}'))

    def test_nested_json_types_and_arrays_are_strict(self):
        case = {"expected": "a JSON object", "checker": "json",
                "expected_json": {"items": [{"enabled": False, "value": 4}], "valid": True}}
        self.assertTrue(bench.check_answer(case, '{"valid":true,"items":[{"value":4,"enabled":false}]}'))
        for text in ['{"valid":true,"items":[{"value":4,"enabled":0}]}',
                     '{"valid":1,"items":[{"value":4,"enabled":false}]}',
                     '{"valid":true,"items":[{"value":4,"enabled":true,"enabled":false}]}',
                     '{"valid":true,"items":{"value":4,"enabled":false}}']:
            with self.subTest(text=text):
                self.assertFalse(bench.check_answer(case, text))


class DoneTests(unittest.TestCase):
    def test_valid_record_and_appended_fields(self):
        result = bench.done_fields("DONE 6 100 2.5 10.0 length 4 5 10 optional=value")
        self.assertEqual(result["generated"], 6)
        self.assertEqual(result["drafts_accepted"], 4)

    def test_failed_finish_and_impossible_acceptance_refused(self):
        for line in ["DONE 6 100 2.5 10 failed 4 5 0", "DONE 6 100 2.5 10 stop 6 5 0"]:
            with self.subTest(line=line), self.assertRaises(bench.Refusal):
                bench.done_fields(line)

    def test_nan_timing_refused(self):
        with self.assertRaises(bench.Refusal):
            bench.done_fields("DONE 6 100 nan 10 length 4 5 0")

    def test_infinite_timing_refused(self):
        with self.assertRaises(bench.Refusal):
            bench.done_fields("DONE 6 100 2.5 inf length 4 5 0")

    def test_negative_timing_refused(self):
        with self.assertRaises(bench.Refusal):
            bench.done_fields("DONE 6 100 -2.5 10 length 4 5 0")

    def test_negative_observed_counts_refused(self):
        for line in ["DONE -1 100 2.5 10 length 0 0 0", "DONE 6 -100 2.5 10 length 0 0 0",
                     "DONE 6 100 2.5 10 length 0 0 -1", "DONE 6 100 2.5 10 length -1 0 0"]:
            with self.subTest(line=line), self.assertRaises(bench.Refusal):
                bench.done_fields(line)


class ProfileTests(unittest.TestCase):
    def test_permitted_prefill_override_does_not_change_capacity(self):
        original = ["--max-context", "131072", "--batch", "8", "--prefill", "8192"]
        result = bench.apply_profile_args(original, {"arg_overrides": {"--prefill": "2048"}})
        self.assertEqual(bench.arg_value(result, "--prefill"), "2048")
        self.assertEqual(bench.arg_value(result, "--max-context"), "131072")
        self.assertEqual(bench.arg_value(result, "--batch"), "8")
        self.assertEqual(bench.arg_value(original, "--prefill"), "8192")

    def test_protected_model_capacity_and_kv_overrides_refused(self):
        for flag in ["--batch", "--slots", "--max-context", "--kv", "--kv-resident", "--native",
                     "--control-vector-scaled", "--mtp"]:
            with self.subTest(flag=flag), self.assertRaises(bench.Refusal):
                bench.apply_profile_args([], {"arg_overrides": {flag: "wrong"}})

    def test_lsa_environment_and_non_strata_overrides_refused(self):
        for env in [{"STRATA_LSA_MAPS": "bad"}, {"LD_LIBRARY_PATH": "/tmp"}]:
            with self.subTest(env=env), self.assertRaises(bench.Refusal):
                bench.launch_env({"lib_dirs": [str(ROOT)]}, {}, {"env": env})

    def test_existing_matched_bundle_directory_used_exclusively(self):
        directory = ROOT / "official-engine-0.1.56"
        env, public = bench.launch_env({"lib_dirs": ["/unwanted/old-libs"]}, {},
                                       {"lib_dirs": [str(directory)], "env": {"STRATA_CKPT_POOL": "1"}})
        self.assertEqual(env["LD_LIBRARY_PATH"], str(directory))
        self.assertEqual(public["STRATA_CKPT_POOL"], "1")

    def test_empty_library_path_cannot_fall_back_to_system(self):
        with self.assertRaises(bench.Refusal):
            bench.launch_env({"lib_dirs": [""]}, {}, {})

    def test_nonexistent_library_directory_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(bench.Refusal):
                bench.launch_env({"lib_dirs": [str(Path(tmp) / "absent")]}, {}, {})

    def test_relative_library_directory_refused(self):
        with self.assertRaises(bench.Refusal):
            bench.library_directories({"lib_dirs": ["."]}, {})

    def test_library_manifest_records_and_checks_full_file_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "libexample.so.1"
            path.write_bytes(b"library-one")
            manifest = bench.library_manifest([tmp])
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["path"], str(path))
            self.assertEqual(manifest[0]["size"], len(b"library-one"))
            self.assertEqual(len(manifest[0]["sha256"]), 64)
            profile = {"lib_dirs": [tmp], "library_manifest": manifest}
            bench.verify_library_manifest(profile, {})
            # A same-size, finite-size edit still changes the full-byte hash.
            path.write_bytes(b"library-two")
            with self.assertRaises(bench.Refusal):
                bench.verify_library_manifest(profile, {})

    def test_library_manifest_requires_at_least_one_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(bench.Refusal):
                bench.library_manifest([tmp])


class ComparisonTests(unittest.TestCase):
    @staticmethod
    def arm(role, quality="passed", rate=200):
        return {"pair": 1, "role": role, "infrastructure_status": "passed",
                "quality_status": quality, "unexecuted_quality_cases": [],
                "expected_quality_ids": ["same_fixture"], "owned_process_group_empty": True,
                "cases": [{"id": "same_fixture", "quality_status": quality,
                           "output_ids": [42], "wall_ms": 100}],
                "sustained": {"eligible_for_speed_comparison": True,
                              "metrics": {"common_window": {"aggregate_output_tokens_per_second": rate},
                                          "whole_request_output_tokens_per_second": rate / 2}}}

    def test_paired_speed_uses_observed_ratio(self):
        result = bench.comparisons([self.arm("control", rate=200), self.arm("candidate", rate=220)])
        self.assertEqual(result["sustained"]["eligible_pairs"], 1)
        self.assertAlmostEqual(result["sustained"]["paired_common_speed_ratios"][0], 1.1)

    def test_shared_baseline_failure_stays_failed_without_new_regression(self):
        result = bench.comparisons([self.arm("control", "failed"), self.arm("candidate", "failed")])
        self.assertEqual(result["quality_regressions"], [])
        case = result["pairs"][0]["quality"][0]
        self.assertFalse(case["control_correct"])
        self.assertFalse(case["candidate_correct"])
        self.assertTrue(result["pairs"][0]["both_arms_complete"])

    def test_new_failure_is_reported_as_regression(self):
        result = bench.comparisons([self.arm("control"), self.arm("candidate", "failed")])
        self.assertEqual(result["quality_regressions"], ["same_fixture"])

    def test_incomplete_unsafe_or_failed_arm_cannot_qualify_speed(self):
        modifications = [{"infrastructure_status": "failed"}, {"owned_process_group_empty": False},
                         {"unexecuted_quality_cases": ["same_fixture"], "cases": []},
                         {"quality_status": "incomplete"}]
        for change in modifications:
            candidate = self.arm("candidate")
            candidate.update(change)
            with self.subTest(change=change):
                result = bench.comparisons([self.arm("control"), candidate])
                self.assertFalse(result["pairs"][0]["both_arms_complete"])
                self.assertFalse(result["pairs"][0]["sustained"]["eligible"])
                self.assertNotIn("sustained", result)


if __name__ == "__main__":
    unittest.main()
