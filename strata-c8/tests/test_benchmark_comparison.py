"""Offline checks for public benchmark comparison qualification and diagnostics."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_c8.py'
SPEC = importlib.util.spec_from_file_location('benchmark_c8', SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def result(rate=40.0):
    return {'dataset': benchmark.DATASET, 'model': 'qwen3.8-flash-next-custom-iq3-s',
            'concurrency': 8, 'max_tokens': 512, 'rounds_requested': 2, 'aux_cpus': '0',
            'errors': [], 'summary': {'all_common_windows_qualified': True,
                                     'median_session_steady_tokens_per_s': rate},
            'rounds': [{'round': r, 'full_run_valid': True, 'common_window': {'valid': True},
                        'streams': [{'request': i, 'checks_passed': True,
                                     'sha256': hashlib.sha256(f'{r}:{i}'.encode()).hexdigest()}
                                    for i in range(8)]} for r in (1, 2)]}


class ComparisonTests(unittest.TestCase):
    def compare(self, baseline, candidate):
        candidate['comparison'] = benchmark.compare_results(baseline, candidate)
        return candidate['comparison']

    def assert_rejected(self, candidate):
        self.assertFalse(candidate['comparison']['comparison_qualified'])
        self.assertIsNone(candidate['comparison']['steady_session_gain_percent'])
        self.assertFalse(benchmark.result_succeeded(candidate, comparison_requested=True))

    def test_matching_outputs_and_qualified_common_windows_succeed(self):
        baseline, candidate = result(), result(60.0)
        comparison = self.compare(baseline, candidate)
        self.assertTrue(comparison['comparison_qualified'])
        self.assertEqual(comparison['complete_output_hash_matches'], 16)
        self.assertEqual(comparison['steady_session_gain_percent'], 50.0)
        self.assertTrue(benchmark.result_succeeded(candidate, comparison_requested=True))

    def test_external_work_after_baseline_common_window_does_not_disqualify_it(self):
        baseline, candidate = result(), result(60.0)
        baseline['rounds'][0]['full_run_valid'] = False
        self.assertTrue(self.compare(baseline, candidate)['comparison_qualified'])

    def test_one_hash_mismatch_blocks_gain_and_comparison_success(self):
        baseline, candidate = result(), result(60.0)
        candidate['rounds'][1]['streams'][7]['sha256'] = 'f' * 64
        comparison = self.compare(baseline, candidate)
        self.assertTrue(comparison['both_common_window_arms_qualified'])
        self.assertEqual(comparison['complete_output_hash_matches'], 15)
        self.assertFalse(comparison['all_outputs_match'])
        self.assert_rejected(candidate)

    def test_auxiliary_cpu_settings_must_match(self):
        baseline, candidate = result(), result(60.0)
        candidate['aux_cpus'] = '1'
        self.assertFalse(self.compare(baseline, candidate)['dataset_compatible'])
        self.assert_rejected(candidate)

    def test_missing_auxiliary_cpu_settings_are_not_implicitly_compatible(self):
        baseline, candidate = result(), result(60.0)
        del baseline['aux_cpus']
        del candidate['aux_cpus']
        self.assertFalse(self.compare(baseline, candidate)['dataset_compatible'])
        self.assert_rejected(candidate)

    def test_each_arm_needs_qualified_common_windows(self):
        for side in ('baseline', 'candidate'):
            with self.subTest(side=side):
                baseline, candidate = result(), result(60.0)
                selected = baseline if side == 'baseline' else candidate
                selected['rounds'][0]['common_window']['valid'] = False
                self.compare(baseline, candidate)
                self.assert_rejected(candidate)

    def test_missing_duplicate_or_invalid_hashes_cannot_qualify(self):
        variants = []
        missing = result(60.0)
        missing['rounds'][1]['streams'].pop()
        variants.append(missing)
        duplicate = result(60.0)
        duplicate['rounds'][1]['streams'][7] = copy.deepcopy(duplicate['rounds'][1]['streams'][6])
        variants.append(duplicate)
        invalid = result(60.0)
        invalid['rounds'][1]['streams'][7]['sha256'] = None
        variants.append(invalid)
        for candidate in variants:
            with self.subTest(candidate=candidate):
                self.compare(result(), candidate)
                self.assert_rejected(candidate)

    def test_failed_canary_and_nonfinite_rates_block_qualification(self):
        candidate = result(60.0)
        candidate['rounds'][0]['streams'][0]['checks_passed'] = False
        self.compare(result(), candidate)
        self.assert_rejected(candidate)
        for rate in (0.0, float('nan'), float('inf')):
            with self.subTest(rate=rate):
                candidate = result(rate)
                self.compare(result(), candidate)
                self.assert_rejected(candidate)

    def test_main_receipt_does_not_expose_connection_exception_text(self):
        endpoint = 'https://private-server.example/secret-endpoint'
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'result.json'
            stdout = io.StringIO()
            with patch.object(sys, 'argv', [str(SCRIPT), '--output', str(output)]), \
                 patch.object(benchmark.urllib.request, 'urlopen', side_effect=urllib.error.URLError(endpoint)), \
                 contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
                benchmark.main()
            self.assertEqual(raised.exception.code, 1)
            receipt = output.read_text()
            self.assertEqual(json.loads(receipt)['errors'], ['URLError'])
            self.assertNotIn(endpoint, receipt)
            self.assertNotIn(endpoint, stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
