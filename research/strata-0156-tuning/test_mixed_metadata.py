#!/usr/bin/env python3
"""CPU regression tests for actual native prompt progress; no engine launch."""
import unittest
from unittest.mock import patch

import mixed_latency as M
from test_mixed_latency import Clock, FakeEngine, FakeTokenizer, successful_events


class MetadataShapeTests(unittest.TestCase):
    def test_nonmetadata_is_left_to_protocol_parser(self):
        self.assertIsNone(M.parse_admission_metadata('BT 0 42', 100))

    def test_native_resume_reused_and_progress_fields(self):
        self.assertEqual(M.parse_admission_metadata('RESUME 0', 100), {'kind': 'RESUME', 'reused_tokens': 0})
        self.assertEqual(M.parse_admission_metadata('REUSED 23', 100)['reused_tokens'], 23)
        row = M.parse_admission_metadata('PP 99 100 336 294.7', 100)
        self.assertEqual(row['position'], 99)
        self.assertEqual(row['total_tokens'], 100)
        self.assertEqual(row['elapsed_ms'], 336)
        self.assertEqual(row['fresh_tokens_per_second'], 294.7)

    def test_malformed_or_out_of_bounds_reuse_is_refused(self):
        for line in ('RESUME', 'RESUME 0 extra', 'RESUME -1', 'RESUME 1.0', 'RESUME nan', 'RESUME 101', 'REUSED -1'):
            with self.subTest(line=line), self.assertRaises(M.B.Refusal):
                M.parse_admission_metadata(line, 100)

    def test_bad_progress_shape_count_timing_or_rate_is_refused(self):
        for line in ('PP 10 100 20', 'PP -1 100 20 500', 'PP 101 100 20 500', 'PP 10 99 20 500',
                     'PP 10 100 nan 500', 'PP 10 100 inf 500', 'PP 10 100 -1 500',
                     'PP 10 100 20 nan', 'PP 10 100 20 inf', 'PP 10 100 20 -1'):
            with self.subTest(line=line), self.assertRaises(M.B.Refusal):
                M.parse_admission_metadata(line, 100)


def events_with_metadata():
    rows = []
    for at, line in successful_events():
        if line.startswith('BDONE ') and not line.startswith('BDONE 4 '):
            fields = line.split(); fields[2] = '5'; line = ' '.join(fields)
        rows.append((at, line))
    for slot in range(4):
        at = .1 * (slot + 1)
        rows.extend([(at - .05, 'RESUME 0'), (at - .03, 'PP 99 100 336 294.7'), (at - .01, 'REUSED 0')])
    rows.extend([(1.1, 'RESUME 0'), (1.25, 'PP 16000 32076 800 20000.0'),
                 (1.5, 'PP 30000 32076 1400 21428.6'), (1.8, 'PP 32075 32076 1800 17819.4'), (1.9, 'REUSED 0')])
    rows.extend((1.2 + slot * .01, f'BT {slot} 42') for slot in range(4))
    return sorted(rows, key=lambda row: row[0])


def run(rows):
    clock = Clock(); clock.now = 0
    engine = FakeEngine(rows, clock)
    backgrounds = [{'max_new': 1024, 'tokens': [1, 2], 'prompt_tokens': 100} for _ in range(4)]
    joiner = {'id': 'fresh32k', 'tokens': [5, 6], 'max_new': 64, 'prompt_tokens': 32076, 'expected': 'MARKER'}
    with patch.object(M.time, 'monotonic', side_effect=lambda: clock.now):
        return M.exercise(engine, backgrounds, joiner, FakeTokenizer(), 10, 2), engine


class MetadataInterleavingTests(unittest.TestCase):
    def test_normal_metadata_is_recorded_without_changing_tokens_lifecycle_or_timing(self):
        rows = events_with_metadata()
        result, engine = run(rows)
        stripped = [(at, line) for at, line in rows if line.split()[0] not in ('RESUME', 'PP', 'REUSED')]
        plain, plain_engine = run(stripped)
        self.assertEqual(result['joiner'], plain['joiner'])
        self.assertEqual(result['established_streams'], plain['established_streams'])
        self.assertEqual(engine.sent, plain_engine.sent)
        self.assertEqual(len(result['admission_metadata']), 17)
        self.assertEqual({x['slot'] for x in result['admission_metadata']}, {0, 1, 2, 3, 4})
        self.assertEqual(result['quality_status'], 'passed')
        self.assertTrue(result['all_owned_streams_cancelled_and_drained'])

    def test_metadata_without_pending_admission_is_refused(self):
        rows = successful_events() + [(.6, 'RESUME 0')]
        with self.assertRaises(M.B.Refusal):
            run(sorted(rows, key=lambda row: row[0]))

    def test_duplicate_resume_is_refused(self):
        rows = events_with_metadata() + [(.06, 'RESUME 0')]
        with self.assertRaises(M.B.Refusal):
            run(sorted(rows, key=lambda row: row[0]))

    def test_metadata_and_done_reuse_must_agree(self):
        rows = [(at, line.replace(' 0', ' 1') if at < .1 and line.startswith(('RESUME ', 'REUSED ')) else line)
                for at, line in events_with_metadata()]
        with self.assertRaises(M.B.Refusal):
            run(rows)

    def test_progress_cannot_move_backwards(self):
        rows = events_with_metadata() + [(.08, 'PP 98 100 340 288.2')]
        with self.assertRaises(M.B.Refusal):
            run(sorted(rows, key=lambda row: row[0]))


if __name__ == '__main__':
    unittest.main()
