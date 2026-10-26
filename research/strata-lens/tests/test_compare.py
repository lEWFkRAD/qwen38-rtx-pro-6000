"""Run-comparison tests. All artifacts are synthetic fixtures built to
the exact binary formats the harness writes; no GPU data is used."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_runs
import phase0_harness as harness


def bf16(value):
    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


WEIGHTS = [0.28, 0.18, 0.13, 0.10, 0.08, 0.07, 0.06, 0.04, 0.03, 0.03]


def routes_bytes(tokens, id_shift=0, weight_overrides=None):
    out = bytearray()
    for layer in range(harness.N_LAYERS):
        out += struct.pack("<4i", layer, 0, tokens, 0)
        ids, weights = [], []
        for token in range(tokens):
            base = (layer * 13 + token * 37 + id_shift) % 500
            ids.extend((base + k) % 512 for k in range(harness.TOP_K))
            row = list(WEIGHTS)
            if weight_overrides and (layer, token) in weight_overrides:
                row = weight_overrides[(layer, token)]
            weights.extend(bf16(w) for w in row)
        out += struct.pack(f"<{len(ids)}H", *ids)
        out += struct.pack(f"<{len(weights)}H", *weights)
    return bytes(out)


def residual_bytes(tokens, scale=1.0):
    per_row = harness.FINAL_R_BYTES_PER_TOKEN // 2
    out = bytearray()
    for token in range(tokens):
        row = []
        for i in range(per_row):
            value = ((i * 7 + token * 31) % 97 - 48) / 64.0 * scale
            row.append(bf16(value if value != 0.0 else 0.25))
        out += struct.pack(f"<{per_row}H", *row)
    return bytes(out)


def write_run(root, tokens=4, token_ids=None, residual_scale=1.0, **route_kwargs):
    run = Path(root)
    run.mkdir(parents=True, exist_ok=True)
    ids = token_ids if token_ids is not None else [100 + i for i in range(tokens)]
    (run / "tokens.txt").write_text(" ".join(str(t) for t in ids) + "\n", encoding="ascii")
    (run / harness.PREFILL_ROUTES_NAME).write_bytes(routes_bytes(tokens, **route_kwargs))
    (run / harness.FINAL_R_NAME).write_bytes(residual_bytes(tokens, scale=residual_scale))
    return run


class CompareRunsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def test_identical_runs(self):
        a = write_run(self.base / "a")
        b = write_run(self.base / "b")
        report = compare_runs.compare_runs(a, b)
        self.assertEqual(report["verdict"], "IDENTICAL")
        self.assertTrue(report["positional"])
        self.assertEqual(report["min_top1_agreement"], 1.0)
        self.assertAlmostEqual(report["residual"]["min_cosine"], 1.0, places=9)

    def test_top1_change_is_divergent_and_located(self):
        a = write_run(self.base / "a")
        b = write_run(self.base / "b", id_shift=1)
        report = compare_runs.compare_runs(a, b)
        self.assertEqual(report["verdict"], "DIVERGENT")
        self.assertLess(report["min_top1_agreement"], 1.0)
        self.assertTrue(report["worst_layers_by_top1"])

    def test_token_mismatch_is_incomparable(self):
        a = write_run(self.base / "a")
        b = write_run(self.base / "b", token_ids=[200, 201, 202, 203])
        report = compare_runs.compare_runs(a, b)
        self.assertEqual(report["verdict"], "INCOMPARABLE")
        self.assertFalse(report["positional"])

    def test_small_weight_shift_stays_within_tolerance(self):
        tweaked = list(WEIGHTS)
        tweaked[3] = WEIGHTS[3] + 0.001
        tweaked[4] = WEIGHTS[4] - 0.001
        overrides = {(layer, 0): tweaked for layer in range(harness.N_LAYERS)}
        a = write_run(self.base / "a")
        b = write_run(self.base / "b", weight_overrides=overrides)
        report = compare_runs.compare_runs(a, b)
        self.assertIn(report["verdict"], ("EQUIVALENT", "IDENTICAL"))
        self.assertEqual(report["min_top1_agreement"], 1.0)

    def test_cli_exit_codes(self):
        a = write_run(self.base / "a")
        b = write_run(self.base / "b")
        self.assertEqual(compare_runs.main([str(a), str(b)]), 0)
        c = write_run(self.base / "c", id_shift=3)
        self.assertEqual(compare_runs.main([str(a), str(c)]), 1)


if __name__ == "__main__":
    unittest.main()
