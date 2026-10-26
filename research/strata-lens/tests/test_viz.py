"""Visualization endpoint tests. All artifacts here are synthetic fixtures
built to the exact binary formats the harness writes; no GPU data is used."""
import http.client
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lens_ui as ui
import phase0_harness as harness


def bf16(value):
    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


WEIGHTS = [0.40, 0.20, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02]


def routes_bytes(tokens):
    out = bytearray()
    for layer in range(harness.N_LAYERS):
        out += struct.pack("<4i", layer, 0, tokens, 0)
        ids, weights = [], []
        for token in range(tokens):
            base = (layer * 13 + token * 37) % 500
            for rank in range(harness.TOP_K):
                ids.append(base + rank)
                weights.append(bf16(WEIGHTS[rank]))
        out += struct.pack(f"<{len(ids)}H", *ids)
        out += struct.pack(f"<{len(weights)}H", *weights)
    return bytes(out)


def residual_bytes(tokens):
    out = bytearray()
    for token in range(tokens):
        for index in range(harness.FINAL_R_WIDTH):
            out += struct.pack("<H", bf16(0.01 * ((index % 7) - 3) + (0.5 if index == token else 0.0)))
    return bytes(out)


def manifest(tokens=3, fixture=False):
    data = {"schema": 3, "status": "complete", "success": True,
            "started_epoch": 1700000000, "finished_epoch": 1700000001.5,
            "prompt": {"real_token_count": tokens},
            "engine": {}, "process_group": {"gone_before_finalize": True},
            "durability": {"completed": True}, "artifact_receipts": {}}
    if fixture:
        data["engine"]["test_override"] = True
    return data


class VizTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runs = Path(self.temp.name) / "runs"
        self.run_dir = self.runs / "synthetic-run"
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "prefill_routes.bin").write_bytes(routes_bytes(3))
        (self.run_dir / "prefill_final_r.bf16").write_bytes(residual_bytes(3))
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest()))
        self.observer = ui.Observer(self.runs)
        self.run_id = self.observer.snapshot()["runs"][0]["id"]

    def test_viz_parses_valid_capture(self):
        viz = self.observer.viz(self.run_id)
        self.assertTrue(viz["available"])
        self.assertEqual((viz["real_tokens"], viz["layer_count"], viz["top_k"]), (3, 48, 10))
        self.assertEqual(len(viz["layers"][0]["ids"]), 3)
        self.assertEqual(len(viz["layers"][0]["ids"][0]), 10)
        self.assertEqual(sum(viz["global_usage"]), 48 * 3 * 10)
        low, high = viz["validation"]["routes_weight_sum_range"]
        self.assertTrue(0.98 <= low <= high <= 1.02)
        self.assertEqual(len(viz["residual"]["per_token"]), 3)
        self.assertTrue(all(t["norm"] > 0 for t in viz["residual"]["per_token"]))
        self.assertEqual(len(viz["residual"]["per_token"][0]["bucket_rms"]), 64)
        self.assertEqual(len(viz["residual"]["per_token"][0]["stream_norms"]), 4)

    def test_viz_refuses_test_fixture(self):
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest(fixture=True)))
        self.assertFalse(ui.Observer(self.runs).viz(self.run_id)["available"])

    def test_viz_refuses_size_mismatch_and_corruption(self):
        (self.run_dir / "prefill_routes.bin").write_bytes(routes_bytes(3)[:-2])
        self.assertFalse(ui.Observer(self.runs).viz(self.run_id)["available"])
        (self.run_dir / "prefill_routes.bin").write_bytes(routes_bytes(3))
        raw = bytearray(routes_bytes(3))
        raw[0] = 9  # wrong layer header
        (self.run_dir / "prefill_routes.bin").write_bytes(bytes(raw))
        self.assertFalse(ui.Observer(self.runs).viz(self.run_id)["available"])

    def test_viz_unknown_id_is_unavailable_not_an_error(self):
        result = self.observer.viz("run-000000000000")
        self.assertFalse(result["available"])


class VizHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        runs = Path(cls.temp.name) / "runs"
        run_dir = runs / "synthetic-run"
        run_dir.mkdir(parents=True)
        (run_dir / "prefill_routes.bin").write_bytes(routes_bytes(3))
        (run_dir / "prefill_final_r.bf16").write_bytes(residual_bytes(3))
        (run_dir / "manifest.json").write_text(json.dumps(manifest()))
        cls.observer = ui.Observer(runs)
        cls.run_id = cls.observer.snapshot()["runs"][0]["id"]
        cls.server = ui.LensServer(0, cls.observer)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.temp.cleanup()

    def request(self, path):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def test_viz_endpoint_and_asset(self):
        status, body = self.request(f"/api/viz?id={self.run_id}")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["available"])
        status, body = self.request("/viz.js")
        self.assertEqual(status, 200)
        self.assertTrue(body)

    def test_viz_endpoint_rejects_extra_params(self):
        self.assertEqual(self.request(f"/api/viz?id={self.run_id}&file=secret")[0], 404)
        self.assertEqual(self.request("/api/viz")[0], 404)
        status, body = self.request("/api/viz?id=run-000000000000")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["available"])


if __name__ == "__main__":
    unittest.main()
