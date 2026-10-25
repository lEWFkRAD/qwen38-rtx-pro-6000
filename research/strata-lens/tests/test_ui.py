"""CPU-only observer tests. All run receipts here are synthetic fixtures."""
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lens_ui as ui


def manifest():
    return {"schema": 3, "status": "complete", "success": True,
            "started_epoch": 1700000000, "finished_epoch": 1700000001.5,
            "prompt": {"real_token_count": 126, "real_token_ids": [42]},
            "engine": {"input_contract": {"contract_path": "/private/input.json"}},
            "process_group": {"gone_before_finalize": True}, "durability": {"completed": True},
            "artifact_receipts": {"prefill_final_r.bf16": {"bytes": 3, "sha256": "a" * 64}}}


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.observer = ui.Observer(self.runs)

    def run_record(self, data=None, name="private-client-id", raw=None):
        folder = self.runs / name
        folder.mkdir(exist_ok=True)
        (folder / "manifest.json").write_text(raw if raw is not None else json.dumps(data if data is not None else manifest()))
        return folder

    def row(self):
        return self.observer.snapshot()["runs"][0]

    def test_missing_root_is_empty_without_creating_it(self):
        missing = self.root / "absent"
        result = ui.Observer(missing).snapshot()
        self.assertFalse(missing.exists())
        self.assertEqual(result["runs"], [])
        self.assertTrue(result["warnings"])

    def test_reference_plan_matches_harness(self):
        plan = self.observer.snapshot()["plan"]
        self.assertEqual((plan["real_tokens"], plan["launch_tokens"]), (126, 127))
        self.assertEqual((plan["residual_bytes"], plan["routes_bytes"]), (2580480, 242688))

    def test_whitelist_omits_private_strings_and_tokens(self):
        data = manifest()
        secret = "NEVER_EXPORT_THIS_VALUE"
        data.update(run_id=secret, argv=[secret], controlled_env={"SECRET": secret},
                    error={"message": secret}, launch_directory=secret, gate_evidence={"unit": secret})
        data["prompt"]["id"] = secret
        data["artifact_receipts"][secret] = {"bytes": 9, "sha256": secret}
        self.run_record(data)
        result = self.observer.snapshot()
        text = json.dumps(result)
        for value in (secret, "private-client-id", "/private/input.json", "real_token_ids", "argv", "controlled_env"):
            self.assertNotIn(value, text)
        self.assertFalse(result["limits"]["independent_validation"])
        self.assertFalse(result["limits"]["gpu_capture_qualified"])

    def test_recorded_success_is_not_validation(self):
        folder = self.run_record()
        (folder / "prefill_final_r.bf16").write_bytes(b"abc")
        data = manifest()
        data["artifact_receipts"]["prefill_routes.bin"] = {"bytes": 3, "sha256": "b" * 64}
        (folder / "manifest.json").write_text(json.dumps(data))
        (folder / "prefill_routes.bin").write_bytes(b"xyz")
        row = self.row()
        self.assertEqual(row["status"], "recorded_success")
        self.assertIn("CLI validator", row["message"])
        self.assertEqual(row["duration_s"], 1.5)
        self.assertTrue(row["checks"]["cleanup_recorded"])
        self.assertTrue(row["checks"]["input_attestation_recorded"])
        self.assertTrue(row["checks"]["artifact_sizes_match"])

    def test_receipt_hash_is_not_recomputed(self):
        folder = self.run_record()
        (folder / "prefill_final_r.bf16").write_bytes(b"abc")
        self.assertEqual(self.row()["artifacts"][0]["sha256"], "a" * 64)

    def test_missing_mandatory_artifact_cannot_pass_aggregate_size_check(self):
        folder = self.run_record()
        (folder / "prefill_final_r.bf16").write_bytes(b"abc")
        self.assertTrue(self.row()["artifacts"][0]["size_matches"])
        self.assertFalse(self.row()["checks"]["artifact_sizes_match"])

    def test_non_utf8_directory_name_is_opaque_and_does_not_break_feed(self):
        name = os.fsdecode(b"private-\xff")
        self.run_record(name=name)
        row = self.row()
        self.assertTrue(row["id"].startswith("run-"))
        self.assertNotIn("private", json.dumps(row))

    def test_missing_or_changed_artifact_size_is_false(self):
        folder = self.run_record()
        self.assertFalse(self.row()["checks"]["artifact_sizes_match"])
        (folder / "prefill_final_r.bf16").write_bytes(b"abcd")
        self.assertFalse(self.row()["artifacts"][0]["size_matches"])

    def test_test_override_is_explicitly_labeled(self):
        data = manifest()
        data["engine"]["test_override"] = True
        self.run_record(data)
        self.assertTrue(self.row()["test_fixture"])
        self.assertIn("TEST FIXTURE", self.row()["message"])

    def test_partial_completion_is_not_success_and_id_survives_rename(self):
        folder = self.run_record(name="secret.partial")
        first = self.row()
        self.assertEqual(first["status"], "unreadable")
        folder.rename(self.runs / "secret")
        self.assertEqual(self.row()["id"], first["id"])
        self.assertEqual(self.row()["status"], "recorded_success")

    def test_running_is_only_a_reported_state(self):
        data = manifest()
        data.update(status="running", success=False)
        self.run_record(data)
        self.assertEqual(self.row()["status"], "in_progress")
        self.assertIn("does not prove", self.row()["message"])

    def test_failed_does_not_expose_error(self):
        data = manifest()
        data.update(status="failed", success=False, error={"message": "/private/secret"})
        self.run_record(data)
        self.assertEqual(self.row()["status"], "failed")
        self.assertNotIn("/private/secret", json.dumps(self.row()))

    def test_malformed_schema_non_object_nested_and_oversized_are_contained(self):
        for raw in ("{", "[]", '{"schema":2}', '{"schema":true}', "[" * 2000 + "]" * 2000,
                    " " * (ui.MAX_MANIFEST_BYTES + 1)):
            with self.subTest(raw=raw[:30]):
                self.run_record(raw=raw)
                self.assertEqual(self.row()["status"], "unreadable")

    def test_nonfinite_and_invalid_types_are_not_json_nan(self):
        data = manifest()
        data.update(started_epoch=float("nan"), finished_epoch=float("inf"))
        data["prompt"]["real_token_count"] = True
        data["artifact_receipts"]["prefill_final_r.bf16"] = {"bytes": -1, "sha256": "malformed"}
        self.run_record(data)
        row = self.row()
        self.assertIsNone(row["started_at"])
        self.assertIsNone(row["duration_s"])
        self.assertIsNone(row["real_tokens"])
        self.assertIsNone(row["artifacts"][0]["sha256"])
        json.dumps(self.observer.snapshot(), allow_nan=False)

    def test_directory_symlinks_are_skipped(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "manifest.json").write_text(json.dumps(manifest()))
        (self.runs / "link").symlink_to(outside, target_is_directory=True)
        self.assertEqual(self.observer.snapshot()["runs"], [])

    def test_manifest_symlink_is_unreadable_and_artifact_symlink_is_not_followed(self):
        folder = self.run_record()
        external = self.root / "external"
        external.write_text(json.dumps(manifest()))
        (folder / "prefill_final_r.bf16").symlink_to(external)
        self.assertIsNone(self.row()["artifacts"][0]["observed_bytes"])
        (folder / "manifest.json").unlink()
        (folder / "manifest.json").symlink_to(external)
        self.assertEqual(self.row()["status"], "unreadable")

    def test_root_symlink_is_refused(self):
        link = self.root / "linked-runs"
        link.symlink_to(self.runs, target_is_directory=True)
        result = ui.Observer(link).snapshot()
        self.assertEqual(result["runs"], [])
        self.assertTrue(result["warnings"])

    def test_fifo_manifest_does_not_block(self):
        folder = self.runs / "pipe"
        folder.mkdir()
        os.mkfifo(folder / "manifest.json")
        self.assertEqual(self.row()["status"], "unreadable")

    def test_scan_is_bounded_and_truncation_is_visible(self):
        for index in range(5):
            self.run_record(name=f"run{index}")
        with patch.object(ui, "MAX_ENTRIES", 3):
            result = self.observer.snapshot()
        self.assertEqual(len(result["runs"]), 3)
        self.assertTrue(result["truncated"])

    def test_demo_does_not_touch_run_directory(self):
        with patch.object(os, "open", side_effect=AssertionError("No filesystem reads")):
            result = ui.Observer(self.runs, demo=True).snapshot()
        self.assertEqual(result["mode"], "demo")
        self.assertEqual(result["runs"][0]["status"], "demo")
        self.assertIsNone(result["runs"][0]["artifacts"][0]["sha256"])
        self.assertIn("SYNTHETIC", result["warnings"][0])


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.server = ui.LensServer(0, ui.Observer(Path(cls.temp.name)))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.temp.cleanup()

    def request(self, method="GET", path="/api/state", headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_server_binds_loopback_and_returns_state(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        status, headers, body = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["mode"], "live")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])

    def test_same_origin_allowed_foreign_origin_and_host_refused(self):
        host = f"127.0.0.1:{self.server.server_port}"
        self.assertEqual(self.request(headers={"Origin": "http://" + host})[0], 200)
        for headers in ({"Host": "attacker.example"}, {"Origin": "https://attacker.example"},
                        {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"}):
            self.assertEqual(self.request(headers=headers)[0], 403)

    def test_only_fixed_routes_are_served(self):
        for path in ("/../lens_ui.py", "/%2e%2e/lens_ui.py", "/manifest.json", "/api/state?file=secret", "/runs/"):
            self.assertEqual(self.request(path=path)[0], 404)
        self.assertEqual(self.request(path="http://attacker.example/")[0], 403)

    def test_write_methods_are_refused(self):
        for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS"):
            self.assertEqual(self.request(method=method)[0], 405)

    def test_static_assets_and_head(self):
        for path in ("/", "/app.js", "/styles.css"):
            status, headers, body = self.request(path=path)
            self.assertEqual(status, 200)
            self.assertTrue(body)
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(self.request(method="HEAD")[2], b"")


if __name__ == "__main__":
    unittest.main()
