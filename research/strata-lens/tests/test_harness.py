from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import phase0_harness as harness


FAKE_ENGINE = Path(__file__).resolve().with_name("fake_engine.py")
SIGNAL_DRIVER = Path(__file__).resolve().with_name("signal_driver.py")


class HarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="phase0-harness-test-")
        self.output_root = Path(self.temporary.name) / "runs"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def request(
        self,
        mode: str = "success",
        *,
        run_id: str = "test-run",
        include_verifier: bool = False,
        timeout: float = 3.0,
    ) -> harness.RunRequest:
        return harness.RunRequest(
            real_token_ids=(11, 12, 13),
            output_root=self.output_root,
            include_verifier_routing=include_verifier,
            timeout_seconds=timeout,
            kill_grace_seconds=0.1,
            command_prefix=(sys.executable, str(FAKE_ENGINE)),
            engine_identity={"test_override": True, "mode": mode},
            extra_child_env={
                "FAKE_ENGINE_MODE": mode,
                "FAKE_CHILD_PID_FILE": str(
                    self.output_root / f"{run_id}.partial" / "fake-child.pid"
                ),
            },
            test_only_allow_unpinned=True,
            run_id=run_id,
        )

    def manifest(self, run_dir: Path) -> dict:
        return json.loads((run_dir / harness.MANIFEST_NAME).read_text(encoding="utf-8"))

    def child_pid(self, run_id: str) -> int:
        path = self.output_root / f"{run_id}.partial" / "fake-child.pid"
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                return int(path.read_text(encoding="ascii").strip())
            except (FileNotFoundError, ValueError):
                time.sleep(0.01)
        self.fail(f"fake child pid file was not created: {path}")

    def wait_for_path(self, path: Path, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                return
            time.sleep(0.01)
        self.fail(f"timed out waiting for path: {path}")

    def assert_pid_not_running(self, pid: int) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                state = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split()[2]
            except (FileNotFoundError, IndexError):
                return
            if state == "Z":
                return
            time.sleep(0.02)
        self.fail(f"pid {pid} is still running")

    def assert_failed_without_false_success(self, run_id: str) -> dict:
        partial = self.output_root / f"{run_id}.partial"
        final = self.output_root / run_id
        self.assertTrue(partial.is_dir())
        self.assertFalse(final.exists())
        manifest = self.manifest(partial)
        self.assertIs(manifest["success"], False)
        self.assertEqual(manifest["status"], "failed")
        return manifest

    def test_plan_has_exact_safe_argv_and_environment(self) -> None:
        plan = harness.make_plan((1, 2, 3), False, identity={"test_override": True})
        argv = plan["argv"]
        placeholder = Path("<unique-run>.partial")
        self.assertEqual(
            argv,
            [str(harness.PINNED_ENGINE)]
            + harness.build_model_args(placeholder / harness.TOKENS_NAME),
        )
        self.assertNotIn("generate", argv)
        self.assertNotIn("--mmap-experts", argv)
        self.assertEqual(argv[argv.index("--expert-cache") + 1], "auto")
        self.assertEqual(argv[argv.index("--prefill") + 1], "1024")
        self.assertEqual(argv[argv.index("--spec") + 1], "4")
        self.assertEqual(argv[argv.index("--max-new") + 1], "1")
        self.assertEqual(plan["sentinel_token"], 248046)
        self.assertEqual(plan["launch_token_count"], 4)
        self.assertEqual(plan["containment"], harness.CONTAINMENT_CONTRACT)
        for option, path in {
            "--pack": harness.PACK,
            "--native": harness.NATIVE,
            "--ple-gguf": harness.PLE,
            "--expert-profile": harness.PROFILE,
            "--mtp": harness.MTP,
        }.items():
            self.assertEqual(argv[argv.index(option) + 1], str(path))
        cvec = argv[argv.index("--control-vector-scaled") + 1]
        self.assertEqual(cvec, harness.CONTROL_VECTOR)
        controlled = plan["controlled_env"]
        self.assertEqual(controlled["STRATA_ARENA_MMAP"], "1")
        self.assertEqual(controlled["STRATA_DBG_FEAT_LAYERS"], "")
        self.assertEqual(controlled["STRATA_PF_FUSED"], "0")
        self.assertEqual(controlled["STRATA_BATCH_SELF_COMMIT"], "1")
        self.assertEqual(controlled["STRATA_QSA_SLOTS"], "1")
        self.assertEqual(controlled["STRATA_SLOT_COPY_DIRECT"], "1")
        self.assertEqual(controlled["STRATA_SLOT_COPY_BATCHSYNC"], "1")
        self.assertEqual(controlled["STRATA_BATCH_DECODE_SHARE"], "0.75")
        self.assertEqual(controlled["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(controlled["LD_LIBRARY_PATH"], str(harness.PINNED_ENGINE.parent))
        self.assertNotIn("STRATA_LOGPOS", controlled)

    def test_child_environment_strips_unsafe_inherited_hooks(self) -> None:
        base = {
            "PATH": "/usr/bin",
            "STRATA_LOGPOS": "/tmp/not-allowed",
            "STRATA_DBG_ROUTE_OVERRIDE": "/tmp/not-allowed",
            "CUDA_VISIBLE_DEVICES": "7",
            "CUDA_LAUNCH_BLOCKING": "1",
            "LD_LIBRARY_PATH": "/tmp/injected",
            "LD_PRELOAD": "/tmp/injected.so",
            "LD_AUDIT": "/tmp/injected-audit.so",
        }
        env, controlled = harness.build_child_env(Path("/tmp/example.partial"), base_env=base)
        self.assertNotIn("STRATA_LOGPOS", env)
        self.assertNotIn("STRATA_DBG_ROUTE_OVERRIDE", env)
        self.assertEqual(env["STRATA_ARENA_MMAP"], "1")
        self.assertEqual(env["STRATA_DBG_FEAT_LAYERS"], "")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["LD_LIBRARY_PATH"], str(harness.PINNED_ENGINE.parent))
        self.assertNotIn("LD_PRELOAD", env)
        self.assertNotIn("CUDA_LAUNCH_BLOCKING", env)
        self.assertNotIn("LD_AUDIT", env)
        self.assertEqual(set(controlled), {
            "STRATA_ARENA_MMAP",
            "STRATA_PREFILL_DUMP_R_ALL",
            "STRATA_DBG_FEAT",
            "STRATA_DBG_FEAT_LAYERS",
            "STRATA_PF_FUSED",
            "STRATA_BATCH_SELF_COMMIT",
            "STRATA_QSA_SLOTS",
            "STRATA_SLOT_COPY_DIRECT",
            "STRATA_SLOT_COPY_BATCHSYNC",
            "STRATA_BATCH_DECODE_SHARE",
            "CUDA_VISIBLE_DEVICES",
            "LD_LIBRARY_PATH",
        })

    def test_extra_environment_cannot_override_engine_loader_or_gpu(self) -> None:
        for key in (
            "STRATA_PF_FUSED", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "LD_PRELOAD",
            "HIP_VISIBLE_DEVICES",
        ):
            with self.subTest(key=key), self.assertRaises(harness.HarnessError):
                harness.build_child_env(
                    Path("/tmp/example.partial"), base_env={}, extra_env={key: "injected"}
                )

    def test_reviewed_file_fingerprint_rejects_content_drift(self) -> None:
        path = Path(self.temporary.name) / "input.bin"
        path.write_bytes(b"independently-reviewed-fixture")
        stat = path.stat()
        entry = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.assertEqual(harness._verify_contract_file(entry)["method"], "sha256+stat")
        path.write_bytes(b"x" * stat.st_size)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(harness.IdentityError, "SHA-256 mismatch"):
            harness._verify_contract_file(entry)

    def test_template_cannot_attest_inputs_or_launch(self) -> None:
        harness.configure_profile(harness.EXAMPLE_PROFILE)
        with self.assertRaisesRegex(harness.IdentityError, "attested operator profile"):
            harness.assert_pinned_engine()
        self.assertFalse(harness.PROFILE_ATTESTED)

    def test_tokenizer_import_is_blocked_before_unreviewed_source_execution(self) -> None:
        harness.configure_profile(harness.EXAMPLE_PROFILE)
        with mock.patch.object(harness, "load_prompt_text", side_effect=AssertionError("unverified access")):
            with self.assertRaisesRegex(harness.IdentityError, "attested operator profile"):
                harness.tokenize_prompt()
        source = Path(self.temporary.name) / "strata_tokenizer.py"
        source.write_text("raise AssertionError('must never import')")
        stat = source.stat()
        entry = {"path": str(source), "size": stat.st_size,
                 "mtime_ns": stat.st_mtime_ns, "sha256": "0" * 64}
        second = {**entry, "path": str(source.with_name("gguf_reader.py"))}
        with mock.patch.object(harness, "_load_input_contract", return_value=({}, "a" * 64)), \
                mock.patch.object(harness, "_contract_entries", return_value=[entry, second]), \
                mock.patch.object(harness, "TOKENIZER_TOOLS", source.parent):
            with self.assertRaisesRegex(harness.IdentityError, "SHA-256 mismatch"):
                harness.tokenize_prompt()

    def test_operator_profile_needs_separately_reviewed_pin(self) -> None:
        profile = json.loads(harness.EXAMPLE_PROFILE.read_text())
        profile["state"] = "attested"
        profile["input_contract"]["sha256"] = "a" * 64
        path = Path(self.temporary.name) / "profile.json"
        path.write_text(json.dumps(profile))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            harness.configure_profile(path)
            self.assertFalse(harness.PROFILE_ATTESTED)
            harness.configure_profile(path, digest)
            self.assertTrue(harness.PROFILE_ATTESTED)
            path.write_text(json.dumps(profile) + "\n")
            with self.assertRaisesRegex(harness.IdentityError, "reviewed pin"):
                harness.configure_profile(path, digest)
        finally:
            harness.configure_profile(harness.EXAMPLE_PROFILE)

    def test_reference_plan_does_not_require_model_assets(self) -> None:
        tokens = harness.reference_prompt_tokens("numbers_01")
        self.assertEqual(len(tokens), 126)
        with mock.patch.object(harness, "assert_pinned_engine", side_effect=AssertionError("live access")):
            plan = harness.make_plan(tokens, False)
        self.assertTrue(plan["engine"]["plan_only"])
        self.assertFalse(plan["engine"]["input_attestation_verified"])
        self.assertEqual(plan["expected_bytes"][harness.FINAL_R_NAME], 2580480)

    def test_success_finalizes_atomically_and_validates_offline(self) -> None:
        completed = harness.run_once(self.request())
        self.assertEqual(completed, self.output_root / "test-run")
        self.assertTrue(completed.is_dir())
        self.assertFalse((self.output_root / "test-run.partial").exists())
        manifest = self.manifest(completed)
        self.assertIs(manifest["success"], True)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["launched_token_ids"], [11, 12, 13, 248046])
        self.assertEqual(
            (completed / harness.FINAL_R_NAME).stat().st_size,
            3 * harness.FINAL_R_BYTES_PER_TOKEN,
        )
        with self.assertRaises(harness.ValidationError):
            harness.validate_run_dir(completed)
        result = harness.validate_run_dir(completed, _allow_test_override=True)
        self.assertTrue(result["prefill_final_r"]["all_finite"])
        self.assertTrue(result["prefill_final_r"]["all_rows_nonzero"])
        self.assertEqual(result["prefill_routes"]["records"], 48)
        receipt = manifest["process_group"]
        self.assertEqual(
            manifest["child_pid"],
            manifest["child_pgid"],
        )
        self.assertEqual(manifest["child_pgid"], manifest["observed_child_pgid"])
        self.assertEqual(receipt["child_pid"], receipt["pgid"])
        self.assertEqual(receipt["pgid"], manifest["child_pgid"])
        self.assertTrue(receipt["gone_before_finalize"])
        self.assertEqual(receipt["containment_scope"], "original-pgid-only")
        self.assertEqual(
            [row["status"] for row in manifest["status_history"]],
            [
                "created", "launching", "gated", "running", "cleaned",
                "validating", "validated_pending_finalize", "complete",
            ],
        )

    def test_optional_verifier_routing_is_plumbing_only(self) -> None:
        completed = harness.run_once(self.request(include_verifier=True))
        result = harness.validate_run_dir(completed, _allow_test_override=True)
        verifier = result["verifier_routing"]
        self.assertEqual(verifier["bytes"], 4224)
        self.assertIn("non-research", verifier["classification"])

    def test_nonzero_exit_never_finalizes(self) -> None:
        with self.assertRaises(harness.ExecutionError):
            harness.run_once(self.request("nonzero", run_id="nonzero"))
        manifest = self.assert_failed_without_false_success("nonzero")
        self.assertEqual(manifest["engine_return_code"], 7)

    def test_timeout_kills_process_group_and_never_finalizes(self) -> None:
        with self.assertRaises(harness.ExecutionTimeout):
            harness.run_once(self.request("hang_child", run_id="timeout", timeout=0.2))
        child_pid = self.child_pid("timeout")
        self.assert_pid_not_running(child_pid)
        manifest = self.assert_failed_without_false_success("timeout")
        self.assertEqual(manifest["error"]["type"], "ExecutionTimeout")
        self.assertTrue(manifest["timed_out"])
        self.assertTrue(manifest["process_group"]["gone_before_finalize"])

    def test_nonzero_parent_with_lingering_child_is_fully_cleaned(self) -> None:
        with self.assertRaises(harness.ExecutionError):
            harness.run_once(self.request("nonzero_child", run_id="nonzero-child"))
        child_pid = self.child_pid("nonzero-child")
        self.assert_pid_not_running(child_pid)
        manifest = self.assert_failed_without_false_success("nonzero-child")
        self.assertEqual(manifest["engine_return_code"], 7)
        self.assertTrue(manifest["process_group"]["lingering_after_parent"])

    def test_simulated_interrupt_cleans_entire_group_and_never_finalizes(self) -> None:
        run_id = "interrupt"

        def interrupt_after_child_starts(process, timeout):
            del process, timeout
            self.child_pid(run_id)
            raise KeyboardInterrupt("injected unit-test interrupt")

        request = replace(
            self.request("hang_child", run_id=run_id, timeout=2.0),
            wait_for_process=interrupt_after_child_starts,
        )
        with self.assertRaises(KeyboardInterrupt):
            harness.run_once(request)
        child_pid = self.child_pid(run_id)
        self.assert_pid_not_running(child_pid)
        manifest = self.assert_failed_without_false_success(run_id)
        self.assertEqual(manifest["error"]["type"], "KeyboardInterrupt")
        self.assertTrue(manifest["process_group"]["gone_before_finalize"])

    def test_normal_parent_with_lingering_child_is_killed_and_run_fails(self) -> None:
        run_id = "linger-success"
        with self.assertRaisesRegex(harness.ExecutionError, "original process group"):
            harness.run_once(self.request("linger_success", run_id=run_id))
        child_pid = self.child_pid(run_id)
        self.assert_pid_not_running(child_pid)
        manifest = self.assert_failed_without_false_success(run_id)
        self.assertEqual(manifest["engine_return_code"], 0)
        self.assertTrue(manifest["process_group"]["lingering_after_parent"])

    def test_running_manifest_write_failure_cleans_original_process_group(self) -> None:
        run_id = "running-write-failure"
        original_write = harness.write_json_atomic
        injected = False

        def fail_running_write(path, payload):
            nonlocal injected
            if payload.get("status") == "running" and not injected:
                child_pid = self.child_pid(run_id)
                self.assertGreater(child_pid, 1)
                injected = True
                raise OSError("injected running-manifest write failure")
            return original_write(path, payload)

        with mock.patch.object(harness, "write_json_atomic", side_effect=fail_running_write):
            with self.assertRaisesRegex(OSError, "running-manifest"):
                harness.run_once(self.request("hang_child", run_id=run_id))
        self.assertTrue(injected)
        child_pid = self.child_pid(run_id)
        self.assert_pid_not_running(child_pid)
        manifest = self.assert_failed_without_false_success(run_id)
        self.assertTrue(manifest["process_group"]["gone_before_finalize"])
        self.assertEqual(manifest["child_pid"], manifest["process_group"]["pgid"])

    def test_getpgid_error_still_cleans_intended_group(self) -> None:
        run_id = "getpgid-failure"

        def fail_getpgid(pid):
            self.assertGreater(pid, 1)
            self.child_pid(run_id)
            raise OSError("injected getpgid failure")

        with mock.patch.object(harness.os, "getpgid", side_effect=fail_getpgid):
            with self.assertRaisesRegex(OSError, "getpgid"):
                harness.run_once(self.request("hang_child", run_id=run_id))
        child_pid = self.child_pid(run_id)
        self.assert_pid_not_running(child_pid)
        manifest = self.assert_failed_without_false_success(run_id)
        self.assertEqual(manifest["child_pid"], manifest["child_pgid"])
        self.assertEqual(manifest["child_pgid"], manifest["process_group"]["pgid"])
        self.assertTrue(manifest["process_group"]["gone_before_finalize"])

    def test_getpgid_mismatch_still_cleans_intended_group(self) -> None:
        run_id = "getpgid-mismatch"

        def mismatch_getpgid(pid):
            self.child_pid(run_id)
            return pid + 1

        with mock.patch.object(harness.os, "getpgid", side_effect=mismatch_getpgid):
            with self.assertRaisesRegex(harness.ExecutionError, "PGID mismatch"):
                harness.run_once(self.request("hang_child", run_id=run_id))
        child_pid = self.child_pid(run_id)
        self.assert_pid_not_running(child_pid)
        manifest = self.assert_failed_without_false_success(run_id)
        self.assertNotEqual(manifest["observed_child_pgid"], manifest["child_pgid"])
        self.assertEqual(manifest["child_pgid"], manifest["process_group"]["pgid"])
        self.assertTrue(manifest["process_group"]["gone_before_finalize"])

    def test_real_sigterm_and_repeat_signals_cannot_interrupt_cleanup(self) -> None:
        run_id = "real-signal"
        driver = subprocess.Popen(
            [sys.executable, str(SIGNAL_DRIVER), str(self.output_root), run_id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        partial = self.output_root / f"{run_id}.partial"
        child_path = partial / "fake-child.pid"
        cleanup_marker = self.output_root / "cleanup-entered"
        try:
            self.wait_for_path(child_path)
            os.kill(driver.pid, signal.SIGTERM)
            self.wait_for_path(cleanup_marker)
            # The first SIGTERM remains authoritative.  These repeats arrive
            # while the fake leader ignores TERM and cleanup is waiting/KILLing.
            for repeat in (signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
                os.kill(driver.pid, repeat)
            stdout, stderr = driver.communicate(timeout=10.0)
        finally:
            if driver.poll() is None:
                driver.kill()
                driver.wait(timeout=5.0)
        self.assertEqual(driver.returncode, 128 + signal.SIGTERM, (stdout, stderr))
        result = json.loads(
            (self.output_root / "signal-driver-result.json").read_text(encoding="utf-8")
        )
        self.assertEqual(result["signal"], signal.SIGTERM)
        self.assertTrue(result["handlers_restored"])
        child_pid = int(child_path.read_text(encoding="ascii").strip())
        self.assert_pid_not_running(child_pid)
        manifest = self.assert_failed_without_false_success(run_id)
        self.assertEqual(manifest["error"]["type"], "ExecutionSignal")
        self.assertEqual(manifest["termination_signal"]["number"], signal.SIGTERM)
        self.assertTrue(manifest["process_group"]["gone_before_finalize"])

    def test_artifact_and_manifest_files_are_fsynced_before_rename(self) -> None:
        calls = []
        original_fsync = harness._fsync_regular_file

        def tracked_fsync(path):
            calls.append(Path(path).name)
            return original_fsync(path)

        with mock.patch.object(harness, "_fsync_regular_file", side_effect=tracked_fsync):
            completed = harness.run_once(self.request(run_id="fsync-tracking"))
        expected = {
            harness.TOKENS_NAME,
            harness.STDOUT_NAME,
            harness.STDERR_NAME,
            harness.FINAL_R_NAME,
            harness.PREFILL_ROUTES_NAME,
            harness.MANIFEST_NAME,
        }
        self.assertTrue(expected.issubset(set(calls)))
        manifest = self.manifest(completed)
        self.assertTrue(manifest["durability"]["completed"])
        self.assertEqual(set(manifest["durability"]["files"]), expected)

    def test_artifact_fsync_failure_never_finalizes(self) -> None:
        original_fsync = harness._fsync_regular_file
        injected = False

        def fail_final_r(path):
            nonlocal injected
            if Path(path).name == harness.FINAL_R_NAME and not injected:
                injected = True
                raise OSError("injected artifact fsync failure")
            return original_fsync(path)

        with mock.patch.object(harness, "_fsync_regular_file", side_effect=fail_final_r):
            with self.assertRaisesRegex(OSError, "artifact fsync"):
                harness.run_once(self.request(run_id="fsync-failure"))
        self.assertTrue(injected)
        self.assert_failed_without_false_success("fsync-failure")

    def test_missing_artifact_never_finalizes(self) -> None:
        with self.assertRaises(harness.ValidationError):
            harness.run_once(self.request("missing", run_id="missing"))
        self.assert_failed_without_false_success("missing")

    def test_truncated_artifact_never_finalizes(self) -> None:
        with self.assertRaises(harness.ValidationError):
            harness.run_once(self.request("truncated", run_id="truncated"))
        manifest = self.assert_failed_without_false_success("truncated")
        self.assertIn("expected", manifest["error"]["message"])

    def test_nonfinite_artifact_never_finalizes(self) -> None:
        with self.assertRaises(harness.ValidationError):
            harness.run_once(self.request("nonfinite", run_id="nonfinite"))
        manifest = self.assert_failed_without_false_success("nonfinite")
        self.assertIn("non-finite", manifest["error"]["message"])

    def test_all_zero_final_r_never_finalizes(self) -> None:
        with self.assertRaises(harness.ValidationError):
            harness.run_once(self.request("allzero", run_id="allzero"))
        manifest = self.assert_failed_without_false_success("allzero")
        self.assertIn("all-zero", manifest["error"]["message"])

    def test_missing_prefill_routes_never_finalizes(self) -> None:
        with self.assertRaises(harness.ValidationError):
            harness.run_once(self.request("missing_routes", run_id="missing-routes"))
        self.assert_failed_without_false_success("missing-routes")

    def test_malformed_invalid_nonnormalized_and_duplicate_routes_are_rejected(self) -> None:
        cases = {
            "malformed_routes": "header",
            "invalid_route_id": "expert id",
            "nonnormalized_routes": "not normalized",
            "duplicate_route_id": "duplicate expert ids",
        }
        for mode, expected_message in cases.items():
            with self.subTest(mode=mode):
                with self.assertRaises(harness.ValidationError):
                    harness.run_once(self.request(mode, run_id=mode))
                manifest = self.assert_failed_without_false_success(mode)
                self.assertIn(expected_message, manifest["error"]["message"])

    def test_validate_rejects_failed_manifest_missing_tokens_and_tampered_receipt(self) -> None:
        completed = harness.run_once(self.request(run_id="validate-contract"))
        manifest_path = completed / harness.MANIFEST_NAME
        original_manifest = self.manifest(completed)

        failed = dict(original_manifest)
        failed["success"] = False
        failed["status"] = "failed"
        manifest_path.write_text(json.dumps(failed), encoding="utf-8")
        with self.assertRaises(harness.ValidationError):
            harness.validate_run_dir(completed, _allow_test_override=True)

        manifest_path.write_text(json.dumps(original_manifest), encoding="utf-8")
        tokens_path = completed / harness.TOKENS_NAME
        token_text = tokens_path.read_text(encoding="ascii")
        tokens_path.unlink()
        with self.assertRaises(harness.ValidationError):
            harness.validate_run_dir(completed, _allow_test_override=True)

        tokens_path.write_text(token_text, encoding="ascii")
        final_path = completed / harness.FINAL_R_NAME
        with final_path.open("r+b") as handle:
            handle.seek(0)
            handle.write(b"\x01\x00")
        with self.assertRaisesRegex(harness.ValidationError, "receipt mismatch"):
            harness.validate_run_dir(completed, _allow_test_override=True)

    def test_validate_rejects_manifest_contract_drift(self) -> None:
        fields = {
            "return_code": lambda m: m.__setitem__("engine_return_code", 1),
            "timed_out": lambda m: m.__setitem__("timed_out", True),
            "count": lambda m: m["prompt"].__setitem__("real_token_count", 99),
            "expected_bytes": lambda m: m.__setitem__("expected_bytes", {}),
            "argv": lambda m: m["argv"].append("generate"),
            "env": lambda m: m["controlled_env"].__setitem__("STRATA_PF_FUSED", "1"),
            "containment": lambda m: m["containment"].__setitem__("cgroup_containment", True),
            "gate": lambda m: m["gate_evidence"].__setitem__("passed", False),
            "cleanup_pgid": lambda m: m["process_group"].__setitem__("pgid", 999999),
            "durability": lambda m: m["durability"].__setitem__("completed", False),
            "status_history": lambda m: m["status_history"][4].__setitem__("epoch", 0),
            "signal": lambda m: m.__setitem__(
                "termination_signal", {"number": 15, "name": "SIGTERM", "received_epoch": 0}
            ),
        }
        for label, mutate in fields.items():
            with self.subTest(label=label):
                run_id = f"drift-{label}"
                completed = harness.run_once(self.request(run_id=run_id))
                manifest = self.manifest(completed)
                mutate(manifest)
                (completed / harness.MANIFEST_NAME).write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                with self.assertRaises(harness.ValidationError):
                    harness.validate_run_dir(completed, _allow_test_override=True)

    def test_token_id_validation_and_sentinel_are_explicit(self) -> None:
        self.assertEqual(harness.parse_token_ids("1, 2 3"), (1, 2, 3))
        with self.assertRaises(harness.HarnessError):
            harness.parse_token_ids("")
        with self.assertRaises(harness.HarnessError):
            harness.validate_token_ids((-1,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.txt"
            launched = harness.write_tokens(path, (7, 8))
            self.assertEqual(launched, (7, 8, 248046))
            self.assertEqual(path.read_text(encoding="ascii"), "7 8 248046\n")

    def test_run_request_bypass_seams_are_fail_closed(self) -> None:
        base = harness.RunRequest(real_token_ids=(1,), output_root=self.output_root)
        with self.assertRaisesRegex(harness.HarnessError, "explicitly supplied"):
            harness.run_once(base)
        with self.assertRaisesRegex(harness.HarnessError, "alternate command"):
            harness.run_once(
                replace(base, command_prefix=(sys.executable, str(FAKE_ENGINE)))
            )
        with self.assertRaisesRegex(harness.HarnessError, "identity overrides"):
            harness.run_once(replace(base, engine_identity={"test_override": True}))
        with self.assertRaisesRegex(harness.HarnessError, "extra child"):
            harness.run_once(replace(base, extra_child_env={"FAKE_ENGINE_MODE": "success"}))
        with self.assertRaisesRegex(harness.HarnessError, "test_override"):
            harness.run_once(
                replace(
                    base,
                    command_prefix=(sys.executable, str(FAKE_ENGINE)),
                    test_only_allow_unpinned=True,
                )
            )

    def test_numbers_prompt_exists_in_pinned_corpus(self) -> None:
        text = harness.load_prompt_text("numbers_01")
        self.assertTrue(text)

    def gate_runner(
        self,
        *,
        active: str = "inactive",
        sub: str = "dead",
        pid: str = "0",
        load: str = "loaded",
        systemctl_rc: int = 0,
        listener: str = "",
        ss_rc: int = 0,
        malformed_output: str | None = None,
        raise_on: str | None = None,
        systemctl_stderr: str = "",
        ss_stderr: str = "",
    ):
        def run(args, **kwargs):
            command = args[0]
            if command == raise_on:
                raise OSError(f"injected {command} failure")
            if command == "systemctl":
                output = malformed_output
                if output is None:
                    output = (
                        f"LoadState={load}\n"
                        f"ActiveState={active}\n"
                        f"SubState={sub}\n"
                        f"MainPID={pid}\n"
                    )
                return subprocess.CompletedProcess(args, systemctl_rc, output, systemctl_stderr)
            if command == "ss":
                return subprocess.CompletedProcess(args, ss_rc, listener, ss_stderr)
            raise AssertionError(f"unexpected command: {args!r}")

        return run

    def owned_hold(self, owner: str = "unit-test-owner") -> Path:
        path = Path(self.temporary.name) / "HOLD"
        path.write_text(owner + "\n", encoding="utf-8")
        return path

    def assert_gate_rejects(self, runner, *, owner: str = "unit-test-owner") -> None:
        with self.assertRaises(harness.HarnessError):
            harness._require_production_inactive(
                True,
                owner,
                command_runner=runner,
                hold_path=self.owned_hold(owner),
            )

    def test_offline_gate_allows_only_exact_quiescent_state(self) -> None:
        harness._require_production_inactive(
            True,
            "unit-test-owner",
            command_runner=self.gate_runner(),
            hold_path=self.owned_hold(),
        )

    def test_offline_gate_rejects_active_activating_deactivating_and_failed(self) -> None:
        cases = {
            "active": self.gate_runner(active="active", sub="running", pid="123"),
            # Regression for the original bug: is-active can return rc=3 while
            # the unit is still activating/start-post.  Any nonzero show result
            # and any activating state are both rejected.
            "activating_rc3": self.gate_runner(
                active="activating", sub="start-post", pid="456", systemctl_rc=3
            ),
            "activating_rc0": self.gate_runner(active="activating", sub="start-post", pid="456"),
            "deactivating": self.gate_runner(active="deactivating", sub="stop-sigterm", pid="789"),
            "failed": self.gate_runner(active="failed", sub="failed", pid="0"),
        }
        for label, runner in cases.items():
            with self.subTest(label=label):
                self.assert_gate_rejects(runner)

    def test_offline_gate_rejects_unknown_errors_and_malformed_state(self) -> None:
        cases = {
            "unknown_unit": self.gate_runner(load="not-found"),
            "systemctl_error": self.gate_runner(systemctl_rc=1),
            "systemctl_exec_error": self.gate_runner(raise_on="systemctl"),
            "systemctl_stderr": self.gate_runner(systemctl_stderr="warning"),
            "missing_property": self.gate_runner(
                malformed_output="LoadState=loaded\nActiveState=inactive\nMainPID=0\n"
            ),
            "invalid_pid": self.gate_runner(pid="unknown"),
            "ss_error": self.gate_runner(ss_rc=1),
            "ss_exec_error": self.gate_runner(raise_on="ss"),
            "ss_stderr": self.gate_runner(ss_stderr="warning"),
        }
        for label, runner in cases.items():
            with self.subTest(label=label):
                self.assert_gate_rejects(runner)

    def test_offline_gate_rejects_stale_pid_and_port_listener(self) -> None:
        self.assert_gate_rejects(self.gate_runner(pid="42"))
        self.assert_gate_rejects(
            self.gate_runner(listener="LISTEN 0 4096 127.0.0.1:8002 0.0.0.0:*\n")
        )

    def test_offline_gate_requires_matching_existing_hold_and_confirmation(self) -> None:
        runner = self.gate_runner()
        hold = self.owned_hold("actual-owner")
        with self.assertRaises(harness.HarnessError):
            harness._require_production_inactive(
                False,
                "actual-owner",
                command_runner=runner,
                hold_path=hold,
            )
        with self.assertRaises(harness.HarnessError):
            harness._require_production_inactive(
                True,
                None,
                command_runner=runner,
                hold_path=hold,
            )
        with self.assertRaises(harness.HarnessError):
            harness._require_production_inactive(
                True,
                "different-owner",
                command_runner=runner,
                hold_path=hold,
            )
        with self.assertRaises(harness.HarnessError):
            harness._require_production_inactive(
                True,
                "actual-owner",
                command_runner=runner,
                hold_path=Path(self.temporary.name) / "missing-HOLD",
            )

    def test_failed_prelaunch_recheck_never_starts_or_finalizes(self) -> None:
        calls = []

        def reject() -> None:
            calls.append("checked")
            raise harness.HarnessError("injected prelaunch ownership change")

        request = replace(
            self.request(run_id="prelaunch-rejected"),
            prelaunch_gate=reject,
        )
        with self.assertRaises(harness.HarnessError):
            harness.run_once(request)
        self.assertEqual(calls, ["checked"])
        manifest = self.assert_failed_without_false_success("prelaunch-rejected")
        self.assertNotIn("child_pid", manifest)


if __name__ == "__main__":
    unittest.main()
