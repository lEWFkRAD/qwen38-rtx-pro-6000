#!/usr/bin/env python3
"""Loopback-only, read-only observer of Strata Lens schema-3 run manifests.

No engine commands, service inspection, log streaming, uploads or mutations.
The browser sees a bounded whitelist, not the manifest or local file paths.
Reported receipts are not independently revalidated by this observer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import itertools
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import time

import phase0_harness as harness

ROOT = Path(__file__).resolve().parent
VERSION = "0.1.0-preview.1"
MAX_MANIFEST_BYTES = 256 * 1024
MAX_ENTRIES = 200
ARTIFACTS = (harness.FINAL_R_NAME, harness.PREFILL_ROUTES_NAME, harness.VERIFIER_ROUTING_NAME)
PROGRESS = {"created", "launching", "gated", "running", "cleaned", "validating", "validated_pending_finalize"}
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/styles.css": ("styles.css", "text/css; charset=utf-8")}


def bounded_int(value, low=0, high=1024 ** 3):
    return value if type(value) is int and low <= value <= high else None


def timestamp(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 4102444800:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def obj(value):
    return value if isinstance(value, dict) else {}


def boolean(value):
    return value if type(value) is bool else None


def reference_plan():
    ids = harness.reference_prompt_tokens("numbers_01")
    sizes = harness.expected_artifact_bytes(len(ids), False)
    return {"prompt_id": "numbers_01", "real_tokens": len(ids), "launch_tokens": len(ids) + 1,
            "residual_bytes": sizes[harness.FINAL_R_NAME], "routes_bytes": sizes[harness.PREFILL_ROUTES_NAME],
            "layers": harness.N_LAYERS, "experts_per_token": harness.TOP_K}


class Observer:
    def __init__(self, runs_dir: Path, demo=False):
        self.runs_dir = runs_dir
        self.demo = demo
        self.secret = secrets.token_bytes(32)
        self.plan = reference_plan()

    def opaque_id(self, name):
        # Ignore the staging suffix so atomic finalization preserves selection.
        return "run-" + hmac.new(self.secret, os.fsencode(name.removesuffix(".partial")), hashlib.sha256).hexdigest()[:12]

    def unreadable(self, name):
        return {"id": self.opaque_id(name), "status": "unreadable", "prompt_id": "Local capture",
                "real_tokens": None, "started_at": None, "updated_at": None, "duration_s": None,
                "artifacts": [], "checks": {}, "test_fixture": False,
                "message": "Manifest unavailable, changing, oversized, or unsupported. No result inferred."}

    def read_run(self, root_fd, name):
        directory_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            fd = os.open("manifest.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            with os.fdopen(fd, "rb") as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_MANIFEST_BYTES:
                    raise ValueError("Unsupported manifest")
                raw = handle.read(MAX_MANIFEST_BYTES + 1)
                if len(raw) > MAX_MANIFEST_BYTES:
                    raise ValueError("Oversized manifest")
            data = json.loads(raw, parse_constant=lambda _: None)
            if not isinstance(data, dict) or type(data.get("schema")) is not int or data["schema"] != 3:
                raise ValueError("Unsupported schema")
            status = data.get("status")
            if not isinstance(status, str) or status not in PROGRESS | {"complete", "failed"}:
                raise ValueError("Unsupported status")
            fixture = obj(data.get("engine")).get("test_override") is True or "test_command_prefix" in data
            if status == "complete" and data.get("success") is True and not name.endswith(".partial"):
                state = "recorded_success"
                message = "Manifest reports completion. Run the CLI validator separately; GPU fidelity is not established here."
            elif status == "failed":
                state, message = "failed", "Manifest reports failure. Inspect private diagnostics locally."
            elif status in PROGRESS:
                state, message = "in_progress", "Manifest reports an unfinished stage; this does not prove an engine is currently running."
            else:
                state, message = "unreadable", "Incomplete or inconsistent completion record. No successful result inferred."
            if fixture:
                message = "TEST FIXTURE, not a GPU capture. " + message
            receipts = obj(data.get("artifact_receipts"))
            artifacts = []
            for artifact in ARTIFACTS:
                receipt = obj(receipts.get(artifact))
                recorded = bounded_int(receipt.get("bytes"))
                digest = receipt.get("sha256")
                digest = digest if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) else None
                try:
                    item = os.stat(artifact, dir_fd=directory_fd, follow_symlinks=False)
                    actual = item.st_size if stat.S_ISREG(item.st_mode) else None
                except OSError:
                    actual = None
                if receipt or actual is not None:
                    artifacts.append({"name": artifact, "bytes": recorded, "observed_bytes": actual,
                                      "sha256": digest, "size_matches": actual == recorded if recorded is not None else None})
            start, finish = data.get("started_epoch"), data.get("finished_epoch")
            started_at, finished_at = timestamp(start), timestamp(finish)
            duration = round(finish - start, 3) if started_at and finished_at and finish >= start else None
            group, engine = obj(data.get("process_group")), obj(data.get("engine"))
            required = set(ARTIFACTS[:2])
            if obj(data.get("requested_artifacts")).get("verifier_routing") is True:
                required.add(harness.VERIFIER_ROUTING_NAME)
            matched = {a["name"] for a in artifacts if a["size_matches"] is True}
            return {"id": self.opaque_id(name), "status": state, "prompt_id": "Local capture",
                    "real_tokens": bounded_int(obj(data.get("prompt")).get("real_token_count"), 1, harness.MAX_REAL_TOKENS),
                    "started_at": started_at, "updated_at": timestamp(info.st_mtime), "duration_s": duration,
                    "artifacts": artifacts, "test_fixture": fixture, "message": message,
                    "checks": {"success_recorded": boolean(data.get("success")),
                               "input_attestation_recorded": bool(obj(engine.get("input_contract"))),
                               "cleanup_recorded": boolean(group.get("gone_before_finalize")),
                               "durability_recorded": boolean(obj(data.get("durability")).get("completed")),
                               "artifact_sizes_match": required <= matched if artifacts else None}}
        finally:
            os.close(directory_fd)

    def snapshot(self):
        result = {"version": VERSION, "mode": "demo" if self.demo else "live",
                  "refreshed_at": timestamp(time.time()), "runs": [], "plan": self.plan,
                  "limits": {"gpu_capture_qualified": False, "independent_validation": False,
                             "read_only": True, "max_entries": MAX_ENTRIES},
                  "warnings": [], "truncated": False}
        if self.demo:
            result["warnings"] = ["SYNTHETIC DEMO. No run directory is read; these are illustrative UI records, not GPU evidence."]
            result["runs"] = [{"id": "demo-example", "status": "demo", "prompt_id": "Synthetic numbers_01",
                               "real_tokens": self.plan["real_tokens"], "started_at": None, "updated_at": None,
                               "duration_s": None, "test_fixture": True, "checks": {},
                               "artifacts": [{"name": harness.FINAL_R_NAME, "bytes": self.plan["residual_bytes"], "sha256": None},
                                             {"name": harness.PREFILL_ROUTES_NAME, "bytes": self.plan["routes_bytes"], "sha256": None}],
                               "message": "Synthetic layout preview only. Sizes come from the reference plan, not captured artifacts."}]
            return result
        try:
            root_fd = os.open(self.runs_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except FileNotFoundError:
            result["warnings"].append("Run directory does not exist yet. Waiting for operator-produced runs; nothing is created by the UI.")
            return result
        except OSError:
            result["warnings"].append("Run directory unavailable. Check the local path and permissions; symlink roots are refused.")
            return result
        try:
            with os.scandir(root_fd) as iterator:
                entries = list(itertools.islice(iterator, MAX_ENTRIES + 1))
            result["truncated"] = len(entries) > MAX_ENTRIES
            if result["truncated"]:
                result["warnings"].append("Directory scan capped at 200 entries. Use a dedicated run directory to see a complete set.")
            for entry in entries[:MAX_ENTRIES]:
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    result["runs"].append(self.read_run(root_fd, entry.name))
                except (OSError, ValueError, UnicodeError, RecursionError, OverflowError):
                    result["runs"].append(self.unreadable(entry.name))
        except OSError:
            result["warnings"].append("Run directory changed or became unavailable during this refresh.")
        finally:
            os.close(root_fd)
        result["runs"].sort(key=lambda row: (row["started_at"] or "", row["id"]), reverse=True)
        return result


class LensServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True

    def __init__(self, port, observer):
        self.observer = observer
        self.assets = {url: ((ROOT / "ui" / name).read_bytes(), mime) for url, (name, mime) in STATIC.items()}
        super().__init__(("127.0.0.1", port), Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "StrataLens"

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *_args):
        pass  # Never echo arbitrary request paths or headers to terminal logs.

    def permitted(self):
        port = self.server.server_port
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0] not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            return False
        origins = self.headers.get_all("Origin", [])
        if origins and (len(origins) != 1 or origins[0] != "http://" + hosts[0]):
            return False
        return self.path != "/api/state" or self.headers.get("Sec-Fetch-Site") != "cross-site"

    def send(self, code, payload, mime="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def do_GET(self):
        if not self.permitted():
            return self.send(403, b"Forbidden\n")
        if self.path in self.server.assets:
            payload, mime = self.server.assets[self.path]
            return self.send(200, payload, mime)
        if self.path == "/api/state":
            payload = json.dumps(self.server.observer.snapshot(), allow_nan=False).encode()
            return self.send(200, payload, "application/json; charset=utf-8")
        return self.send(404, b"Not found\n")

    do_HEAD = do_GET

    def reject_write(self):
        self.close_connection = True
        self.send(405, b"Read-only observer\n")

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = reject_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs", help="Operator-produced run directories to observe (no creation).")
    parser.add_argument("--port", type=int, default=8765, help="Loopback HTTP port (default 8765).")
    parser.add_argument("--demo", action="store_true", help="Explicit synthetic layout demo; does not read runs.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        server = LensServer(args.port, Observer(args.runs_dir.absolute(), args.demo))
    except OSError:
        parser.exit(1, "Cannot start observer. Check that the port is free and the UI assets exist.\n")
    print(f"Strata Lens {VERSION} — {'SYNTHETIC DEMO' if args.demo else 'read-only local feed'}\nhttp://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
