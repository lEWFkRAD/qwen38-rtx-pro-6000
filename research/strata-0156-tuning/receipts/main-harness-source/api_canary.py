#!/usr/bin/env python3
"""Private loopback qualification of a stock Strata engine through the retained API wrapper.

Default: inspect and print a plan. No service, production contract, readiness
receipt, HOLD or fleet route is changed. Execution is externally guarded by the
exact production maintenance owner and by process-bound capacity validation.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import datetime as dt
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import queue
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
RUNTIME = Path("/opt/onyx-strata-0142/runtime")
UNIT = Path("/etc/systemd/system/qwen38-flash-next-strata.service")
PYTHON = Path("/home/USER/model-tests/strata-20261005/Strata/.venv/bin/python")
_spec = importlib.util.spec_from_file_location("integrated_engine_guard", ROOT / "benchmark_strata.py")
CLI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CLI)
Refusal = CLI.Refusal


def private_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def private_config(production, candidate, settings, directory, port, token):
    cfg = copy.deepcopy(production)
    cfg.update(exe=str(candidate), host="127.0.0.1", port=port, api_key=token,
               compatibility_api_keys=[], log=str(directory / "engine-and-vision.log"), open_browser=False)
    cfg["lib_dirs"] = list(settings["lib_dirs"])
    cfg["vision"]["exe"] = settings["vision"]
    cfg.setdefault("env", {}).pop("STRATA_LSA_MAPS", None)
    for key, value in settings.get("profile", {}).get("env", {}).items():
        cfg["env"][key] = str(value)
    for flag, value in settings.get("profile", {}).get("arg_overrides", {}).items():
        cfg["args"] = CLI.replace_arg(cfg["args"], flag, value)
    return cfg


def wrapper_env(contract):
    # Reproduce the unit's explicit runtime environment. The retained wrapper
    # itself adds cfg.env and cfg.lib_dirs to its decoder/encoder children.
    # Do not inherit a developer shell's CUDA, linker or thread-count experiment.
    prefixes = ("STRATA_", "CUDA_", "NVIDIA_", "LD_", "OMP_", "MKL_", "OPENBLAS_", "PYTHON", "ONYX_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    env.update({str(k): str(v) for k, v in contract["environment"].items()})
    env["ONYX_STRATA_READINESS_GATE"] = "0"
    return env


def runtime_manifest(cfg, contract, candidate, settings):
    expected = [str(PYTHON), "-m", "serve.server", "--engine", "strata", "--config", str(CLI.CONFIG),
                "--host", "0.0.0.0", "--port", "8002"]
    if contract.get("exec_start") != expected or cfg.get("cwd") != str(RUNTIME):
        raise Refusal("retained production interpreter/wrapper entry point changed")
    if cfg.get("fit_max_tokens") is not True or cfg.get("reasoning_loop_recovery") != "recover" or cfg.get("reasoning_close_retry") is not True:
        raise Refusal("expected retained wrapper recovery/fit-max settings changed")
    if not isinstance(cfg.get("vision"), dict) or not cfg["vision"].get("gpu"):
        raise Refusal("production matched GPU vision configuration is missing")
    if any(cfg.get(key) for key in ("mcp_servers", "mcpServers", "mcp_config", "before_load", "replicas")):
        raise Refusal("private API canary requires the current local-only configuration without external hooks")
    files = [UNIT, RUNTIME / "serve/server.py", RUNTIME / "serve/frontend.py", RUNTIME / "serve/onyx_readiness.py",
             Path(cfg["vision"]["exe"])]
    hashes = {}
    for path in files:
        actual = CLI.sha(path)
        if contract.get("hashes", {}).get(str(path)) != actual:
            raise Refusal("retained wrapper/unit/vision hash differs from the production contract: " + str(path))
        hashes[str(path)] = actual
    # The independent candidate is deliberately outside the production contract.
    hashes[str(candidate)] = CLI.sha(candidate) if candidate.is_file() else None
    for path in [settings["vision"], *settings["library_files"]]:
        hashes[str(path)] = CLI.sha(path)
    model_files = {}
    for filename, expected_meta in contract.get("model_files", {}).items():
        info = Path(filename).stat()
        if info.st_size != expected_meta["size"] or ("mtime_ns" in expected_meta and info.st_mtime_ns != expected_meta["mtime_ns"]):
            raise Refusal("production-pinned model asset metadata changed: " + filename)
        model_files[filename] = {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
    return {"retained_python": str(PYTHON), "runtime": str(RUNTIME), "sha256": hashes,
            "vision": copy.deepcopy(cfg["vision"]), "unit_exec_start_matched": True,
            "production_pinned_model_metadata": model_files}


def start_ticks(pid):
    return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]


def process_group(pgid):
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getpgid(pid) != pgid:
                continue
            found.append({"pid": pid, "start_ticks": start_ticks(pid), "exe": str((entry / "exe").resolve()),
                          "argv": [os.fsdecode(s) for s in (entry / "cmdline").read_bytes().split(b"\0") if s]})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def process_identity(wrapper, expected_ticks, cfg, private_path):
    if wrapper.poll() is not None or start_ticks(wrapper.pid) != expected_ticks or os.getpgid(wrapper.pid) != wrapper.pid:
        raise Refusal("private wrapper process identity changed")
    members = process_group(wrapper.pid)
    owner = [p for p in members if p["pid"] == wrapper.pid]
    expected_wrapper = [str(PYTHON), "-m", "serve.server", "--engine", "strata", "--config", str(private_path),
                        "--host", "127.0.0.1", "--port", str(cfg["port"])]
    if len(owner) != 1 or owner[0]["argv"] != expected_wrapper or Path(owner[0]["exe"]).resolve() != PYTHON.resolve():
        raise Refusal("private wrapper executable/arguments do not match the retained entry point")
    engine = [p for p in members if Path(p["exe"]).resolve() == Path(cfg["exe"]).resolve()]
    expected_engine = [cfg["exe"], "--serve", *CLI.engine_args(cfg, "full")]
    if len(engine) != 1 or engine[0]["argv"] != expected_engine:
        raise Refusal("exactly one matching candidate engine must belong to the private process group")
    vision = [p for p in members if Path(p["exe"]).resolve() == Path(cfg["vision"]["exe"]).resolve()]
    vcfg = cfg["vision"]
    expected_vision = [vcfg["exe"], "--mmproj", vcfg["mmproj"], "--model", vcfg["model"], "--gpu"]
    for key, flag in (("threads", "--threads"), ("max_tokens", "--max-tokens"), ("min_tokens", "--min-tokens")):
        if vcfg.get(key):
            expected_vision += [flag, str(vcfg[key])]
    if len(vision) != 1 or vision[0]["argv"] != expected_vision:
        raise Refusal("exactly one matched official vision encoder must belong to the private process group")
    listener = CLI.command(["ss", "-H", "-ltnp", f"sport = :{cfg['port']}"])
    lines = listener.splitlines()
    if len(lines) != 1 or f"127.0.0.1:{cfg['port']}" not in lines[0] or f"pid={wrapper.pid}," not in lines[0]:
        raise Refusal("private API listener is not bound exclusively to the owned loopback wrapper")
    return {"wrapper": owner[0], "engine": engine[0], "vision": vision[0],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "process_group": wrapper.pid,
            "loopback_listener_verified": True}


def http_json(base, path, token, guard, payload=None, timeout=120, expected_status=200):
    if not re.fullmatch(r"http://127\.0\.0\.1:[0-9]+", base):
        raise Refusal("API canary is restricted to an explicit IPv4 loopback address")
    if not path.startswith("/") or path.startswith("//"):
        raise Refusal("invalid private API path")
    output = queue.Queue(maxsize=1)
    born = time.monotonic()

    def perform():
        try:
            data = None if payload is None else json.dumps(payload).encode()
            headers = {"Content-Type": "application/json"}
            if token is not None:
                headers["Authorization"] = "Bearer " + token
            request = urllib.request.Request(base + path, data=data, headers=headers)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                response = opener.open(request, timeout=timeout)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                raw = response.read((2 << 20) + 1)
                if len(raw) > 2 << 20:
                    raise Refusal("private API response exceeds the canary limit")
                output.put((response.code, json.loads(raw), time.monotonic()))
        except BaseException as exc:
            output.put(exc)

    threading.Thread(target=perform, daemon=True).start()
    while True:
        guard.check()
        remaining = min(born + timeout, guard.deadline) - time.monotonic()
        if remaining <= 0:
            raise Refusal("private API request deadline expired")
        try:
            result = output.get(timeout=min(1, remaining))
        except queue.Empty:
            continue
        if isinstance(result, BaseException):
            raise result
        status, value, finished = result
        if status != expected_status:
            # The body contains only this harness's synthetic requests, but
            # headers (including credentials) are never included in an error.
            raise Refusal(f"private API returned HTTP {status} for {path}; expected {expected_status}")
        return {"http_status": status, "wall_ms": (finished - born) * 1000, "response": value}


def validate_capacity(health, metrics, status):
    for key, value in {"status": "ok", "service": "strata", "loaded": True, "max_context": 131072,
                       "images": True, "api_key": True}.items():
        if health.get(key) != value:
            raise Refusal("private API health differs: " + key)
    engine = metrics.get("engine", {})
    for key, value in {"batch_slots": 8, "context": 131072, "kv": "int8", "expert_slots": 24576,
                       "kv_resident": 0, "spec": 8, "mtp_max": 6, "lookup": 3}.items():
        if engine.get(key) != value:
            raise Refusal("private API engine capacity differs: " + key)
    if status.get("context", {}).get("native") != 131072 or status.get("concurrency", {}).get("serving") != 8:
        raise Refusal("private API status does not retain 131072 context and eight slots")
    if status.get("vision", {}).get("available") is not True:
        raise Refusal("private API does not advertise matched image input")


def make_image():
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (768, 768), (0, 70, 220))
    ImageDraw.Draw(image).rectangle((128, 128, 640, 640), fill=(35, 235, 35))
    data = io.BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()


def request_cases(model, image):
    common = {"model": model, "temperature": 0, "stream": False, "experimental_speed_projection": True}
    plain = {**common, "messages": [{"role": "user", "content": "Reply with exactly STRATA_0156_API_OK"}],
             "max_tokens": 48, "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    tool = {**common, "messages": [{"role": "user", "content": "Call engine_probe exactly once with marker STRATA_0156_API_TOOL_OK and count 7."}],
            "max_tokens": 192, "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False},
            "tools": [{"type": "function", "function": {"name": "engine_probe", "description": "A synthetic canary call; it is never executed.",
                       "parameters": {"type": "object", "properties": {"marker": {"type": "string"}, "count": {"type": "integer"}},
                                      "required": ["marker", "count"], "additionalProperties": False}}}],
            "tool_choice": {"type": "function", "function": {"name": "engine_probe"}}}
    reasoning = {**common, "messages": [{"role": "user", "content": "Calculate 17 times 19 carefully. Put only the decimal integer in your final answer."}],
                 "max_tokens": 512, "reasoning_effort": "high"}
    vision = {**common, "messages": [{"role": "user", "content": [
                  {"type": "text", "text": "What color is the large square in the center? Answer with just the lowercase color name."},
                  {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode("ascii")}}]}],
              "max_tokens": 64, "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    return [("plain_exact_marker", plain), ("forced_tool_call", tool), ("high_reasoning_arithmetic", reasoning), ("fresh_image_color", vision)]


def validate_completion(name, value):
    choices = value.get("choices") or []
    if len(choices) != 1:
        raise Refusal("API canary did not return exactly one choice")
    choice, message = choices[0], choices[0].get("message", {})
    content = message.get("content") or ""
    if name == "forced_tool_call":
        calls = message.get("tool_calls") or []
        if len(calls) != 1 or calls[0].get("function", {}).get("name") != "engine_probe":
            raise Refusal("forced synthetic tool call name/count differs")
        args = json.loads(calls[0]["function"]["arguments"])
        if args != {"marker": "STRATA_0156_API_TOOL_OK", "count": 7} or type(args.get("count")) is not int:
            raise Refusal("forced synthetic tool call arguments differ")
        if choice.get("finish_reason") != "tool_calls":
            raise Refusal("forced tool call finish reason differs")
    else:
        expected = {"plain_exact_marker": "STRATA_0156_API_OK", "high_reasoning_arithmetic": "323", "fresh_image_color": "green"}[name]
        if content.strip() != expected or message.get("tool_calls") or choice.get("finish_reason") != "stop":
            raise Refusal("exact API answer/finish differs: " + name)
        if name == "high_reasoning_arithmetic" and not (message.get("reasoning_content") or "").strip():
            raise Refusal("high reasoning request did not expose a nonempty reasoning passage")


class PrivateWrapper:
    def __init__(self, cfg, contract, directory, candidate, settings, guard, port):
        self.cfg, self.contract, self.directory = cfg, contract, directory
        self.candidate, self.settings, self.guard, self.port = candidate, settings, guard, port
        self.proc = None
        self.log = None
        self.token = secrets.token_urlsafe(48)
        self.private = private_config(cfg, candidate, settings, directory, port, self.token)
        self.path = directory / "candidate-private.json"

    def __enter__(self):
        self.guard.check(force=True, before_start=True)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", self.port))
        private_json(self.path, self.private)
        env = wrapper_env(self.contract)
        self.log = (self.directory / "wrapper.log").open("x", encoding="utf-8")
        argv = [str(PYTHON), "-m", "serve.server", "--engine", "strata", "--config", str(self.path),
                "--host", "127.0.0.1", "--port", str(self.port)]
        try:
            self.proc = subprocess.Popen(argv, cwd=RUNTIME, env=env, stdin=subprocess.DEVNULL,
                                         stdout=self.log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            self.ticks = start_ticks(self.proc.pid)
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.proc is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=20)
            if process_group(self.proc.pid):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=5)
            # A killed watcher or child may briefly await reaping after its
            # parent exited. Bound that cleanup, without touching other groups.
            deadline = time.monotonic() + 5
            while process_group(self.proc.pid) and time.monotonic() < deadline:
                time.sleep(0.1)
        # Revoke even a credential whose wrapper failed before Popen completed.
        with contextlib.suppress(OSError):
            self.path.unlink()
        if self.log:
            self.log.close()

    def __exit__(self, *_):
        self.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    ap.add_argument("--owner", default="")
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--vision", type=Path, required=True)
    ap.add_argument("--lib-dir", type=Path, action="append", required=True)
    ap.add_argument("--profile", type=Path)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--port", type=int, default=18022)
    ap.add_argument("--ready-seconds", type=int, default=900)
    ap.add_argument("--case-seconds", type=int, default=600)
    ap.add_argument("--total-seconds", type=int, default=3600)
    cli = ap.parse_args()
    if not 1024 <= cli.port <= 65535 or cli.port == 8002:
        raise Refusal("private API port must differ from production")
    if not 60 <= cli.ready_seconds <= 1800 or not 30 <= cli.case_seconds <= 1800 or not 60 <= cli.total_seconds <= 7200:
        raise Refusal("API canary deadlines exceed bounded limits")
    cfg, contract, config_sha, contract_sha = CLI.current_config()
    CLI.engine_args(cfg, "full")
    profile = json.loads(cli.profile.read_text()) if cli.profile else {"id": "same-settings", "env": {}, "arg_overrides": {}}
    if any(key.startswith("STRATA_LSA") for key in profile.get("env", {})):
        raise Refusal("LSA settings are excluded from this stock engine qualification")
    settings = {"lib_dirs": [str(p.resolve()) for p in cli.lib_dir], "vision": str(cli.vision.resolve()), "profile": profile,
                "library_files": [str(p) for directory in cli.lib_dir for p in sorted(directory.glob("*.so*")) if p.is_file() and not p.is_symlink()]}
    manifest = runtime_manifest(cfg, contract, cli.candidate, settings)
    plan = {"scope": "isolated stock-engine qualification through the retained production API wrapper and matched official vision encoder",
            "production_config_sha256": config_sha, "production_contract_sha256": contract_sha,
            "manifest": manifest, "private_bind": f"127.0.0.1:{cli.port}", "gpu_work_started": False,
            "preserved_features": {k: cfg.get(k) for k in ("parallel", "fit_max_tokens", "reasoning_loop_recovery", "reasoning_close_retry", "sampling")},
            "private_configuration_changes": ["candidate decoder executable", "matched candidate vision/libraries and explicit tuning profile", "loopback host/private port",
                                               "fresh random authentication; production compatibility keys excluded", "private log destination", "browser disabled"],
            "admission": {"private_wrapper_ONYX_STRATA_READINESS_GATE": "0", "production_gate_unchanged": True,
                          "reason": "retained admission paths are deliberately fixed to production, whose owned HOLD remains present",
                          "replacement": ["loopback-only listener and undisclosed random API key", "exact owned production HOLD and inactive production throughout",
                                          "exact wrapper/decoder/encoder PID, start ticks, executable and arguments", "eight slots,131072 context,full resident INT8 and24576 experts before inference",
                                          "production config/contract hashes unchanged", "known synthetic requests only"]},
            "cases": ["authentication rejection", "authenticated health,status,metrics", "plain exact marker", "forced synthetic tool call with exact arguments",
                      "high reasoning arithmetic with a reasoning passage", "fresh synthetic PNG color"],
            "limitations": ["standalone private test; no production-readiness receipt or promotion claim", "finite API compatibility canaries; no statistical speed benchmark"]}
    if not cli.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,100}", cli.owner):
        raise Refusal("execution requires the exact owned production maintenance owner")
    if not cli.candidate.is_file():
        raise Refusal("candidate engine is missing")
    guard = CLI.Guard(cli.owner, config_sha, contract_sha, cli.total_seconds)
    guard.check(force=True, before_start=True)
    directory = cli.run_dir or ROOT / "receipts" / ("api-canary-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    if stat.S_IMODE(directory.stat().st_mode) != 0o700:
        directory.chmod(0o700)
    report = {**plan, "started_at": CLI.utc(), "owner": cli.owner, "status": "running", "cases": []}
    base = f"http://127.0.0.1:{cli.port}"
    image = make_image()
    (directory / "fresh-square.png").write_bytes(image)
    report["image"] = {"path": str(directory / "fresh-square.png"), "sha256": hashlib.sha256(image).hexdigest(),
                       "width": 768, "height": 768, "expected_center_square": "green", "background": "blue"}
    with (ROOT / ".integrated-engine.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with PrivateWrapper(cfg, contract, directory, cli.candidate, settings, guard, cli.port) as wrapper:
                report["gpu_work_started"] = True
                born = time.monotonic()
                while True:
                    if wrapper.proc.poll() is not None:
                        raise Refusal("private wrapper exited before authenticated readiness")
                    if time.monotonic() - born > cli.ready_seconds:
                        raise Refusal("private API startup deadline expired")
                    try:
                        health = http_json(base, "/health", wrapper.token, guard, timeout=5)["response"]
                        metrics = http_json(base, "/metrics", wrapper.token, guard, timeout=5)["response"]
                        status = http_json(base, "/v1/status", wrapper.token, guard, timeout=5)["response"]
                    except (urllib.error.URLError, ConnectionError, TimeoutError, socket.timeout):
                        guard.check()
                        time.sleep(0.5)
                        continue
                    validate_capacity(health, metrics, status)
                    identity = process_identity(wrapper.proc, wrapper.ticks, wrapper.private, wrapper.path)
                    log = (directory / "engine-and-vision.log").read_text(encoding="utf-8", errors="replace")
                    if "strata LSA: native learned-map selector enabled;" in log:
                        raise Refusal("LSA must remain absent in this stock engine qualification")
                    report.update(identity=identity, initial_health=health, initial_metrics=metrics, initial_status=status,
                                  startup_ms=(time.monotonic() - born) * 1000, native_lsa_disabled_verified=True)
                    break
                # Normal auth is tested before any inference. The random private
                # credential is never returned, printed or passed in argv.
                rejected = http_json(base, "/v1/status", None, guard, timeout=10, expected_status=401)
                report["cases"].append({"id": "unauthenticated_status_rejected", "status": "passed", **rejected})
                for name, payload in request_cases(cfg["model_name"], image):
                    guard.check(force=True)
                    if process_identity(wrapper.proc, wrapper.ticks, wrapper.private, wrapper.path) != identity:
                        raise Refusal("API wrapper/engine/vision identity changed between canaries")
                    result = http_json(base, "/v1/chat/completions", wrapper.token, guard, payload, timeout=cli.case_seconds)
                    record = {"id": name, "status": "failed", **result}
                    report["cases"].append(record)
                    validate_completion(name, result["response"])
                    record["status"] = "passed"
                    print(json.dumps({"event": "api_case", "case": name, "status": "passed", "wall_ms": result["wall_ms"]}), flush=True)
                final_health = http_json(base, "/health", wrapper.token, guard, timeout=15)["response"]
                final_metrics = http_json(base, "/metrics", wrapper.token, guard, timeout=15)["response"]
                final_status = http_json(base, "/v1/status", wrapper.token, guard, timeout=15)["response"]
                validate_capacity(final_health, final_metrics, final_status)
                if process_identity(wrapper.proc, wrapper.ticks, wrapper.private, wrapper.path) != identity:
                    raise Refusal("private API process identity changed during qualification")
                report.update(final_health=final_health, final_metrics=final_metrics, final_status=final_status,
                              status="passed")
            if process_group(wrapper.proc.pid):
                raise Refusal("owned private wrapper process group did not fully exit")
            guard.check(force=True)
            report["owned_process_group_empty"] = True
            report["private_credential_file_removed"] = not wrapper.path.exists()
        except BaseException as exc:
            report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            if isinstance(exc, KeyboardInterrupt):
                raise
        finally:
            report["finished_at"] = CLI.utc()
            private_json(directory / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(directory / "report.json")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"private API canary interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        raise SystemExit(main())
    except (Refusal, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
