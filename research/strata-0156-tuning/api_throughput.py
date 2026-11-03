#!/usr/bin/env python3
"""Guarded eight-client API throughput spot. Default: CPU-only plan.

The retained production wrapper and sampling defaults are preserved. SSE usage,
not chunk counts, supplies completion-token accounting. No service is changed.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import random
import re
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

import api_canary as API

B = API.CLI
ROOT = Path(__file__).resolve().parent
Refusal = B.Refusal
LIMIT = 2 << 20


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def workload(cfg, fixture_path, seed):
    tok, template, token_hashes = B.load_tokenizer(cfg)
    if fixture_path:
        raw = json.loads(fixture_path.read_text())
        cases, solo = raw.get("batch", raw.get("cases")), raw.get("solo")
        origin = {"path": str(fixture_path.resolve()), "sha256": B.sha(fixture_path),
                  "fresh_heldout": False, "scope": "same saved native inputs; no new heldout claim"}
    else:
        rng = random.Random(seed)
        cases = []
        for slot in range(9):
            name = f"api-heldout-{seed}-{slot}"
            filler = "\n".join(f"Fictional {name} record {i:05d}: parcel {rng.randrange(1000000):06d}; "
                               f"bay {rng.randrange(1000):03d}; revision {rng.randrange(1000):03d}."
                               for i in range(700))
            body = tok.decode(tok.encode(filler)[:3900])
            task = ("Write a complete Python module for this fictional ledger with parsing, validation, filtering, "
                    "grouping, statistics, JSON export, and at least twenty detailed tests. Include docstrings and "
                    "examples. Produce at least 150 lines; continue until the whole module is complete."
                    if slot % 2 else
                    "Write a detailed operational runbook for this fictional ledger with at least 150 numbered steps. "
                    "Explain validation, reconciliation, retention, error handling, recovery, monitoring, and worked "
                    "examples. Continue until the complete runbook is written.")
            cases.append(B.encode_case(tok, template, name,
                         f"Independent workload {name}. All records below are fictional.\n{body}\n{task}", None, 256))
        origin = {"seed": seed, "fresh_heldout": True, "scope": "new synthetic API-only inputs frozen across arms"}
        cases, solo = cases[:8], cases[8]
    if not isinstance(cases, list) or len(cases) != 8:
        raise Refusal("API throughput requires exactly eight saved or generated inputs")
    if not isinstance(solo, dict):
        raise Refusal("API responsiveness spot requires one saved or generated 4000-token solo input")
    clean = []
    for slot, case in enumerate([*cases, solo]):
        messages = case.get("messages")
        if not isinstance(messages, list) or not messages or any(
            not isinstance(m, dict) or m.get("role") not in ("system", "user") or not isinstance(m.get("content"), str)
            for m in messages
        ):
            raise Refusal("throughput fixtures must contain only plain synthetic system/user text")
        # Re-tokenize the frozen messages with the same no-thinking template used by the native screen.
        item = B.encode_case(tok, template, str(case.get("id", f"stream{slot}")), "", None, 256, messages)
        if not 3500 <= item["prompt_tokens"] <= 4500:
            raise Refusal("each API workload must retain a roughly 4000-token prompt")
        if fixture_path and (case.get("token_sha256") != item["token_sha256"] or case.get("prompt_tokens") != item["prompt_tokens"]):
            raise Refusal("saved native fixture does not match the retained tokenizer/template")
        clean.append(item)
    if len({c["token_sha256"] for c in clean}) != 9:
        raise Refusal("solo and eight throughput prompts must be distinct")
    return clean[:8], clean[8], origin, token_hashes


def arm_settings(cfg, exe, vision, lib_dirs, profile):
    exe, vision = exe.resolve(), vision.resolve()
    dirs = [p.resolve() for p in lib_dirs]
    if not exe.is_file() or not vision.is_file() or not dirs or any(not p.is_dir() for p in dirs):
        raise Refusal("an engine, matched encoder, or matched library directory is missing")
    if profile.get("executable") and Path(profile["executable"]).resolve() != exe:
        raise Refusal("selected profile executable differs from the API candidate")
    if profile.get("sha256") and profile["sha256"] != B.sha(exe):
        raise Refusal("API candidate differs from its pinned profile hash")
    if profile.get("lib_dirs") and [str(p) for p in dirs] != profile["lib_dirs"]:
        raise Refusal("selected profile library paths differ from the API candidate")
    B.apply_profile_args(B.engine_args(cfg, "full"), profile)
    # Reuse the benchmark's constrained stock-environment validation without starting a child.
    B.launch_env(cfg, {"environment": {}}, {**profile, "lib_dirs": [str(p) for p in dirs]})
    if any(k.startswith("STRATA_LSA") for k in profile.get("env", {})):
        raise Refusal("LSA is excluded from stock API throughput")
    if int(B.arg_value(B.apply_profile_args(B.engine_args(cfg, "full"), profile), "--spec")) != 6:
        raise Refusal("this retained API campaign requires its fixed MTP6 policy")
    return exe, {"vision": str(vision), "lib_dirs": [str(p) for p in dirs], "profile": profile,
                 "library_files": [str(p) for d in dirs for p in sorted(d.glob("*.so*")) if p.is_file() and not p.is_symlink()],
                 "library_manifest": B.library_manifest([str(p) for p in dirs])}


def verify_bundle(cfg, settings):
    return B.verify_library_manifest({**settings["profile"], "lib_dirs": settings["lib_dirs"],
                                     "library_manifest": settings["library_manifest"]}, cfg)


def loaded_bundle(identity, pinned):
    return {role: B.loaded_library_manifest(identity[role]["pid"], pinned) for role in ("engine", "vision")}


def read_sse(response, born):
    """Read the wrapper's real OpenAI SSE format, including its final usage chunk."""
    body, reasoning, data = [], [], []
    usage = finish = timings = None
    first = None
    byte_count = chunks = 0
    saw_done = False
    while True:
        line = response.readline(65537)
        byte_count += len(line)
        if len(line) > 65536 or byte_count > LIMIT:
            raise Refusal("API SSE response exceeded its bounded size")
        if not line:
            break
        line = line.rstrip(b"\r\n")
        if line.startswith(b"data:"):
            data.append(line[5:].lstrip(b" "))
        elif line:
            continue  # comments/event names are not content or token events
        elif data:
            payload = b"\n".join(data); data = []
            if payload == b"[DONE]":
                saw_done = True
                break
            value = json.loads(payload)
            if "error" in value:
                raise Refusal("retained API emitted an SSE error")
            chunks += 1
            choices = value.get("choices", [])
            if len(choices) > 1:
                raise Refusal("API emitted multiple choices for a single throughput stream")
            if choices:
                choice = choices[0]
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text:
                    if not isinstance(text, str):
                        raise Refusal("API content delta is not text")
                    if first is None:
                        first = time.monotonic()
                    body.append(text)
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if delta.get("tool_calls"):
                    raise Refusal("plain throughput workload unexpectedly requested a tool")
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
            if value.get("usage") is not None:
                if usage is not None:
                    raise Refusal("API emitted more than one usage record")
                usage = value["usage"]
            if value.get("timings") is not None:
                timings = value["timings"]
    if not saw_done or data or finish not in ("stop", "length") or not isinstance(usage, dict):
        raise Refusal("API stream lacks a normal finish, final usage, or DONE boundary")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if type(usage.get(key)) is not int or usage[key] < 0:
            raise Refusal("API usage token counts must be nonnegative integers")
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise Refusal("API usage total disagrees with prompt and completion counts")
    if not 0 < usage["completion_tokens"] <= 256 or first is None:
        raise Refusal("API throughput stream emitted no visible content or exceeded its token cap")
    done = time.monotonic()
    return {"actual_content": "".join(body), "reasoning_content": "".join(reasoning), "usage": usage,
            "finish_reason": finish, "first_nonempty_content_ttft_ms": (first - born) * 1000,
            "wall_ms": (done - born) * 1000, "sse_chunks_observed": chunks,
            "timings": timings, "completed_at": done}


def measure_requests(wrapper, cases, guard, request_seconds, identity, sampler):
    count = len(cases)
    if count not in (1, 8):
        raise Refusal("API spot supports one solo client or eight synchronized clients")
    base = f"http://127.0.0.1:{wrapper.port}"
    results, ready = queue.Queue(), queue.Queue()
    release, abort = threading.Event(), threading.Event()
    responses, response_lock = {}, threading.Lock()
    threads = []

    def worker(slot, case):
        try:
            payload = {"model": wrapper.cfg["model_name"], "messages": case["messages"], "max_tokens": 256,
                       "stream": True, "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
            # Sampling is intentionally absent: the wrapper applies the unchanged production sampling block.
            request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json", "Authorization": "Bearer " + wrapper.token})
            ready.put(slot)
            if not release.wait(15) or abort.is_set():
                return
            sent = time.monotonic()
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=request_seconds) as response:
                with response_lock:
                    responses[slot] = response
                if response.status != 200 or not response.headers.get("Content-Type", "").startswith("text/event-stream"):
                    raise Refusal("API did not accept the expected SSE request")
                result = read_sse(response, sent)
            results.put((slot, {"id": case["id"], "http_status": 200, "sent_at": sent, **result}))
        except BaseException as exc:
            # No arbitrary HTTP body/header or request is included in errors (the credential stays private).
            error = str(exc) if isinstance(exc, Refusal) else type(exc).__name__
            results.put((slot, {"error": error}))
        finally:
            with response_lock:
                responses.pop(slot, None)

    try:
        for slot, case in enumerate(cases):
            thread = threading.Thread(target=worker, args=(slot, case), daemon=True)
            thread.start(); threads.append(thread)
        started = time.monotonic()
        prepared = set()
        while len(prepared) < count:
            guard.check(); sampler.check()
            try:
                prepared.add(ready.get(timeout=.5))
            except queue.Empty:
                if time.monotonic() - started > 10:
                    raise Refusal("API clients could not prepare before release")
        guard.check(force=True)
        if API.process_identity(wrapper.proc, wrapper.ticks, wrapper.private, wrapper.path) != identity:
            raise Refusal("private API process identity changed before concurrent requests")
        born = time.monotonic(); release.set()
        completed, samples = {}, []
        peak_active = peak_decoding = 0
        next_sample = born
        while len(completed) < count:
            guard.check(); sampler.check()
            if time.monotonic() >= min(born + request_seconds, guard.deadline):
                raise Refusal("API workload deadline expired")
            try:
                slot, result = results.get(timeout=.1)
                if slot in completed or "error" in result:
                    raise Refusal("API throughput client failed: " + result.get("error", "duplicate stream"))
                completed[slot] = result
            except queue.Empty:
                pass
            if time.monotonic() >= next_sample and len(completed) < count:
                if API.process_identity(wrapper.proc, wrapper.ticks, wrapper.private, wrapper.path) != identity:
                    raise Refusal("private API process identity changed during throughput")
                metrics = API.http_json(base, "/metrics", wrapper.token, guard, timeout=5)["response"]
                slots = metrics.get("live", {}).get("slots", [])
                active = sum(s.get("state") != "idle" for s in slots)
                decoding = sum(s.get("state") == "decoding" for s in slots)
                peak_active, peak_decoding = max(peak_active, active), max(peak_decoding, decoding)
                samples.append({"elapsed_ms": (time.monotonic() - born) * 1000,
                                "active_slots": active, "decoding_slots": decoding,
                                "live": metrics.get("live", {})})
                next_sample = time.monotonic() + .5
        ordered = [completed[s] for s in range(count)]
        for case, row in zip(cases, ordered):
            if row["usage"]["prompt_tokens"] != case["prompt_tokens"]:
                raise Refusal("API prompt token count differs from the frozen native template")
        finish = max(r["completed_at"] for r in ordered)
        counts = [r["usage"]["completion_tokens"] for r in ordered]
        duration = finish - born
        cap_reached = all(n == 256 for n in counts)
        if count == 8 and peak_active != 8:
            raise Refusal("API metrics never observed all eight owned engine slots active")
        report = {"streams": ordered, "concurrent_clients": count, "engine_slots_allocated": 8, "requested_output_tokens_each": 256,
                  "actual_usage_completion_tokens_each": counts, "usage_completion_tokens_total": sum(counts),
                  "whole_request_seconds": duration,
                  "whole_request_output_tokens_per_second": sum(counts) / duration,
                  "all_output_caps_reached": cap_reached, "eligible_for_same_budget_speed_comparison": cap_reached,
                  "client_release_spread_ms": (max(r["sent_at"] for r in ordered) - min(r["sent_at"] for r in ordered)) * 1000,
                  "peak_observed_active_engine_slots": peak_active, "peak_observed_decoding_slots": peak_decoding,
                  "metrics_samples": samples,
                  "metric_scope": "real usage completion tokens / total time from synchronized client release through final DONE",
                  "ttft_scope": "first nonempty visible-content SSE delta; chunks are never treated as tokens",
                  "common_token_window_rate": None,
                  "common_token_window_note": "SSE chunks do not provide token arrival timestamps; native harness supplies that separate metric"}
        for row in ordered:
            row["sent_after_release_ms"] = (row.pop("sent_at") - born) * 1000
            row["completed_after_release_ms"] = (row.pop("completed_at") - born) * 1000
        return report
    finally:
        abort.set(); release.set()
        # Worker threads are daemonized; wrapper process-group cleanup closes sockets even on an error.
        for thread in threads:
            thread.join(timeout=.1)


def readiness(wrapper, guard, ready_seconds):
    base = f"http://127.0.0.1:{wrapper.port}"
    born = time.monotonic()
    while True:
        guard.check()
        if wrapper.proc.poll() is not None or time.monotonic() - born > ready_seconds:
            raise Refusal("retained API wrapper failed to reach authenticated readiness")
        try:
            health = API.http_json(base, "/health", wrapper.token, guard, timeout=5)["response"]
            metrics = API.http_json(base, "/metrics", wrapper.token, guard, timeout=5)["response"]
            status = API.http_json(base, "/v1/status", wrapper.token, guard, timeout=5)["response"]
        except (urllib.error.URLError, ConnectionError, TimeoutError, socket.timeout):
            time.sleep(.2)
            continue
        API.validate_capacity(health, metrics, status)
        identity = API.process_identity(wrapper.proc, wrapper.ticks, wrapper.private, wrapper.path)
        API.http_json(base, "/v1/status", None, guard, timeout=5, expected_status=401)
        return {"health": health, "metrics": metrics, "status": status, "identity": identity,
                "startup_ms": (time.monotonic() - born) * 1000, "authentication_rejection_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(); mode.add_argument("--plan", action="store_true"); mode.add_argument("--execute", action="store_true")
    parser.add_argument("--arm", choices=("baseline", "candidate", "both"), default="both")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-vision", type=Path)
    parser.add_argument("--candidate-lib-dir", type=Path, action="append")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--baseline-exe", type=Path)
    parser.add_argument("--baseline-vision", type=Path)
    parser.add_argument("--baseline-lib-dir", type=Path, action="append")
    parser.add_argument("--fixtures", "--fixtures-json", dest="fixtures", type=Path)
    parser.add_argument("--seed", type=int, default=202611147)
    parser.add_argument("--owner", default="")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--port", type=int, default=18023)
    parser.add_argument("--ready-seconds", type=int, default=300)
    parser.add_argument("--request-seconds", type=int, default=300)
    parser.add_argument("--total-seconds", type=int, default=900)
    cli = parser.parse_args()
    if not 600 <= cli.total_seconds <= 1200 or not 30 <= cli.ready_seconds <= 600 or not 30 <= cli.request_seconds <= 600:
        raise Refusal("API spot deadlines exceed the bounded campaign")
    if not 1024 <= cli.port <= 65535 or cli.port == 8002:
        raise Refusal("API throughput requires a separate unprivileged loopback port")
    cfg, contract, config_sha, contract_sha = B.current_config()
    if cfg.get("sampling") != {"experimental_speed_projection": True}:
        raise Refusal("production sampling block changed; re-review its intended benchmark semantics")
    cases, solo, origin, tokenizer_hashes = workload(cfg, cli.fixtures, cli.seed)
    profile = json.loads(cli.profile.read_text()) if cli.profile else {"env": {}, "arg_overrides": {}}
    arms = []
    if cli.arm in ("baseline", "both"):
        exe, settings = arm_settings(cfg, cli.baseline_exe or Path(cfg["exe"]), cli.baseline_vision or Path(cfg["vision"]["exe"]),
                                     cli.baseline_lib_dir or list(map(Path, cfg["lib_dirs"])), {"env": {}, "arg_overrides": {}})
        arms.append(("baseline", exe, settings))
    if cli.arm in ("candidate", "both"):
        if not cli.candidate or not cli.candidate_vision or not cli.candidate_lib_dir:
            raise Refusal("candidate requires explicit matched engine, vision and library paths")
        exe, settings = arm_settings(cfg, cli.candidate, cli.candidate_vision, cli.candidate_lib_dir, profile)
        arms.append(("candidate", exe, settings))
    manifests = {role: API.runtime_manifest(cfg, contract, exe, settings) for role, exe, settings in arms}
    plan = {"scope": "finite eight-client retained-wrapper API throughput spot; no statistical benchmark or promotion claim",
            "sampling": copy.deepcopy(cfg["sampling"]), "request_sampling_overrides": {},
            "thinking": "disabled explicitly to match native no-thinking inputs", "streams": 8,
            "context": 131072, "engine_slots": 8, "max_completion_tokens_each": 256,
            "fixture_origin": origin, "tokenizer_sha256": tokenizer_hashes,
            "inputs": [{k: c[k] for k in ("id", "prompt_tokens", "token_sha256")} for c in cases],
            "solo_input": {k: solo[k] for k in ("id", "prompt_tokens", "token_sha256")},
            "request_order": "one4000-token solo256 response within8slotallocation, then8concurrent4000-token256 responses",
            "arms": [{"role": role, "engine": str(exe), "settings": settings, "manifest": manifests[role]} for role, exe, settings in arms],
            "private_bind": f"127.0.0.1:{cli.port}", "production_config_sha256": config_sha,
            "production_contract_sha256": contract_sha, "total_seconds": cli.total_seconds,
            "private_gate_override": {"ONYX_STRATA_READINESS_GATE": "0", "production_gate_unchanged": True,
                                      "replacement": "exact owned HOLD/inactive production, private auth, process-bound capacity/PID admission"},
            "gpu_work_started": False}
    if not cli.execute:
        print(json.dumps(plan, indent=2)); return 0
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,100}", cli.owner):
        raise Refusal("execution requires the exact owned maintenance owner")
    directory = cli.run_dir or ROOT / "receipts" / ("api-throughput-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    directory.mkdir(parents=True, mode=0o700, exist_ok=False); directory.chmod(0o700)
    API.private_json(directory / "fixtures.json", {"cases": cases, "solo": solo, "origin": origin})
    guard = B.Guard(cli.owner, config_sha, contract_sha, cli.total_seconds)
    report = {**plan, "started_at": B.utc(), "status": "running", "arms": []}
    with contextlib.ExitStack() as stack:
        for filename in (".benchmark-engine.lock", ".integrated-engine.lock"):
            lock = stack.enter_context((ROOT / filename).open("a+"))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for role, exe, settings in arms:
                guard.check(force=True, before_start=True)
                pinned = verify_bundle(cfg, settings)
                if API.runtime_manifest(cfg, contract, exe, settings) != manifests[role]:
                    raise Refusal("API engine, encoder or retained runtime changed after plan creation")
                arm_dir = directory / role; arm_dir.mkdir(mode=0o700)
                arm = {"role": role, "status": "running", "started_at": B.utc()}; report["arms"].append(arm)
                wrapper = None
                sampler = B.TelemetrySampler(arm_dir / "telemetry.jsonl"); sampler.start()
                try:
                    with API.PrivateWrapper(cfg, contract, arm_dir, exe, settings, guard, cli.port) as wrapper:
                        report["gpu_work_started"] = True
                        if wrapper.private.get("sampling") != cfg["sampling"]:
                            raise Refusal("private wrapper altered production sampling")
                        initial = readiness(wrapper, guard, cli.ready_seconds); arm["initial"] = initial
                        arm["initial_loaded_libraries"] = loaded_bundle(initial["identity"], pinned)
                        arm["solo"] = measure_requests(wrapper, [solo], guard, cli.request_seconds,
                                                       initial["identity"], sampler)
                        arm["solo_metrics"] = API.http_json(f"http://127.0.0.1:{cli.port}", "/metrics", wrapper.token, guard, timeout=5)["response"]
                        arm["throughput"] = measure_requests(wrapper, cases, guard, cli.request_seconds,
                                                             initial["identity"], sampler)
                        arm["final"] = readiness(wrapper, guard, 30)
                        if arm["final"]["identity"] != initial["identity"]:
                            raise Refusal("retained API process identity changed during the workload")
                        arm["final_loaded_libraries"] = loaded_bundle(initial["identity"], pinned)
                        verify_bundle(cfg, settings)
                        if API.runtime_manifest(cfg, contract, exe, settings) != manifests[role]:
                            raise Refusal("API engine, encoder or retained runtime changed during the workload")
                    if API.process_group(wrapper.proc.pid):
                        raise Refusal("owned API wrapper process group remains after cleanup")
                    arm.update(status="passed", owned_process_group_empty=True, private_credential_removed=not wrapper.path.exists())
                    guard.check(force=True, before_start=True)
                except BaseException as exc:
                    arm.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                               error=str(exc) if isinstance(exc, (Refusal, KeyboardInterrupt)) else type(exc).__name__)
                    raise
                finally:
                    arm["telemetry"] = sampler.stop(); arm["finished_at"] = B.utc()
                    if wrapper is not None:
                        arm["owned_process_group_empty"] = wrapper.proc is None or not API.process_group(wrapper.proc.pid)
                        arm["private_credential_removed"] = not wrapper.path.exists()
                    if arm["telemetry"]["errors"]:
                        arm.update(status="failed", error="API throughput telemetry failed")
                    API.private_json(arm_dir / "report.json", arm)
                if arm["telemetry"]["errors"]:
                    raise Refusal("API throughput telemetry failed")
            report["status"] = "passed"
            if len(report["arms"]) == 2:
                old, new = [a["throughput"] for a in report["arms"]]
                report["comparison"] = {"eligible": old["eligible_for_same_budget_speed_comparison"] and new["eligible_for_same_budget_speed_comparison"],
                    "candidate_over_baseline_whole_request_tps": new["whole_request_output_tokens_per_second"] / old["whole_request_output_tokens_per_second"],
                    "scope": "one sequential API spot per arm, same frozen inputs; statistical confidence not implied"}
                old_solo, new_solo = [a["solo"] for a in report["arms"]]
                report["comparison"]["solo"] = {
                    "eligible": old_solo["eligible_for_same_budget_speed_comparison"] and new_solo["eligible_for_same_budget_speed_comparison"],
                    "candidate_over_baseline_whole_request_tps": new_solo["whole_request_output_tokens_per_second"] / old_solo["whole_request_output_tokens_per_second"],
                    "baseline_first_content_ttft_ms": old_solo["streams"][0]["first_nonempty_content_ttft_ms"],
                    "candidate_first_content_ttft_ms": new_solo["streams"][0]["first_nonempty_content_ttft_ms"]}
        except BaseException as exc:
            report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          error=str(exc) if isinstance(exc, (Refusal, KeyboardInterrupt)) else type(exc).__name__)
        finally:
            report["finished_at"] = B.utc(); API.private_json(directory / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(directory / "report.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"API throughput interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGHUP, interrupted)
    try:
        raise SystemExit(main())
    except (Refusal, OSError, ValueError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc) if isinstance(exc, Refusal) else type(exc).__name__}), file=sys.stderr)
        raise SystemExit(2)
