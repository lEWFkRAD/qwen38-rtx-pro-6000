#!/usr/bin/env python3
"""Plan-first mixed-load diagnostic: four generating streams plus a fresh 32K joiner.

All eight 131072-token slots remain allocated. This measures actual arrivals and
silent intervals, not a throughput benchmark or a production promotion gate.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import math
from pathlib import Path
import re
import signal
import subprocess
import time

import benchmark_strata as B

ROOT = Path(__file__).resolve().parent
ESTABLISHED = tuple(range(4))
JOINER = 4


def parse_admission_metadata(line, prompt_tokens):
    """0.1.50/0.1.56 prompt progress; never an output or lifecycle event.

    generate.cpp prints RESUME before reading, PP while reading, and REUSED
    before the first verify window. Counts refer to the pending main request.
    """
    fields = line.split()
    if not fields or fields[0] not in ("RESUME", "PP", "REUSED"):
        return None
    kind = fields[0]
    if not isinstance(prompt_tokens, int) or not 0 < prompt_tokens <= B.CAPACITY:
        raise B.Refusal("invalid pending prompt bound for native progress metadata")
    expected_arity = 5 if kind == "PP" else 2
    if len(fields) != expected_arity:
        raise B.Refusal("malformed native " + kind + " progress metadata")
    numeric_counts = fields[1:3] if kind == "PP" else fields[1:2]
    if any(not re.fullmatch(r"[0-9]+", value) for value in numeric_counts):
        raise B.Refusal("native prompt progress counts must be nonnegative integers")
    if kind != "PP":
        count = int(fields[1])
        if count > prompt_tokens:
            raise B.Refusal("native reused count exceeds the pending prompt")
        return {"kind": kind, "reused_tokens": count}
    position, total = map(int, fields[1:3])
    try:
        elapsed_ms, rate = map(float, fields[3:5])
    except ValueError as error:
        raise B.Refusal("native prompt progress has invalid timing or rate") from error
    if total != prompt_tokens or not 0 <= position <= total:
        raise B.Refusal("native prompt progress disagrees with the pending prompt length")
    if any(not math.isfinite(value) or value < 0 for value in (elapsed_ms, rate)):
        raise B.Refusal("native prompt progress has nonfinite or negative timing/rate")
    return {"kind": kind, "position": position, "total_tokens": total,
            "elapsed_ms": elapsed_ms, "fresh_tokens_per_second": rate}


def interval_metrics(arrivals, start, end):
    """Actual complete gaps plus boundary-censored silence inside one interval."""
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise B.Refusal("mixed-load interval must have finite positive duration")
    if any(not math.isfinite(t) or t < 0 for t in arrivals) or any(b < a for a, b in zip(arrivals, arrivals[1:])):
        raise B.Refusal("mixed-load arrival clocks are invalid")
    inside = [t for t in arrivals if start <= t <= end]
    complete = [b - a for a, b in zip(arrivals, arrivals[1:]) if start <= a <= b <= end]
    ending_inside = [b - a for a, b in zip(arrivals, arrivals[1:]) if start <= b <= end]
    boundaries = [start, *inside, end]
    silent = [b - a for a, b in zip(boundaries, boundaries[1:])]
    return {"seconds": end - start, "observed_tokens": len(inside),
            "output_tokens_per_second": len(inside) / (end - start),
            "complete_intertoken_gaps": len(complete),
            "maximum_complete_intertoken_gap_ms": max(complete) * 1000 if complete else None,
            "maximum_observed_intertoken_gap_ending_in_interval_ms": max(ending_inside) * 1000 if ending_inside else None,
            "ending_gap_scope": "consecutive actual arrivals; the first gap may begin before this interval",
            "maximum_silent_interval_ms": max(silent, default=0) * 1000,
            "silent_interval_scope": "clipped to interval; start/end waits are censored, not invented token events"}


def exercise(engine, backgrounds, joiner, tok, seconds, warm_tokens, sampler=None):
    """Strict native protocol state machine. Each slot has one explicit owner."""
    if len(backgrounds) != len(ESTABLISHED):
        raise B.Refusal("mixed-load diagnostic requires exactly four established streams")
    born = time.monotonic()
    deadline = born + seconds
    outputs = {s: [] for s in (*ESTABLISHED, JOINER)}
    arrivals = {s: [] for s in (*ESTABLISHED, JOINER)}
    pending = list(ESTABLISHED)
    admitted, active, completed, admission_done = {}, set(), {}, {}
    admission_metadata, reuse_announced, reuse_finished, progress_position = [], {}, {}, {}
    join_sent = join_first = join_done = None
    cancel_sent = None
    active_at_join = None
    for slot, case in zip(ESTABLISHED, backgrounds):
        engine.send(f"BGEN {slot} {case['max_new']} temperature=0 seed={case.get('decode_seed', 202611853)} "
                    + ",".join(map(str, case["tokens"])))
    while True:
        if sampler is not None:
            sampler.check()
        when, line = engine.next(deadline)
        tag = line.split(maxsplit=1)
        if tag and tag[0] in ("RESUME", "PP", "REUSED"):
            if not pending or pending[0] in admission_done:
                raise B.Refusal("prompt progress has no owned pending admission")
            slot = pending[0]
            case = joiner if slot == JOINER else backgrounds[slot]
            metadata = parse_admission_metadata(line, case.get("prompt_tokens", B.CAPACITY))
            if metadata["kind"] == "RESUME":
                if slot in reuse_announced:
                    raise B.Refusal("duplicate native RESUME metadata for one admission")
                reuse_announced[slot] = metadata["reused_tokens"]
            elif metadata["kind"] == "REUSED":
                if slot in reuse_finished or (slot in reuse_announced and reuse_announced[slot] != metadata["reused_tokens"]):
                    raise B.Refusal("native REUSED metadata is duplicated or differs from RESUME")
                reuse_finished[slot] = metadata["reused_tokens"]
            else:
                if metadata["position"] < progress_position.get(slot, 0):
                    raise B.Refusal("native prompt progress moved backwards")
                progress_position[slot] = metadata["position"]
            admission_metadata.append({"slot": slot, "arrival_ms": (when - born) * 1000, **metadata})
        elif line.startswith("T "):
            if not pending:
                raise B.Refusal("standalone token has no owned pending admission")
            slot = pending[0]
            fields = line.split()
            if len(fields) != 2:
                raise B.Refusal("malformed mixed-load admission token")
            outputs[slot].append(int(fields[1])); arrivals[slot].append(when)
            if slot == JOINER and join_first is None:
                join_first = when
        elif line.startswith("DONE "):
            if not pending or pending[0] in admission_done:
                raise B.Refusal("unexpected or duplicate mixed-load admission DONE")
            slot = pending[0]
            value = B.done_fields(line)
            if value["generated"] != len(outputs[slot]) or value["finish"] == "cancel":
                raise B.Refusal("mixed-load admission token count or finish is invalid")
            if any(counts.get(slot, value["reused_tokens"]) != value["reused_tokens"]
                   for counts in (reuse_announced, reuse_finished)):
                raise B.Refusal("native progress reuse counts disagree with admission DONE")
            admission_done[slot] = value
        elif line.startswith("BADM "):
            fields = line.split()
            if len(fields) != 3:
                raise B.Refusal("malformed mixed-load BADM")
            slot, continuation = int(fields[1]), int(fields[2])
            if not pending or slot != pending[0] or slot not in admission_done or slot in admitted:
                raise B.Refusal("mixed-load admission did not preserve requested slot ownership")
            if continuation != 1:
                raise B.Refusal("a mixed-load request ended during admission; sustained continuation is required")
            pending.pop(0); admitted[slot] = when; active.add(slot)
        elif line.startswith("BT "):
            fields = line.split()
            if len(fields) != 3:
                raise B.Refusal("malformed mixed-load batch token")
            slot, token = int(fields[1]), int(fields[2])
            if slot not in active:
                raise B.Refusal("mixed-load token belongs to an unadmitted or completed slot")
            outputs[slot].append(token); arrivals[slot].append(when)
            if slot == JOINER and join_first is None:
                join_first = when
        elif line.startswith("BDONE "):
            fields = line.split()
            if len(fields) < 5:
                raise B.Refusal("short mixed-load BDONE")
            slot, count, finish, native_ms = int(fields[1]), int(fields[2]), fields[3], float(fields[4])
            if slot not in active or count != len(outputs[slot]) or finish not in ("stop", "length", "cancel"):
                raise B.Refusal("mixed-load batch completion lost token count or lifecycle identity")
            if not math.isfinite(native_ms) or native_ms < 0:
                raise B.Refusal("mixed-load batch completion has invalid timing")
            completed[slot] = {"generated": count, "finish": finish, "native_ms": native_ms, "at": when}
            active.remove(slot)
            if slot in ESTABLISHED and join_done is None:
                raise B.Refusal("an established stream ended before the joiner completed; four-stream latency is unqualified")
            if slot == JOINER:
                if finish not in ("stop", "length") or join_sent is None or not set(ESTABLISHED).issubset(active):
                    raise B.Refusal("joiner did not complete normally with four established streams still active")
                join_done = when
                cancel_sent = time.monotonic()
                for owned in ESTABLISHED:
                    engine.send(f"BSTOP {owned}")
        elif line.startswith(("INFO ", "READY ")):
            continue
        else:
            raise B.Refusal("unsupported native event during bounded mixed-load protocol: " + line[:120])
        if join_sent is None and not pending and all(len(outputs[s]) >= warm_tokens for s in ESTABLISHED):
            if active != set(ESTABLISHED):
                raise B.Refusal("four established streams are not active at join submission")
            active_at_join = sorted(active)
            join_sent = time.monotonic()
            pending.append(JOINER)
            engine.send(f"BGEN {JOINER} {joiner['max_new']} temperature=0 seed=202611853 "
                        + ",".join(map(str, joiner["tokens"])))
        if join_done is not None and all(s in completed for s in ESTABLISHED):
            break
    if pending or active or join_first is None or join_sent is None or join_first < join_sent:
        raise B.Refusal("mixed-load protocol did not drain every owned request cleanly")
    if not all(completed[s]["finish"] == "cancel" for s in ESTABLISHED):
        raise B.Refusal("an established stream reached its natural end before cancellation; increase its bounded output allowance")
    baseline_start = max(admitted[s] for s in ESTABLISHED)
    streams = []
    for slot in ESTABLISHED:
        before = interval_metrics(arrivals[slot], baseline_start, join_sent)
        during = interval_metrics(arrivals[slot], join_sent, join_done)
        streams.append({"slot": slot, "generated": len(outputs[slot]),
                        "before_join": before, "during_join": during,
                        "arrival_ms": [(t - born) * 1000 for t in arrivals[slot]],
                        "finish": completed[slot]["finish"],
                        "cancel_drain_ms": (completed[slot]["at"] - cancel_sent) * 1000})
    text = B.scrub_output(tok, outputs[JOINER])
    correct = B.check_answer(joiner, text)
    return {"infrastructure_status": "passed", "quality_status": "passed" if correct else "failed",
            "active_established_slots_at_join": active_at_join,
            "all_four_survived_until_joiner_completion": True,
            "warm_tokens_required_each": warm_tokens,
            "admission_metadata": admission_metadata,
            "joiner": {"id": joiner["id"], "prompt_tokens": joiner["prompt_tokens"],
                       "expected": joiner["expected"], "actual": text, "correct": correct,
                       "output_ids": outputs[JOINER], "ttft_ms": (join_first - join_sent) * 1000,
                       "admission_wall_ms": (admitted[JOINER] - join_sent) * 1000,
                       "completion_wall_ms": (join_done - join_sent) * 1000,
                       "native_admission": admission_done[JOINER],
                       "generated": len(outputs[JOINER]), "finish": completed[JOINER]["finish"]},
            "established_streams": streams,
            "maximum_silent_interval_during_join_ms": max(x["during_join"]["maximum_silent_interval_ms"] for x in streams),
            "maximum_complete_gap_during_join_ms": max((x["during_join"]["maximum_complete_intertoken_gap_ms"] for x in streams
                                                        if x["during_join"]["maximum_complete_intertoken_gap_ms"] is not None), default=None),
            "maximum_observed_gap_ending_during_join_ms": max((x["during_join"]["maximum_observed_intertoken_gap_ending_in_interval_ms"] for x in streams
                                                              if x["during_join"]["maximum_observed_intertoken_gap_ending_in_interval_ms"] is not None), default=None),
            "all_owned_streams_cancelled_and_drained": True,
            "metric_scope": "four observed active streams during one fresh32K join; eight full slots allocated; finite fixture diagnostic"}


def run_profile(profile, cfg, contract, backgrounds, joiner, tok, directory, guard, cli):
    executable = Path(profile["executable"])
    if B.sha(executable) != profile["sha256"]:
        raise B.Refusal("mixed-load executable differs from its pinned profile")
    pinned = B.verify_library_manifest(profile, cfg)
    args = B.apply_profile_args(B.engine_args(cfg), profile)
    env, public = B.launch_env(cfg, contract, profile)
    report = {"profile": profile["name"], "started_at": B.utc(), "status": "running",
              "infrastructure_status": "running", "engine_sha256": profile["sha256"],
              "args": args, "environment": public, "library_manifest": pinned}
    sampler = B.TelemetrySampler(directory.parent / (directory.name + "-telemetry.jsonl"))
    engine = None
    try:
        guard.check(force=True, before_start=True)
        report["gpu_before"] = B.telemetry()
        sampler.start()
        engine = B.NativeEngine(executable, args, env, cfg.get("cwd"), directory, guard, cli.ready_seconds)
        with engine:
            report.update(info=engine.info, ready_ms=engine.ready_ms, resolved_drafting=B.validate_capacity(engine),
                          loaded_libraries=B.loaded_library_manifest(engine.proc.pid, pinned))
            report["mixed"] = exercise(engine, backgrounds, joiner, tok, cli.case_seconds, cli.warm_tokens, sampler)
            B.verify_library_manifest(profile, cfg)
            sampler.check()
            report["gpu_after_work"] = B.telemetry()
        report.update(infrastructure_status="passed", quality_status=report["mixed"]["quality_status"])
        report["status"] = "passed" if report["quality_status"] == "passed" else "completed_with_quality_failure"
    except (B.Refusal, OSError, ValueError, subprocess.SubprocessError) as error:
        report.update(status="failed", infrastructure_status="failed", error=str(error))
    except KeyboardInterrupt as error:
        report.update(status="interrupted", infrastructure_status="failed", error=str(error))
        raise
    finally:
        report["gpu_work_started"] = engine is not None and engine.proc is not None
        report["owned_process_group_empty"] = engine is None or getattr(engine, "owned_process_group_empty", engine.proc is None)
        if sampler.thread.ident is not None:
            report["periodic_telemetry"] = sampler.stop()
            if sampler.errors:
                report.update(status="failed", infrastructure_status="failed", error=sampler.errors[0])
        report["finished_at"] = B.utc()
        directory.mkdir(exist_ok=True)
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def comparisons(arms):
    """Diagnostic ratios against the first explicitly selected profile."""
    if not arms:
        return []
    baseline = arms[0]
    rows = []
    for arm in arms[1:]:
        valid = all(x.get("infrastructure_status") == "passed"
                    and x.get("owned_process_group_empty") is True
                    and x.get("mixed", {}).get("joiner", {}).get("correct") is True
                    and x.get("mixed", {}).get("all_four_survived_until_joiner_completion") is True
                    and x.get("mixed", {}).get("all_owned_streams_cancelled_and_drained") is True
                    for x in (baseline, arm))
        row = {"baseline": baseline["profile"], "candidate": arm["profile"],
               "eligible_diagnostic_comparison": valid,
               "scope": "one finite mixed-load fixture; lower latency ratio is better; no statistical speed claim"}
        if valid:
            a, b = baseline["mixed"], arm["mixed"]
            for key in ("ttft_ms", "admission_wall_ms", "completion_wall_ms"):
                old, new = a["joiner"][key], b["joiner"][key]
                row[key] = {"baseline": old, "candidate": new, "candidate_over_baseline": new / old if old > 0 else None}
            key = "maximum_silent_interval_during_join_ms"
            row[key] = {"baseline": a[key], "candidate": b[key], "candidate_over_baseline": b[key] / a[key] if a[key] > 0 else None}
        rows.append(row)
    return rows


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true"); mode.add_argument("--execute", action="store_true")
    result.add_argument("--profiles-file", type=Path, default=ROOT / "profiles.json")
    result.add_argument("--profile", nargs="+", default=["official0150_current", "recommended_gdn"])
    result.add_argument("--background-tokens", type=int, default=1024)
    result.add_argument("--warm-tokens", type=int, default=32)
    result.add_argument("--quality-seed", type=int, default=202611853)
    result.add_argument("--owner", default="")
    result.add_argument("--run-dir", type=Path)
    result.add_argument("--ready-seconds", type=int, default=600)
    result.add_argument("--case-seconds", type=int, default=600)
    result.add_argument("--total-seconds", type=int, default=3600)
    return result


def main():
    cli = parser().parse_args()
    if not 1024 <= cli.background_tokens <= 4096 or not 32 <= cli.warm_tokens <= 256:
        raise B.Refusal("mixed-load output/warmup allowance is outside the bounded diagnostic")
    if not 30 <= cli.ready_seconds <= 1800 or not 30 <= cli.case_seconds <= 1800 or not 60 <= cli.total_seconds <= 10800:
        raise B.Refusal("mixed-load timeout is outside the bounded diagnostic")
    cfg, contract, config_sha, contract_sha = B.current_config()
    control, candidates = B.read_profiles(cli.profiles_file, cfg)
    available = {control["name"]: control, **candidates}
    if len(cli.profile) != len(set(cli.profile)) or any(name not in available for name in cli.profile):
        raise B.Refusal("mixed-load profiles must be distinct known JSON profile names")
    expected = contract.get("hashes", {}).get(str(cfg["exe"]))
    if expected and expected != control["sha256"]:
        raise B.Refusal("actual production engine differs from the recovery contract")
    tok, template, hashes = B.load_tokenizer(cfg)
    _, batch = B.performance_cases(tok, template, cli.background_tokens, 384, cli.quality_seed)
    backgrounds = batch[:4]
    joiner = B.retrieval_case(tok, template, "mixed_fresh_retrieval_32k", 32000, cli.quality_seed)
    profiles = [available[name] for name in cli.profile]
    plan = {"schema_version": 1, "scope": "finite mixed-load latency diagnostic; no statistical claim",
            "gpu_work_started": False, "context": B.CAPACITY, "allocated_slots": B.SLOTS,
            "generating_streams_before_join": 4, "joiner_slot": JOINER, "joiner_prompt_tokens": joiner["prompt_tokens"],
            "background_output_allowance_each": cli.background_tokens, "warm_tokens_each": cli.warm_tokens,
            "joiner_marker": joiner["expected"], "quality_seed": cli.quality_seed,
            "tokenizer_sha256": hashes, "config_sha256": config_sha, "contract_sha256": contract_sha,
            "profiles_sha256": B.sha(cli.profiles_file), "benchmark_helper_sha256": B.sha(B.__file__),
            "mixed_source_sha256": B.sha(__file__),
            "schedule": [{"name": p["name"], "executable": p["executable"], "sha256": p["sha256"],
                          "args": B.apply_profile_args(B.engine_args(cfg), p),
                          "environment": B.launch_env(cfg, contract, p)[1], "library_manifest": p["library_manifest"]} for p in profiles],
            "limitations": ["four generating streams plus one joining request, with all8fullslots allocated",
                            "output-arrival timing includes native protocol delivery",
                            "boundary-censored silence is reported separately from complete intertoken gaps",
                            "single fixture per profile; fresh heldout quality and paired throughput are separate",
                            "no service or production configuration changes"]}
    if not cli.execute:
        print(json.dumps(plan, indent=2)); return 0
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,100}", cli.owner):
        raise B.Refusal("execution requires the exact owned maintenance owner")
    directory = cli.run_dir or ROOT / "receipts" / ("mixed-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "fixtures.json").write_text(json.dumps({"backgrounds": backgrounds, "joiner": joiner}, indent=2) + "\n")
    (directory / "profiles.frozen.json").write_text(json.dumps(profiles, indent=2) + "\n")
    report = {**plan, "status": "running", "started_at": B.utc(), "arms": []}
    guard = B.Guard(cli.owner, config_sha, contract_sha, cli.total_seconds)
    with (ROOT / ".benchmark-engine.lock").open("a+") as lock:
        try:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise B.Refusal("another native benchmark owns the shared model window") from error
            guard.check(force=True, before_start=True)
            for profile in profiles:
                print(json.dumps({"event": "mixed_launch", "profile": profile["name"]}), flush=True)
                arm = run_profile(profile, cfg, contract, backgrounds, joiner, tok, directory / profile["name"], guard, cli)
                report["arms"].append(arm)
                report["gpu_work_started"] = any(x["gpu_work_started"] for x in report["arms"])
                (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"event": "mixed_result", "profile": profile["name"], "status": arm["status"],
                                  "joiner": arm.get("mixed", {}).get("joiner"),
                                  "max_silent_during_join_ms": arm.get("mixed", {}).get("maximum_silent_interval_during_join_ms")}), flush=True)
                if arm["infrastructure_status"] != "passed" or not arm["owned_process_group_empty"]:
                    raise B.Refusal("mixed-load arm failed infrastructure; stopping sequential diagnostic")
                guard.check(force=True, before_start=True)
            report["status"] = "completed_with_quality_failure" if any(x["quality_status"] != "passed" for x in report["arms"]) else "passed"
            report["comparison"] = comparisons(report["arms"])
        except BaseException as error:
            report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            raise
        finally:
            report["finished_at"] = B.utc()
            (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(directory / "report.json")}), flush=True)
    return 0


if __name__ == "__main__":
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"mixed-load diagnostic interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGHUP, interrupted)
    try:
        raise SystemExit(main())
    except (B.Refusal, OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "refused", "error": str(error)}))
        raise SystemExit(2)
