#!/usr/bin/env python3
"""Bounded stock Strata comparison. Default: plan only; no service mutations.

Every phase retains eight slots, 131072 context, resident INT8 and all experts.
Native output events measure actual streams, including admission and fairness.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import signal
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
SOURCE = Path("/home/USER/Documents/strata-lsa-resume-20261103/strata-src")
CONFIG = Path("/etc/onyx-inference/strata-server.json")
CONTRACT = Path("/etc/onyx-inference/strata.json")
MAINTENANCE = "/usr/local/libexec/onyx-inference-recovery/strata-maintenance.py"
PRIMARY = "qwen38-flash-next-strata.service"
CAPACITY = 131072
SLOTS = 8
CRITICAL_MARKERS = {"exact_canary", "retrieval_4k", "retrieval_16k", "accepted_prefix_reuse", "heldout_retrieval_32k", "heldout_retrieval_125k", "json_structure"}

class Refusal(RuntimeError):
    pass


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(argv, timeout=15):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        # Do not include arbitrary command output: configuration paths can carry
        # unrelated account information. These commands have no model input.
        raise Refusal(f"required state read failed: {Path(argv[0]).name} ({result.returncode})")
    return result.stdout.strip()


def current_config():
    raw = CONFIG.read_bytes()
    cfg = json.loads(raw)
    contract_raw = CONTRACT.read_bytes()
    contract = json.loads(contract_raw)
    if not isinstance(cfg.get("args"), list) or not all(isinstance(x, str) for x in cfg["args"]):
        raise Refusal("production engine arguments are missing or invalid")
    return cfg, contract, hashlib.sha256(raw).hexdigest(), hashlib.sha256(contract_raw).hexdigest()


def replace_arg(args, flag, value):
    args = list(args)
    if flag in args:
        index = args.index(flag)
        if index + 1 == len(args):
            raise Refusal(f"missing value for {flag}")
        args[index + 1] = str(value)
    else:
        args.extend((flag, str(value)))
    return args


def arg_value(args, flag):
    if flag not in args:
        return None
    index = args.index(flag)
    return args[index + 1] if index + 1 < len(args) else None


def load_tokenizer(cfg):
    sys.path.insert(0, str(SOURCE / "tools"))
    sys.path.insert(0, str(SOURCE))
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    directory = Path(cfg["tokenizer"])
    vocab = json.loads((directory / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    merges = (directory / "merges.txt").read_text().split("\n")
    types = json.loads((directory / "token_type.json").read_text())
    options = json.loads((directory / "tokenizer.json").read_text())
    tok = Tokenizer(tokens, merges, types, options.get("pre") or "qwen35", options.get("special_ids") or None)
    template = ChatTemplate(directory / "chat_template.jinja")
    files = {name: sha(directory / name) for name in
             ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json", "chat_template.jinja")}
    return tok, template, files


def encode_case(tok, template, name, question, expected, max_new=48, messages=None):
    messages = messages or [{"role": "system", "content": "Follow the user's output format exactly. Do not explain."},
                            {"role": "user", "content": question}]
    prompt = template.render(messages, enable_thinking=False)
    ids = tok.encode(prompt, parse_special=True)
    return {"id": name, "messages": messages, "expected": expected, "max_new": max_new,
            "prompt": prompt, "tokens": ids, "prompt_tokens": len(ids),
            "token_sha256": hashlib.sha256((",".join(map(str, ids)) + "\n").encode()).hexdigest()}


def scrub_output(tok, ids):
    text = tok.decode(ids)
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return text.strip()


def done_fields(line):
    parts = line.split()
    if len(parts) < 9:
        raise Refusal("short native DONE record")
    values = {"generated": int(parts[1]), "prompt_tokens": int(parts[2]), "prompt_ms": float(parts[3]),
              "decode_ms": float(parts[4]), "finish": parts[5], "drafts_accepted": int(parts[6]),
              "drafts_offered": int(parts[7]), "reused_tokens": int(parts[8]), "raw": line}
    if len(parts) > 14 and "=" not in parts[14]:
        values["read_tokens"] = int(parts[14])
    if values["finish"] not in ("stop", "length", "cancel"):
        raise Refusal("native engine reported a failed or nonfinite completion")
    if not 0 <= values["drafts_accepted"] <= values["drafts_offered"]:
        raise Refusal("native speculative counters are invalid")
    return values


class Guard:
    def __init__(self, owner, config_sha, contract_sha, total_seconds):
        self.owner, self.config_sha, self.contract_sha = owner, config_sha, contract_sha
        self.deadline = time.monotonic() + total_seconds
        self.last_check = 0.0

    def check(self, force=False, before_start=False):
        now = time.monotonic()
        if now >= self.deadline:
            raise Refusal("global engine qualification deadline expired")
        if not force and now - self.last_check < 3:
            return
        self.last_check = now
        if sha(CONFIG) != self.config_sha or sha(CONTRACT) != self.contract_sha:
            raise Refusal("production configuration changed during qualification")
        cmd = ["python3", MAINTENANCE, "status"]
        if os.geteuid() != 0:
            cmd = ["sudo", "-n", *cmd]
        state = json.loads(command(cmd))
        hold = state.get("hold") or {}
        if hold.get("kind") != "owned-strata-maintenance" or hold.get("owner") != self.owner:
            raise Refusal("exact owned production maintenance HOLD is absent")
        status = command(["systemctl", "show", PRIMARY, "--property=ActiveState", "--value"])
        if status != "inactive":
            raise Refusal("production must remain inactive throughout qualification")
        if before_start:
            rows = command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"])
            for line in rows.splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) != 3:
                    raise Refusal("cannot establish existing GPU process identity")
                raise Refusal("another compute process already owns the GPU")
            gpu = command(["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
            if int(gpu) < 80000:
                raise Refusal("insufficient free GPU memory for an uncontended full-model launch")


class NativeEngine:
    def __init__(self, executable, args, env, cwd, directory, guard, ready_seconds):
        self.executable, self.args, self.env = executable, args, env
        self.cwd, self.directory, self.guard = cwd, directory, guard
        self.ready_seconds = ready_seconds
        self.proc = None
        self.lines = queue.Queue()
        self.info = {}
        self.started = None
        self.log = None
        self.protocol = None

    def __enter__(self):
        self.guard.check(force=True, before_start=True)
        self.directory.mkdir(exist_ok=False)
        self.log = (self.directory / "engine.stderr.log").open("x", encoding="utf-8")
        self.protocol = (self.directory / "engine.stdout.log").open("x", encoding="utf-8")
        try:
            self.started = time.monotonic()
            self.proc = subprocess.Popen([str(self.executable), "--serve", *self.args], cwd=self.cwd, env=self.env,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                         encoding="utf-8", text=True, bufsize=1, start_new_session=True)
            def pump():
                try:
                    for line in self.proc.stdout:
                        text = line.rstrip("\n")
                        self.protocol.write(text + "\n")
                        self.protocol.flush()
                        self.lines.put((time.monotonic(), text))
                finally:
                    self.lines.put((time.monotonic(), None))
            self.reader = threading.Thread(target=pump, daemon=True)
            self.reader.start()
            deadline = self.started + self.ready_seconds
            while True:
                _, line = self.next(deadline)
                if line.startswith("INFO "):
                    self.info.update(dict(x.split("=", 1) for x in line[5:].split() if "=" in x))
                elif line.startswith("READY "):
                    self.ready = line
                    self.ready_ms = (time.monotonic() - self.started) * 1000
                    return self
        except BaseException:
            self.close()
            raise

    def next(self, deadline):
        while True:
            self.guard.check()
            remaining = min(deadline, self.guard.deadline) - time.monotonic()
            if remaining <= 0:
                raise Refusal("native engine phase deadline expired")
            try:
                timestamp, line = self.lines.get(timeout=min(1, remaining))
            except queue.Empty:
                continue
            if line is None:
                raise Refusal(f"native engine exited before completing the protocol (code {self.proc.poll()})")
            if line.startswith(("ERR ", "SERR ")):
                raise Refusal("native engine refused the test protocol: " + line[:1000])
            return timestamp, line

    def send(self, text):
        self.guard.check()
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def close(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                with contextlib.suppress(Exception):
                    self.proc.stdin.write("STOP\nQUIT\n")
                    self.proc.stdin.flush()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(self.proc.pid, signal.SIGTERM)
                    try:
                        self.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(self.proc.pid, signal.SIGKILL)
                        self.proc.wait(timeout=5)
            with contextlib.suppress(Exception):
                self.proc.stdin.close()
            if hasattr(self, "reader"):
                self.reader.join(timeout=2)
            with contextlib.suppress(Exception):
                self.proc.stdout.close()
        if self.log:
            self.log.close()
        if self.protocol:
            self.protocol.close()

    def __exit__(self, *_):
        self.close()

    def generate(self, case, tok, seconds):
        born = time.monotonic()
        self.send(f"GEN {case['max_new']} temperature=0 seed=20261103 " + ",".join(map(str, case["tokens"])))
        ids, first, arrivals = [], None, []
        while True:
            when, line = self.next(born + seconds)
            if line.startswith("T "):
                first = first or when
                ids.append(int(line.split()[1]))
                arrivals.append(when)
            elif line.startswith("DONE "):
                done = done_fields(line)
                if done["generated"] != len(ids):
                    raise Refusal("native token count disagrees with DONE")
                text = scrub_output(tok, ids)
                correct = check_answer(case, text)
                reused = done["reused_tokens"] > 0
                reuse_gate = not case.get("require_reused") or reused
                return {"id": case["id"], "status": "passed" if correct and reuse_gate else "failed",
                        "quality_status": "not_scored" if case["expected"] is None else "passed" if correct else "failed",
                        "infrastructure_status": "passed" if reuse_gate else "failed",
                        "expected": case["expected"], "actual": text, "output_ids": ids,
                        "prompt_tokens": case["prompt_tokens"], "token_sha256": case["token_sha256"],
                        "ttft_ms": None if first is None else (first - born) * 1000,
                        "wall_ms": (when - born) * 1000, "prefix_reuse_verified": reused,
                        "prefix_reuse_gate_passed": reuse_gate, "native": done,
                        "output_arrival_ms": [(x - born) * 1000 for x in arrivals],
                        "output_metrics": output_metrics([arrivals], born, [when], trim_seconds=0.25)}

    def batch(self, cases, tok, seconds):
        born = time.monotonic()
        outputs = {slot: [] for slot in range(8)}
        first = {}
        admission = {}
        admitted = set()
        active = set()
        completed = {}
        peak = 0
        current = 0
        stop_sent = False
        for slot, case in enumerate(cases):
            self.send(f"BGEN {slot} {case['max_new']} temperature=0 seed=20261103 " + ",".join(map(str, case["tokens"])))
        while len(completed) != 8:
            when, line = self.next(born + seconds)
            if line.startswith("T "):
                if current >= 8:
                    raise Refusal("unexpected admission token after eight BADM records")
                outputs[current].append(int(line.split()[1]))
                first.setdefault(current, when)
            elif line.startswith("DONE "):
                admission[current] = done_fields(line)
            elif line.startswith("BADM "):
                _, slot, cont = line.split()
                slot, cont = int(slot), int(cont)
                if slot != current or slot in admitted or cont != 1:
                    raise Refusal("batch admission did not preserve eight requested live slots")
                admitted.add(slot)
                active.add(slot)
                peak = max(peak, len(active))
                current += 1
            elif line.startswith("BT "):
                _, slot, token = line.split()
                slot = int(slot)
                if slot not in admitted:
                    raise Refusal("batch token preceded its admission")
                outputs[slot].append(int(token))
                first.setdefault(slot, when)
            elif line.startswith("BDONE "):
                parts = line.split()
                slot, count, finish = int(parts[1]), int(parts[2]), parts[3]
                if slot not in admitted or slot in completed or count != len(outputs[slot]):
                    raise Refusal("native batch token count or slot lifecycle mismatch")
                if finish not in ("stop", "length", "cancel"):
                    raise Refusal("batch completion was failed or nonfinite")
                completed[slot] = {"generated": count, "finish": finish, "native_ms": float(parts[4]),
                                   "wall_ms": (when - born) * 1000}
                active.discard(slot)
            if not stop_sent and len(admitted) == 8 and all(len(outputs[s]) >= 3 for s in range(8)):
                for slot in sorted(active):
                    self.send(f"BSTOP {slot}")
                stop_sent = True
        status = "passed" if peak == 8 and stop_sent and any(x["finish"] == "cancel" for x in completed.values()) else "failed"
        return {"id": "eight_slot_copy_decode_cancel", "status": status, "peak_live_slots": peak,
                "all_slots_cancel_requested": stop_sent,
                "slots": [{"slot": s, "output_ids": outputs[s], "text": scrub_output(tok, outputs[s]),
                           "ttft_ms": (first[s] - born) * 1000, **completed[s]} for s in range(8)],
                "admissions": {str(s): admission[s] for s in sorted(admission)}}

    def throughput(self, cases, tok, seconds, trim_seconds=0.25):
        """Eight real queued streams, completed normally; no three-token stop."""
        if len(cases) != SLOTS:
            raise Refusal("throughput workload must contain exactly eight streams")
        born = time.monotonic()
        outputs = {slot: [] for slot in range(SLOTS)}
        arrivals = {slot: [] for slot in range(SLOTS)}
        admitted, active, admission, completed = set(), set(), {}, {}
        peak, current = 0, 0
        last_admission = born
        for slot, case in enumerate(cases):
            self.send(f"BGEN {slot} {case['max_new']} temperature=0 seed={case.get('decode_seed', 20261104)} "
                      + ",".join(map(str, case["tokens"])))
        while len(completed) < SLOTS:
            when, line = self.next(born + seconds)
            if line.startswith("T "):
                if current >= SLOTS:
                    raise Refusal("unexpected standalone admission token after all slots")
                outputs[current].append(int(line.split()[1])); arrivals[current].append(when)
            elif line.startswith("DONE "):
                if current >= SLOTS or current in admission:
                    raise Refusal("duplicate or unexpected admission completion")
                admission[current] = done_fields(line)
            elif line.startswith("BADM "):
                _, slot, cont = line.split(); slot, cont = int(slot), int(cont)
                if slot != current or slot in admitted or cont != 1 or slot not in admission:
                    raise Refusal("eight-stream admission lost slot identity or continuation")
                admitted.add(slot); active.add(slot); current += 1
                peak = max(peak, len(active)); last_admission = when
            elif line.startswith("BT "):
                _, slot, token = line.split(); slot = int(slot)
                if slot not in active:
                    raise Refusal("batch token belongs to an unadmitted or completed slot")
                outputs[slot].append(int(token)); arrivals[slot].append(when)
            elif line.startswith("BDONE "):
                fields = line.split(); slot, count, finish = int(fields[1]), int(fields[2]), fields[3]
                if slot not in active or count != len(outputs[slot]):
                    raise Refusal("batch output count differs from observed token events")
                if finish not in ("length", "stop"):
                    raise Refusal("sustained workload was cancelled, failed or nonfinite")
                completed[slot] = {"generated": count, "finish": finish,
                                   "native_ms": float(fields[4]), "completed_at": when}
                active.remove(slot)
        events = [arrivals[s] for s in range(SLOTS)]
        ends = [completed[s]["completed_at"] for s in range(SLOTS)]
        metrics = output_metrics(events, born, ends, trim_seconds, live_from=last_admission)
        cap_reached = all(len(outputs[s]) == cases[s]["max_new"] for s in range(SLOTS))
        eligible = (peak == SLOTS and cap_reached and metrics["common_window"]["valid"]
                    and min(map(len, events)) >= 128)
        return {"id": "eight_stream_sustained", "status": "passed", "peak_live_slots": peak,
                "requested_output_tokens_each": [c["max_new"] for c in cases],
                "all_output_caps_reached": cap_reached, "eligible_for_speed_comparison": eligible,
                "metric_scope": "observed native output-token arrivals; not eight times a solo rate",
                "metrics": metrics,
                "slots": [{"slot": s, "output_ids": outputs[s], "text": scrub_output(tok, outputs[s]),
                           "arrival_ms": [(x - born) * 1000 for x in arrivals[s]],
                           "ttft_ms": (arrivals[s][0] - born) * 1000 if arrivals[s] else None,
                           "wall_ms": (completed[s]["completed_at"] - born) * 1000,
                           **{k: v for k, v in completed[s].items() if k != "completed_at"}}
                          for s in range(SLOTS)],
                "admissions": {str(s): admission[s] for s in sorted(admission)}}


def check_answer(case, text):
    if case.get("expected") is None:
        return True
    if case.get("checker") == "json":
        try:
            return json.loads(text) == case["expected_json"]
        except (ValueError, TypeError):
            return False
    return text == case["expected"]


def output_metrics(events, born, completed, trim_seconds=0.25, live_from=None):
    """One shared time interval with every stream alive, based on real arrivals."""
    if not events or len(events) != len(completed) or any(not row for row in events):
        return {"observed_output_tokens": sum(map(len, events)),
                "common_window": {"valid": False, "reason": "a stream emitted no tokens"}}
    if any(any(b < a for a, b in zip(row, row[1:])) for row in events):
        raise Refusal("token arrival clocks are not monotonic")
    finish = max(completed)
    wall = finish - born
    if wall <= 0:
        raise Refusal("nonpositive observed request duration")
    start = max(max(row[0] for row in events), live_from if live_from is not None else born) + trim_seconds
    end = min(completed) - trim_seconds
    valid = end > start
    counts = [sum(start <= t <= end for t in row) if valid else 0 for row in events]
    valid = valid and all(n > 0 for n in counts)
    duration = end - start if end > start else 0
    rates = [n / duration for n in counts] if valid else []
    windows = [[t for t in row if start <= t <= end] for row in events]
    gaps = [max((b - a for a, b in zip(row, row[1:])), default=0) * 1000 for row in windows]
    fairness = (sum(rates) ** 2 / (len(rates) * sum(r * r for r in rates))) if rates else None
    return {"observed_output_tokens": sum(map(len, events)), "whole_request_seconds": wall,
            "whole_request_output_tokens_per_second": sum(map(len, events)) / wall,
            "per_stream_output_tokens": list(map(len, events)),
            "per_stream_ttft_ms": [(row[0] - born) * 1000 for row in events],
            "common_window": {"valid": valid, "trim_seconds": trim_seconds,
                              "start_ms": (start - born) * 1000, "end_ms": (end - born) * 1000,
                              "seconds": duration, "per_stream_output_tokens": counts,
                              "aggregate_output_tokens_per_second": sum(rates) if rates else None,
                              "per_stream_output_tokens_per_second": rates,
                              "jain_fairness": fairness,
                              "minimum_over_maximum_stream_rate": min(rates) / max(rates) if rates else None,
                              "per_stream_max_intertoken_gap_ms": gaps}}


def engine_args(cfg, phase="full"):
    args = list(cfg["args"])
    if cfg.get("user_args") or cfg.get("drafting") is not None or cfg.get("mtp") is False:
        raise Refusal("configuration needs wrapper argument shaping before a matched test")
    if cfg.get("parallel") != SLOTS:
        raise Refusal("production no longer requests eight slots")
    args = replace_arg(args, "--batch", SLOTS)
    if arg_value(args, "--max-context") != str(CAPACITY) or arg_value(args, "--kv-resident") != str(CAPACITY):
        raise Refusal("all phases require eight full 131072-token resident slots")
    if arg_value(args, "--kv") != "int8" or not arg_value(args, "--mtp"):
        raise Refusal("retained INT8/MTP assets are missing")
    if not arg_value(args, "--control-vector-scaled") or not arg_value(args, "--native"):
        raise Refusal("same-model/control-vector contract is missing")
    return args


def apply_profile_args(args, profile):
    allowed = {"--prefill", "--spec", "--spec-min-p", "--lookup-max-t", "--lookup-sizes", "--draft-fallback",
               "--draft-mtp-min-p", "--draft-mtp-spec", "--draft-rule"}
    result = list(args)
    overrides = profile.get("arg_overrides", {})
    if not isinstance(overrides, dict) or any(k not in allowed for k in overrides):
        raise Refusal("profile changes a protected model/capacity argument or an unsupported tuning flag")
    for key, value in overrides.items():
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise Refusal("profile argument values must be explicit strings/numbers")
        result = replace_arg(result, key, value)
    if any("lsa" in value.lower() for value in result if value.startswith("STRATA_")):
        raise Refusal("research selector settings are not part of this stock-engine campaign")
    return result


def launch_env(cfg, contract, profile=None):
    profile = profile or {}
    env = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_", "CUDA_", "NVIDIA_", "LD_"))}
    public = {}
    for source in (contract.get("environment", {}), cfg.get("env", {})):
        for key, value in source.items():
            if key.startswith("STRATA_") and not key.startswith("STRATA_LSA"):
                env[key] = public[key] = str(value)
    extra = profile.get("env", {})
    if not isinstance(extra, dict) or any(not k.startswith("STRATA_") or k.startswith("STRATA_LSA") for k in extra):
        raise Refusal("profile env must contain explicit stock STRATA settings only")
    for key, value in extra.items():
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise Refusal("profile environment values must be strings/numbers")
        env[key] = public[key] = str(value)
    directories = profile.get("lib_dirs", cfg.get("lib_dirs", []))
    if not isinstance(directories, list) or not directories or any(not isinstance(p, str) for p in directories):
        raise Refusal("profile must supply the matched bundle library directory")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(directories)
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.get("gpu", 0))
    env["CUDA_MODULE_LOADING"] = "EAGER"
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    return env, public


def validate_capacity(engine, phase="full"):
    info = engine.info
    if info.get("kv") != "int8" or int(info.get("context", -1)) != CAPACITY:
        raise Refusal("engine did not retain full context and INT8 KV")
    if int(info.get("batch_slots", -1)) != SLOTS or int(info.get("expert_slots", -1)) != 24576:
        raise Refusal("engine did not retain eight slots and all 24576 resident experts")
    if int(info.get("kv_resident", -1)) != 0 or info.get("cvec") in (None, "none", "off", "0"):
        raise Refusal("full resident KV or selected control vector is missing")
    requested = int(arg_value(engine.args, "--spec") or 0)
    resolved = {key: int(info.get(key, -1)) for key in ("spec", "mtp_max", "lookup")}
    # Lookup can increase verification capacity beyond the MTP proposal cap.
    # Validate the requested cap and ordering, not one historical spec=8 value.
    if resolved["mtp_max"] != requested or resolved["spec"] < max(1, requested) or resolved["lookup"] < 0:
        raise Refusal("resolved MTP/verification policy is incompatible with the requested profile")
    return {"requested_mtp_cap": requested, **resolved}


def retrieval_case(tok, template, name, target_tokens, seed):
    rng = random.Random(seed)
    filler = "\n".join(f"Fictional frame {i:06d}, parcel {rng.randrange(1000000000):09d}, "
                       f"rack {rng.randrange(1000):03d}, checksum {rng.randrange(1000000000):09d}."
                       for i in range(9000))
    ids = tok.encode(filler)
    if len(ids) < target_tokens:
        raise Refusal("generated retrieval corpus is shorter than its declared budget")
    midpoint = target_tokens // 2
    marker = f"CHECK_{seed:08d}_{hashlib.sha256(filler.encode()).hexdigest()[:12].upper()}"
    question = ("Read this fictional ledger and recover its single midpoint marker.\n"
                + tok.decode(ids[:midpoint]) + "\nMidpoint marker: " + marker + "\n"
                + tok.decode(ids[midpoint:target_tokens]) + "\nReturn only the complete midpoint marker.")
    case = encode_case(tok, template, name, question, marker, 64)
    if case["prompt_tokens"] + case["max_new"] > CAPACITY:
        raise Refusal("retrieval fixture exceeds the retained answer allowance")
    case.update(corpus_seed=seed, corpus_sha256=hashlib.sha256(filler.encode()).hexdigest(),
                marker_prompt_offset_tokens=len(tok.encode(case["prompt"][:case["prompt"].index("Midpoint marker: " + marker)], parse_special=True)))
    return case


def performance_cases(tok, template, output_tokens, solo_tokens, seed):
    topics = ["an append-only event journal", "a bounded task scheduler", "a sensor calibration pipeline",
              "an immutable file manifest", "a retry policy", "an in-memory message queue",
              "a cache invalidation policy", "a typed configuration loader"]
    batch = []
    for slot, topic in enumerate(topics):
        common = f"Fictional project {seed}-{slot}: design {topic}. "
        if slot % 2:
            request = (common + "Write complete Python implementations of three cooperating classes, followed by six meaningful unit tests. "
                       "Include type hints, docstrings and comments explaining edge cases. Use no external libraries. "
                       "Provide substantial working code, at least 120 lines, and explain remaining limitations after it.")
        else:
            request = (common + "Write a detailed technical proposal of at least 1200 words covering requirements, data structures, "
                       "failure recovery, concurrency, observability, three worked scenarios and a testing strategy. "
                       "Use complete prose, not a short outline. Continue developing the proposal through every section.")
        case = encode_case(tok, template, f"sustained_slot_{slot}", request, None, output_tokens)
        case["decode_seed"] = seed + slot
        batch.append(case)
    solo = encode_case(tok, template, "solo_prose_with_eight_slot_allocation",
                       f"Fictional review {seed}: write a thorough 1400-word explanation of a durable task queue, covering its "
                       "state machine, leases, retries, duplicate suppression, crashes and six concrete failure scenarios. "
                       "Explain each tradeoff in connected prose and include a substantial Python pseudocode example. "
                       "Do not summarize early; develop every section.", None, solo_tokens)
    return solo, batch


def fixtures(tok, template, phase="full", long_context=False, stage="tuning", seed=202611041,
             output_tokens=256, solo_tokens=384):
    marker = f"STRATA_STOCK_OK_{seed}"
    cases = [encode_case(tok, template, "exact_canary", "Reply with exactly " + marker + ".", marker)]
    if phase == "canary":
        cases.append(retrieval_case(tok, template, "retrieval_4k", 4000, seed + 1))
        return cases, None
    if stage == "heldout":
        cases.extend([retrieval_case(tok, template, "heldout_retrieval_32k", 32000, seed + 31),
                      retrieval_case(tok, template, "heldout_retrieval_125k", 125000, seed + 127)])
    else:
        cases.append(retrieval_case(tok, template, "retrieval_4k", 4000, seed + 1))
        reused = dict(cases[-1]); reused.update(id="accepted_prefix_reuse", require_reused=True)
        cases.append(reused)
        cases.append(retrieval_case(tok, template, "retrieval_16k", 16000, seed + 17))
    rng = random.Random(seed + 1009)
    values = rng.sample(range(2, 23), 6)
    question = "Compute the exact output of this Python program. Return only the decimal integer.\nvalues=" + repr(values) + "\nprint(sum(x*x for x in values))"
    cases.append(encode_case(tok, template, "code_exact", question, str(sum(x*x for x in values))))
    # Preserve the known baseline error as its own failed quality observation.
    cases.append(encode_case(tok, template, "known_code_208", "Compute the exact output. Return only the decimal integer.\n"
                             "values=[2,3,5,7,11]\nprint(sum(x*x for x in values))", "208"))
    expected = {"marker": f"fresh-{seed}", "count": int(seed % 19 + 2), "ready": True}
    case = encode_case(tok, template, "json_structure", "Return one JSON object with exactly these fields: marker equals "
                       + json.dumps(expected["marker"]) + ", count equals " + str(expected["count"])
                       + ", ready equals true. Include no prose or code fence.", json.dumps(expected, separators=(",", ":")), 80)
    case.update(checker="json", expected_json=expected); cases.append(case)
    _, batch = performance_cases(tok, template, output_tokens, solo_tokens, seed)
    return cases, batch


def read_profiles(path, cfg):
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or not isinstance(payload.get("profiles"), list):
        raise Refusal("profiles file must use schema_version 1 and a profiles list")
    control = dict(payload.get("control", {}))
    control.setdefault("name", "official0150_current")
    control["executable"] = str(cfg["exe"])
    if control.get("env") or control.get("arg_overrides"):
        raise Refusal("control must retain the actual production arguments and environment")
    control.setdefault("lib_dirs", cfg.get("lib_dirs", []))
    control["sha256"] = sha(cfg["exe"])
    candidates = {}
    for raw in payload["profiles"]:
        profile = dict(raw); name = profile.get("name", profile.get("id"))
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", name) or name in candidates:
            raise Refusal("profile names must be unique short identifiers")
        profile["name"] = name
        profile["executable"] = profile.get("executable", profile.get("exe"))
        if not isinstance(profile["executable"], str) or not re.fullmatch(r"[0-9a-f]{64}", profile.get("sha256", "")):
            raise Refusal("candidate profiles must pin an executable and its SHA-256")
        if "lib_dirs" not in profile:
            raise Refusal("candidate profile must name its matched bundle libraries")
        candidates[name] = profile
    return control, candidates


def quality_summary(records, expected_ids):
    records = [x for x in records if x.get("expected") is not None]
    observed = {x["id"] for x in records}
    failed = [x["id"] for x in records if x["quality_status"] == "failed"]
    missing = sorted(set(expected_ids) - observed)
    return {"quality_status": "incomplete" if missing else "failed" if failed else "passed",
            "failed_quality_cases": failed, "unexecuted_quality_cases": missing,
            "critical_marker_failures": sorted(set(failed) & CRITICAL_MARKERS)}


def telemetry():
    fields = "uuid,name,driver_version,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.free,utilization.gpu"
    raw = command(["nvidia-smi", "--id=0", "--query-gpu=" + fields, "--format=csv,noheader,nounits"])
    return {"captured_at": utc(), "fields": fields.split(","), "csv": raw}


def run_arm(role, profile, cfg, contract, cases, batch, solo, tok, directory, guard, cli, pair):
    executable = Path(profile["executable"])
    if not executable.is_file() or sha(executable) != profile["sha256"]:
        raise Refusal("engine executable changed or does not match its selected profile")
    args = apply_profile_args(engine_args(cfg, "full"), profile)
    env, public = launch_env(cfg, contract, profile)
    report = {"role": role, "profile": profile["name"], "pair": pair, "started_at": utc(),
              "engine": str(executable), "engine_sha256": sha(executable), "engine_args": args,
              "public_environment": public, "library_directories": profile["lib_dirs"],
              "status": "running", "infrastructure_status": "running", "cases": []}
    engine = None
    try:
        report["gpu_before"] = telemetry()
        with NativeEngine(executable, args, env, cfg.get("cwd"), directory, guard, cli.ready_seconds) as engine:
            report.update(info=engine.info, ready=engine.ready, startup_ms=engine.ready_ms,
                          resolved_drafting=validate_capacity(engine, "full"))
            for case in cases:
                result = engine.generate(case, tok, cli.case_seconds)
                report["cases"].append(result)
                print(json.dumps({"event": "quality", "pair": pair, "role": role, "profile": profile["name"],
                                  "case": case["id"], "status": result["status"], "wall_ms": result["wall_ms"]}), flush=True)
                if not result["prefix_reuse_gate_passed"]:
                    raise Refusal("required native prefix reuse was not observed")
            if solo is not None:
                report["solo"] = engine.generate(solo, tok, cli.case_seconds)
                report["solo"]["eligible_for_speed_comparison"] = (report["solo"]["native"]["generated"] == solo["max_new"]
                    and report["solo"]["native"]["generated"] >= 192
                    and report["solo"]["output_metrics"]["common_window"]["valid"])
            if batch:
                report["sustained"] = engine.throughput(batch, tok, cli.case_seconds, cli.trim_seconds)
                print(json.dumps({"event": "sustained", "pair": pair, "role": role, "profile": profile["name"],
                                  "eligible": report["sustained"]["eligible_for_speed_comparison"],
                                  "metrics": report["sustained"]["metrics"]}), flush=True)
            report["gpu_after_work"] = telemetry()
            report["infrastructure_status"] = "passed"
            report.update(quality_summary(report["cases"], [c["id"] for c in cases]))
            report["status"] = "completed_with_quality_failures" if report["quality_status"] == "failed" else "passed"
    except (Refusal, OSError, ValueError, subprocess.SubprocessError) as error:
        report.update(status="failed", infrastructure_status="failed", error=str(error))
    finally:
        if engine is not None:
            report["owned_process_group_empty"] = getattr(engine, "owned_process_group_empty", engine.proc.poll() is not None)
        report.update(quality_summary(report["cases"], [c["id"] for c in cases]))
        report["finished_at"] = utc()
        directory.mkdir(exist_ok=True)
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def comparisons(arms):
    pairs = []
    for pair in sorted({a["pair"] for a in arms}):
        selected = {a["role"]: a for a in arms if a["pair"] == pair}
        if set(selected) != {"control", "candidate"}:
            continue
        control, candidate = selected["control"], selected["candidate"]
        old = {x["id"]: x for x in control["cases"]}
        quality = []
        for new in candidate["cases"]:
            before = old.get(new["id"])
            if before is None: continue
            quality.append({"id": new["id"], "control_correct": before["quality_status"] == "passed",
                            "candidate_correct": new["quality_status"] == "passed",
                            "output_ids_equal": before["output_ids"] == new["output_ids"],
                            "regression": before["quality_status"] == "passed" and new["quality_status"] != "passed",
                            "control_wall_ms": before["wall_ms"], "candidate_wall_ms": new["wall_ms"]})
        row = {"pair": pair, "quality": quality}
        for section in ("solo", "sustained"):
            a, b = control.get(section), candidate.get(section)
            if a is None or b is None: continue
            key = "output_metrics" if section == "solo" else "metrics"
            am, bm = a[key], b[key]
            row[section] = {"eligible": a.get("eligible_for_speed_comparison", False) and b.get("eligible_for_speed_comparison", False),
                            "control_common_output_tps": am["common_window"].get("aggregate_output_tokens_per_second"),
                            "candidate_common_output_tps": bm["common_window"].get("aggregate_output_tokens_per_second"),
                            "control_whole_request_output_tps": am.get("whole_request_output_tokens_per_second"),
                            "candidate_whole_request_output_tps": bm.get("whole_request_output_tokens_per_second")}
        pairs.append(row)
    summary = {"pairs": pairs, "quality_regressions": sorted({q["id"] for p in pairs for q in p["quality"] if q["regression"]}),
               "scope": "paired observed stream accounting; statistical confidence not implied by three pairs"}
    for section in ("solo", "sustained"):
        rows = [p[section] for p in pairs if section in p and p[section]["eligible"]]
        if not rows: continue
        summary[section] = {"eligible_pairs": len(rows),
                            "control_common_median_output_tps": statistics.median(x["control_common_output_tps"] for x in rows),
                            "candidate_common_median_output_tps": statistics.median(x["candidate_common_output_tps"] for x in rows),
                            "paired_common_speed_ratios": [x["candidate_common_output_tps"] / x["control_common_output_tps"] for x in rows],
                            "control_whole_request_median_output_tps": statistics.median(x["control_whole_request_output_tps"] for x in rows),
                            "candidate_whole_request_median_output_tps": statistics.median(x["candidate_whole_request_output_tps"] for x in rows)}
    return summary


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_mutually_exclusive_group(); modes.add_argument("--plan", action="store_true"); modes.add_argument("--execute", action="store_true")
    ap.add_argument("--profiles", type=Path, default=ROOT / "profiles.json")
    ap.add_argument("--profile", default="stock0156_same_settings")
    ap.add_argument("--phase", choices=("canary", "full"), default="full")
    ap.add_argument("--stage", choices=("tuning", "heldout"), default="tuning")
    ap.add_argument("--quality-seed", type=int)
    ap.add_argument("--arm", choices=("control", "candidate", "both"), default="both")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--output-tokens", type=int, choices=(256, 512), default=256)
    ap.add_argument("--solo-tokens", type=int, default=384)
    ap.add_argument("--trim-seconds", type=float, default=0.25)
    ap.add_argument("--owner", default="")
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--ready-seconds", type=int, default=600)
    ap.add_argument("--case-seconds", type=int, default=1200)
    ap.add_argument("--total-seconds", type=int, default=7200)
    return ap


def main():
    cli = parser().parse_args()
    if not 1 <= cli.pairs <= 6 or not 192 <= cli.solo_tokens <= 1024 or not 0 <= cli.trim_seconds <= 1:
        raise Refusal("pair count, solo token budget or common-window trim is outside the bounded campaign")
    if not 30 <= cli.ready_seconds <= 1800 or not 30 <= cli.case_seconds <= 3600 or not 60 <= cli.total_seconds <= 21600:
        raise Refusal("qualification timeout is outside the bounded campaign")
    cfg, contract, config_sha, contract_sha = current_config()
    control, profiles = read_profiles(cli.profiles, cfg)
    if cli.profile not in profiles: raise Refusal("selected profile is absent")
    candidate = profiles[cli.profile]
    expected_control = contract.get("hashes", {}).get(str(cfg["exe"]))
    if expected_control and expected_control != control["sha256"]:
        raise Refusal("actual production executable differs from the pinned recovery contract")
    tok, template, token_hashes = load_tokenizer(cfg)
    seed = cli.quality_seed if cli.quality_seed is not None else (202611041 if cli.stage == "tuning" else 202611391)
    cases, batch = fixtures(tok, template, cli.phase, stage=cli.stage, seed=seed, output_tokens=cli.output_tokens, solo_tokens=cli.solo_tokens)
    solo, _ = performance_cases(tok, template, cli.output_tokens, cli.solo_tokens, seed)
    if cli.phase == "canary": solo = None
    schedule = []
    for pair in range(1, cli.pairs + 1):
        order = ["control", "candidate"] if pair % 2 else ["candidate", "control"]
        for role in order:
            if cli.arm != "both" and role != cli.arm: continue
            schedule.append({"pair": pair, "role": role, "profile": control if role == "control" else candidate})
    plan = {"schema_version": 1, "phase": cli.phase, "stage": cli.stage, "quality_seed": seed,
            "context": CAPACITY, "slots": SLOTS, "gpu_work_started": False,
            "production_config_sha256": config_sha, "production_contract_sha256": contract_sha,
            "profiles_sha256": sha(cli.profiles), "tokenizer_sha256": token_hashes,
            "schedule": [{"pair": x["pair"], "role": x["role"], "profile": x["profile"]["name"],
                          "executable": x["profile"]["executable"], "expected_sha256": x["profile"]["sha256"],
                          "args": apply_profile_args(engine_args(cfg, "full"), x["profile"]),
                          "env": launch_env(cfg, contract, x["profile"])[1], "lib_dirs": x["profile"]["lib_dirs"]} for x in schedule],
            "quality_cases": [{k: x[k] for k in ("id", "expected", "max_new", "prompt_tokens", "token_sha256")} for x in cases],
            "quality_separation": "tuning and heldout use distinct corpus seeds; further tuning after heldout consumption requires a new heldout seed",
            "performance": {"streams": 8 if batch else 0, "requested_output_tokens_each": cli.output_tokens,
                            "solo_output_tokens": cli.solo_tokens, "common_window_trim_seconds": cli.trim_seconds,
                            "order": "AB, BA, AB for three paired rounds", "cancel_after_three_tokens": False},
            "limitations": ["finite synthetic quality fixtures", "native CLI only; API/vision qualification is separate",
                            "no mixed-load 32k joiner gate; latency_chunks profile is diagnostic until independently qualified",
                            "no service, recovery, power-policy or production promotion changes"]}
    if not cli.execute:
        print(json.dumps(plan, indent=2)); return 0
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,100}", cli.owner): raise Refusal("execution requires exact owned maintenance owner")
    directory = cli.run_dir or ROOT / "receipts" / (cli.stage + "-" + cli.profile + "-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "fixtures.json").write_text(json.dumps({"cases": cases, "batch": batch, "solo": solo}, indent=2) + "\n")
    (directory / "profiles.frozen.json").write_text(json.dumps({"control": control, "candidate": candidate}, indent=2) + "\n")
    report = {**plan, "started_at": utc(), "owner": cli.owner, "status": "running", "arms": [], "gpu_work_started": True}
    guard = Guard(cli.owner, config_sha, contract_sha, cli.total_seconds)
    with (ROOT / ".benchmark-engine.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            report.update(status="refused", gpu_work_started=False, error="another native benchmark owns the shared model window")
            (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            raise Refusal(report["error"]) from error
        try:
            guard.check(force=True, before_start=True)
            for item in schedule:
                arm_dir = directory / f"pair-{item['pair']:02d}-{item['role']}"
                record = run_arm(item["role"], item["profile"], cfg, contract, cases, batch, solo, tok, arm_dir, guard, cli, item["pair"])
                report["arms"].append(record)
                (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                if record["infrastructure_status"] != "passed":
                    raise Refusal("native arm failed infrastructure; stopping the sequential campaign")
            guard.check(force=True)
            report["comparison"] = comparisons(report["arms"])
            report["status"] = "completed_with_quality_failures" if any(x["quality_status"] == "failed" for x in report["arms"]) else "passed"
            report["infrastructure_status"] = "passed"
        except BaseException as error:
            report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            raise
        finally:
            report["finished_at"] = utc()
            (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(directory / "report.json")}), flush=True)
    # Baseline quality failures remain explicit but do not erase a completed
    # infrastructure/performance campaign or prevent paired regression analysis.
    return 0 if report["infrastructure_status"] == "passed" else 1


if __name__ == "__main__":
    def interrupted(signum, _frame): raise KeyboardInterrupt(f"benchmark interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGHUP, interrupted)
    try: raise SystemExit(main())
    except (Refusal, OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "refused", "error": str(error)}), file=sys.stderr)
        raise SystemExit(2)

