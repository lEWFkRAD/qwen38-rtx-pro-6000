#!/usr/bin/env python3
"""Bounded, isolated native Strata engine qualification. Default: plan only.

No HTTP listener, fleet routing, service changes or recovery mutations. Execution
requires the caller's exact owned production HOLD throughout. Each native engine
runs sequentially in its own process group. Only that group is terminated.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "strata-src"
CONFIG = Path("/etc/onyx-inference/strata-server.json")
CONTRACT = Path("/etc/onyx-inference/strata.json")
MAINTENANCE = "/usr/local/libexec/onyx-inference-recovery/strata-maintenance.py"
PRIMARY = "qwen38-flash-next-strata.service"
CONTROL = Path("/path/to/lsa-prep-20261103/flashnext-maps/build/strata")
MAPS = ROOT / "receipts/flashnext-learned-native-v1.bin"
PUBLIC_ENV = {
    "STRATA_ADMIT_ROTATE", "STRATA_ADMIT_ROTATE_QUANTUM", "STRATA_ARENA_MMAP", "STRATA_SLOT_QUEUE",
    "STRATA_VERIFY_ALL_RESIDENT", "STRATA_BATCH_DECODE_SHARE", "STRATA_BATCH_SELF_COMMIT",
    "STRATA_OWNED_PRICE", "STRATA_PF_FUSED", "STRATA_QSA_SLOTS", "STRATA_SLOT_COPY_BATCHSYNC",
    "STRATA_SLOT_COPY_DIRECT",
}
CRITICAL_MARKERS = {"exact_canary", "boundary_retrieval", "accepted_prefix_reuse", "independent_long_retrieval", "near_capacity_retrieval"}


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


def engine_args(cfg, phase):
    args = list(cfg["args"])
    if cfg.get("user_args") or cfg.get("drafting") is not None or cfg.get("mtp") is False:
        raise Refusal("configuration needs wrapper argument shaping; update this harness before testing")
    # The current production wrapper adds --batch from parallel. Do not silently
    # fall back to fewer slots when a test engine cannot fit them.
    if phase == "full":
        if cfg.get("parallel") != 8:
            raise Refusal("production no longer requests eight slots")
        args = replace_arg(args, "--batch", 8)
        if arg_value(args, "--max-context") != "131072" or arg_value(args, "--kv-resident") != "131072":
            raise Refusal("production full-context/resident INT8 contract changed")
    else:
        args = replace_arg(args, "--max-context", 8192)
        args = replace_arg(args, "--prefill", 8192)
        args = replace_arg(args, "--kv-resident", 8192)
        # No --batch means the native engine's single main session. An explicit
        # --batch 1 is rejected by some older engines, so omit both spellings.
        for flag in ("--batch", "--slots"):
            if flag in args:
                index = args.index(flag)
                del args[index:index + 2]
    if arg_value(args, "--kv") != "int8" or not arg_value(args, "--mtp") or arg_value(args, "--spec") != "6":
        raise Refusal("INT8/MTP/speculation contract changed")
    if not arg_value(args, "--control-vector-scaled") or not arg_value(args, "--native"):
        raise Refusal("same-model/control-vector contract is missing")
    return args


def launch_env(cfg, contract, arm, maps):
    # Preserve ordinary OS settings, exclude inherited experiment and dynamic
    # linker overrides, then restore the pinned production engine settings.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_", "CUDA_", "NVIDIA_", "LD_"))}
    public = {}
    for source in (contract.get("environment", {}), cfg.get("env", {})):
        for key, value in source.items():
            if key in PUBLIC_ENV:
                env[key] = public[key] = str(value)
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.get("gpu", 0))
    env["CUDA_MODULE_LOADING"] = "EAGER"
    env["LD_LIBRARY_PATH"] = os.pathsep.join(["/usr/local/cuda-13.0/lib64", *map(str, cfg.get("lib_dirs", []))])
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if arm == "candidate":
        env["STRATA_LSA_MAPS"] = str(maps)
        public["STRATA_LSA_MAPS"] = str(maps)
    return env, public


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


def fixtures(tok, template, phase, long_context=False):
    cases = [encode_case(tok, template, "exact_canary", "Reply with exactly STRATA_LSA_CANARY_OK.",
                         "STRATA_LSA_CANARY_OK")]
    records = "\n".join(f"Entry {i:04d}: parcel-{(i * 7919 + 104729) % 1000003:06d} stored in bay {(i * 37) % 997:03d}."
                        for i in range(64))
    boundary = "Verification code: AMBER_7319.\n" + records + "\nReturn only the verification code from the first line."
    cases.append(encode_case(tok, template, "boundary_retrieval", boundary, "AMBER_7319"))
    if cases[-1]["prompt_tokens"] <= 512:
        raise Refusal("boundary fixture does not exercise a closed LSA block")
    if phase == "canary":
        return cases, None
    # Keep this immediately after its seed request: the production policy has
    # no parked conversation cache, so arbitrary old conversations need not be
    # retained after unrelated requests.
    reused = dict(cases[1])
    reused["id"] = "accepted_prefix_reuse"
    reused["require_reused"] = True
    cases.append(reused)
    long_records = "\n".join(f"Record R{i:05d} has tag N{(i * 15485863 + 32452843) % 100000007:08d}; owner is fictional team {(i * 17) % 101:03d}."
                             for i in range(256))
    target = (173 * 15485863 + 32452843) % 100000007
    question = long_records + "\nReturn only the tag for Record R00173."
    cases.append(encode_case(tok, template, "independent_long_retrieval", question, f"N{target:08d}"))
    cases.append(encode_case(tok, template, "code_result", "Compute the exact output of this Python code. Return only the decimal integer.\n"
                             "values = [2, 3, 5, 7, 11]\nprint(sum(x*x for x in values))", "208"))
    csv = ",".join(str(i) for i in range(1, 33))
    cases.append(encode_case(tok, template, "speculative_copy", "Copy this CSV exactly with no spaces, prefix, suffix, or code fence:\n" + csv,
                             csv, 160))
    if long_context:
        # A new independent synthetic corpus. Slice plain ASCII filler tokens,
        # then render the complete chat again: the actual final token count is
        # measured, including the middle marker and generation header.
        filler = "\n".join(
            f"Frame {i:06d} holds synthetic parcel {(i * 49979687 + 67867967) % 1000000007:09d} "
            f"at rack {(i * 31 + 17) % 997:03d}; calibration note {(i * 86028121 + 104395303) % 1000000007:09d}."
            for i in range(6000))
        filler_ids = tok.encode(filler)
        if len(filler_ids) < 125000:
            raise Refusal("near-capacity fixture corpus is too short")
        left = tok.decode(filler_ids[:62500])
        right = tok.decode(filler_ids[62500:125000])
        marker = "BASALT_582610_OMEGA"
        question = ("Read this independent synthetic calibration ledger. It contains one midpoint verification marker.\n"
                    + left + "\nMidpoint verification marker: " + marker + "\n" + right
                    + "\nReturn only the complete midpoint verification marker value. Do not explain.")
        near = encode_case(tok, template, "near_capacity_retrieval", question, marker, 64)
        if not 124000 <= near["prompt_tokens"] < 130000 or near["prompt_tokens"] + near["max_new"] > 131072:
            raise Refusal("near-capacity fixture does not leave its declared answer allowance")
        marker_at = near["prompt"].index("Midpoint verification marker: " + marker)
        near["marker_prompt_offset_tokens"] = len(tok.encode(near["prompt"][:marker_at], parse_special=True))
        near["corpus_sha256"] = hashlib.sha256(filler.encode()).hexdigest()
        cases.append(near)
    stream_question = records + "\nSlot identifier: {slot}. Output the integers from 1 through 400 as comma-separated decimal integers, " \
        "starting immediately with 1. Keep going until 400. Do not explain or use a code fence."
    batch = [encode_case(tok, template, f"batch_slot_{slot}", stream_question.replace("{slot}", str(slot)), None, 1024)
             for slot in range(8)]
    return cases, batch


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
                name = parts[1].lower()
                if any(term in name for term in ("strata", "sglang", "vllm", "tensorfold")):
                    raise Refusal("another model process already owns the GPU")
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
        ids, first = [], None
        while True:
            when, line = self.next(born + seconds)
            if line.startswith("T "):
                first = first or when
                ids.append(int(line.split()[1]))
            elif line.startswith("DONE "):
                done = done_fields(line)
                if done["generated"] != len(ids):
                    raise Refusal("native token count disagrees with DONE")
                text = scrub_output(tok, ids)
                correct = text == case["expected"]
                reused = done["reused_tokens"] > 0
                reuse_gate = not case.get("require_reused") or reused
                return {"id": case["id"], "status": "passed" if correct and reuse_gate else "failed",
                        "quality_status": "passed" if correct else "failed",
                        "infrastructure_status": "passed" if reuse_gate else "failed",
                        "expected": case["expected"], "actual": text, "output_ids": ids,
                        "prompt_tokens": case["prompt_tokens"], "token_sha256": case["token_sha256"],
                        "ttft_ms": None if first is None else (first - born) * 1000,
                        "wall_ms": (when - born) * 1000, "prefix_reuse_verified": reused,
                        "prefix_reuse_gate_passed": reuse_gate, "native": done}

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


def validate_capacity(engine, phase):
    info = engine.info
    # generate.cpp's normal prompt-lookup policy keeps the requested six-token
    # MTP window and raises the verifier capacity by two (capped at eight).
    # `lookup` reports the default three-token suffix-match threshold. Comparing
    # INFO spec directly to the CLI --spec value rejects an unchanged baseline.
    resolved_drafting = {key: int(info.get(key, -1)) for key in ("spec", "mtp_max", "lookup")}
    if info.get("kv") != "int8" or resolved_drafting != {"spec": 8, "mtp_max": 6, "lookup": 3}:
        raise Refusal("engine did not retain resident INT8 and the resolved MTP6/verifier8/suffix3 policy")
    if int(info.get("context", -1)) != (131072 if phase == "full" else 8192):
        raise Refusal("engine context differs from the requested qualification phase")
    if info.get("cvec") in (None, "none", "off", "0"):
        raise Refusal("engine does not report an enabled control vector")
    if phase == "full":
        if int(info.get("batch_slots", -1)) != 8:
            raise Refusal("full qualification requires all eight serving slots")
        if int(info.get("expert_slots", -1)) != 24576:
            raise Refusal("full qualification requires all 24576 experts GPU resident")
        if int(info.get("kv_resident", -1)) != 0:
            raise Refusal("full qualification requires fully resident main INT8 KV")


def quality_summary(records, expected_ids):
    quality = [case for case in records if "expected" in case]
    failed = [case["id"] for case in quality if case["actual"] != case["expected"]]
    observed = {case["id"] for case in quality}
    missing = sorted(set(expected_ids) - observed)
    critical = set(expected_ids) & CRITICAL_MARKERS
    critical_failed = sorted(set(failed) & critical)
    critical_missing = sorted(critical - observed)
    return {"quality_status": "incomplete" if missing else "failed" if failed else "passed",
            "failed_quality_cases": failed, "unexecuted_quality_cases": missing,
            "critical_markers_status": "incomplete" if critical_missing else "failed" if critical_failed else "passed" if critical else "not_run",
            "failed_critical_markers": critical_failed, "unexecuted_critical_markers": critical_missing}


def compare_arms(control, candidate):
    baseline = {case["id"]: case for case in control["cases"]}
    comparison = []
    for case in candidate["cases"]:
        previous = baseline.get(case["id"], {})
        if "wall_ms" not in case or "wall_ms" not in previous:
            continue
        old_correct = previous["actual"] == previous["expected"]
        new_correct = case["actual"] == case["expected"]
        comparison.append({"id": case["id"], "control_wall_ms": previous["wall_ms"],
                           "candidate_wall_ms": case["wall_ms"],
                           "candidate_over_control": case["wall_ms"] / max(previous["wall_ms"], 1e-9),
                           "exact_output_match": case["output_ids"] == previous["output_ids"],
                           "control_correct": old_correct, "candidate_correct": new_correct,
                           "quality_regression": old_correct and not new_correct,
                           "quality_improvement": not old_correct and new_correct,
                           "critical_marker": case["id"] in CRITICAL_MARKERS})
    regressions = [case["id"] for case in comparison if case["quality_regression"]]
    compared = {case["id"] for case in comparison}
    missing = sorted(case["id"] for case in control["cases"] if "expected" in case and case["id"] not in compared)
    return {"comparison": comparison, "candidate_quality_regressions": regressions,
            "candidate_uncompared_quality_cases": missing,
            "candidate_no_quality_regression": not regressions and not missing,
            "candidate_critical_markers_status": candidate.get("critical_markers_status", "incomplete"),
            "performance_scope": "one sequential pass per arm; no statistical speed qualification"}


def run_arm(name, executable, cfg, contract, args, cases, batch, tok, directory, guard, cli):
    env, public_env = launch_env(cfg, contract, name, cli.maps)
    report = {"arm": name, "engine": str(executable), "engine_sha256": sha(executable), "args": args,
              "public_environment": public_env, "started_at": utc(), "status": "running", "infrastructure_status": "running", "cases": [],
              "map_sha256": sha(cli.maps) if name == "candidate" else None}
    try:
        with NativeEngine(executable, args, env, cfg.get("cwd"), directory, guard, cli.ready_seconds) as engine:
            report.update(info=engine.info, ready=engine.ready, startup_ms=engine.ready_ms, pid=engine.proc.pid)
            validate_capacity(engine, cli.phase)
            startup_log = (directory / "engine.stderr.log").read_text(encoding="utf-8", errors="replace")
            enabled_marker = "strata LSA: native learned-map selector enabled;"
            report["native_lsa_startup_verified"] = enabled_marker in startup_log
            if (name == "candidate") != report["native_lsa_startup_verified"]:
                raise Refusal("native LSA startup state does not match the selected arm")
            for case in cases:
                result = engine.generate(case, tok, cli.case_seconds)
                report["cases"].append(result)
                print(json.dumps({"event": "case", "arm": name, "case": case["id"], "status": result["status"],
                                  "prompt_tokens": case["prompt_tokens"], "wall_ms": result["wall_ms"]}), flush=True)
                # Preserve a model's wrong answer and finish the remaining
                # quality/capacity cases. A protocol/native error still raises
                # inside generate; required prefix reuse is structural too.
                if not result["prefix_reuse_gate_passed"]:
                    raise Refusal("required accepted-prefix reuse was not observed: " + case["id"])
            if batch:
                result = engine.batch(batch, tok, cli.case_seconds)
                report["cases"].append(result)
                if result["status"] != "passed":
                    raise Refusal("eight-slot decode/cancellation did not meet its gate")
            offered = sum(c.get("native", {}).get("drafts_offered", 0) for c in report["cases"])
            accepted = sum(c.get("native", {}).get("drafts_accepted", 0) for c in report["cases"])
            report["speculative_evidence"] = {"offered": offered, "accepted": accepted, "rejected": offered - accepted}
            if cli.phase == "full" and not cli.long_context_only and (offered == 0 or offered - accepted == 0):
                raise Refusal("full qualification did not observe a rejected MTP proposal")
            report["infrastructure_status"] = "passed"
            report.update(quality_summary(report["cases"], [case["id"] for case in cases]))
            report["status"] = "completed_with_quality_failures" if report["quality_status"] == "failed" else "passed"
    except (Refusal, OSError, ValueError, subprocess.SubprocessError) as exc:
        report.update(status="failed", infrastructure_status="failed", error=str(exc))
    except BaseException as exc:
        report.update(status="interrupted", infrastructure_status="interrupted", error=type(exc).__name__)
        raise
    finally:
        report.update(quality_summary(report["cases"], [case["id"] for case in cases]))
        report["finished_at"] = utc()
        # The directory may not exist when preflight refused before a launch.
        directory.mkdir(exist_ok=True)
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--execute", action="store_true")
    ap.add_argument("--phase", choices=("canary", "full"), default="canary")
    ap.add_argument("--long-context", action="store_true", help="Append the independent approximately 125000-token retrieval case to the full suite")
    ap.add_argument("--long-context-only", action="store_true", help="Run only the near-capacity retrieval case with full production capacity; does not claim the complete suite")
    ap.add_argument("--arm", choices=("candidate", "control", "both"), default="both")
    ap.add_argument("--owner", default="")
    ap.add_argument("--candidate", type=Path, default=ROOT / "build-engine/strata")
    ap.add_argument("--control", type=Path, default=CONTROL)
    ap.add_argument("--maps", type=Path, default=MAPS)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--ready-seconds", type=int, default=600)
    ap.add_argument("--case-seconds", type=int, default=1200)
    ap.add_argument("--total-seconds", type=int, default=3600)
    return ap


def main():
    cli = parser().parse_args()
    if (cli.long_context or cli.long_context_only) and cli.phase != "full":
        raise Refusal("near-capacity retrieval requires the full production capacity phase")
    if not 30 <= cli.ready_seconds <= 1800 or not 30 <= cli.case_seconds <= 3600 or not 60 <= cli.total_seconds <= 7200:
        raise Refusal("timeouts are outside the bounded qualification limits")
    cfg, contract, config_sha, contract_sha = current_config()
    args = engine_args(cfg, cli.phase)
    tok, template, token_hashes = load_tokenizer(cfg)
    cases, batch = fixtures(tok, template, cli.phase, cli.long_context or cli.long_context_only)
    if cli.long_context_only:
        cases = [case for case in cases if case["id"] == "near_capacity_retrieval"]
        batch = None
    plan = {"phase": cli.phase, "arm": cli.arm, "production_config_sha256": config_sha,
            "production_contract_sha256": contract_sha, "engine_args": args,
            "resolved_drafting": {"requested_mtp_window": 6, "verify_window": 8, "suffix_match_threshold": 3},
            "context": 131072 if cli.phase == "full" else 8192, "slots": 8 if cli.phase == "full" else 1,
            "near_capacity_included": cli.long_context or cli.long_context_only,
            "qualification_subset": cli.long_context_only,
            "tokenizer_sha256": token_hashes, "gpu_work_started": False,
            "scope": "native integration canary only" if cli.phase == "canary" else "same-model deterministic quality, reuse and eight-slot integration gates",
            "cases": [{k: c[k] for k in ("id", "expected", "max_new", "prompt_tokens", "token_sha256")} for c in cases],
            "batch_cases": 0 if not batch else 8,
            "limitations": ["finite synthetic fixtures do not prove general task quality", "no vision encoder/image-request validation",
                            "131072 capacity does not establish a 131072-token prompt latency measurement"]}
    if cli.long_context_only:
        plan["scope"] = "near-capacity synthetic retrieval only, using eight-slot 131072-token production allocation"
    for summary, case in zip(plan["cases"], cases):
        for key in ("marker_prompt_offset_tokens", "corpus_sha256"):
            if key in case:
                summary[key] = case[key]
    if not cli.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,100}", cli.owner):
        raise Refusal("execution requires the exact maintenance owner")
    selected = [("control", cli.control), ("candidate", cli.candidate)] if cli.arm == "both" else [(cli.arm, getattr(cli, cli.arm))]
    for name, executable in selected:
        if not executable.is_file():
            raise Refusal(f"{name} engine file is missing")
    guard = Guard(cli.owner, config_sha, contract_sha, cli.total_seconds)
    guard.check(force=True, before_start=True)
    directory = cli.run_dir or ROOT / "receipts" / ("integrated-" + cli.phase + "-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    directory.mkdir(parents=True, exist_ok=False)
    report = {**plan, "started_at": utc(), "owner": cli.owner, "status": "running", "arms": [], "gpu_work_started": True}
    # Preserve exact new synthetic inputs and tokenization for reproduction.
    (directory / "fixtures.json").write_text(json.dumps({"cases": cases, "batch": batch}, indent=2) + "\n")
    with (ROOT / ".integrated-engine.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            report.update(status="refused", error="another integrated engine test holds the research lock")
            (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            raise Refusal(report["error"]) from exc
        try:
            for name, executable in selected:
                arm = run_arm(name, executable, cfg, contract, args, cases, batch, tok, directory / name, guard, cli)
                report["arms"].append(arm)
                if arm["infrastructure_status"] != "passed":
                    report["status"] = "failed"
                    break
            else:
                report["status"] = "completed_with_quality_failures" if any(arm["quality_status"] == "failed" for arm in report["arms"]) else "passed"
            report["infrastructure_status"] = "passed" if len(report["arms"]) == len(selected) and all(arm["infrastructure_status"] == "passed" for arm in report["arms"]) else "failed"
            report["quality_status"] = "incomplete" if any(arm["quality_status"] == "incomplete" for arm in report["arms"]) else "failed" if any(arm["quality_status"] == "failed" for arm in report["arms"]) else "passed"
            guard.check(force=True)
            if len(report["arms"]) == 2:
                report.update(compare_arms(report["arms"][0], report["arms"][1]))
        except BaseException as exc:
            report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            raise
        finally:
            report["finished_at"] = utc()
            (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(directory / "report.json")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"qualification interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        raise SystemExit(main())
    except (Refusal, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
