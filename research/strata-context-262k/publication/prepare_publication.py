#!/usr/bin/env python3
"""Explicit, local-only context-test publication staging; never commits/pushes.

Plan/self-test never copy receipts. Execute only after the root supplies a
final production-readiness/publication attestation. Resident production is OK.
"""
from __future__ import annotations
import argparse
import ast
import csv
import datetime as dt
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = Path("research/strata-context-262k")
GUARDS = {
    "preflight_guard-running-77b810ea.py": "77b810ea2d830dfe7cb76871d24a824b0b13a78082eadbc68786b8d9a3e1863a",
    "preflight_guard.py": "3011d627633691e0567507d7fe497d929d30a6a0050eb6f4f391cecc9b85bf84",
}
MEASURED_HELPERS = {
    "api_deep_concurrency-r3.py": "dca42e9ad983f86beed8733e3a93e885fdc750354cc7b441e1dd3c240fa3466e",
    "api_deep_concurrency-r4.py": "9bbe42c004988a3dd4105cfe5e177e7b117ea84029d96aebe89e0dcf1f77bca8",
}
PRIVATE_PINS = {
    "<private-configuration-sha256>": "<production-config-sha256>",
    "<private-configuration-sha256>": "<production-contract-sha256>",
}
OWNER = re.compile(r"\bcodex-context262k-(?:promote-)?\d{8}-[A-Za-z0-9._:-]+\b")
PROCESS_ID = re.compile(r"(?i)\b(pid|ppid|pgid|sid|start_ticks)\s*[=:]\s*\d+")
INFO_KEYS = {"engine", "version", "context", "max_context", "kv", "kv_resident", "expert_slots", "expert_slots_primary",
             "batch_slots", "batch_groups", "prefill_cap", "prefill_pipe", "spec", "mtp_max", "lookup", "cvec", "spec_min_p",
             "arena_mib", "expert_cache_mib", "expert_cache_primary_mib", "vram_free_mib", "pool_workers", "pcie_frac",
             "conversation_cache_mib", "images", "slot_cache", "logit_bias"}
SIMPLE_KEYS = {"scope", "limits", "status", "started_at", "finished_at", "recorded_at", "context", "slots", "prefill",
               "engine_sha256", "selected_profile_sha256", "fixture_sha256", "tokenizer_sha256", "preflight_guard_sha256",
               "matching_native_report_sha256", "quality_status", "infrastructure_status", "owned_process_group_empty",
               "private_credential_file_removed", "private_credential_removed", "GPU_work_started", "gpu_work_started",
               "startup_ms", "wall_ms", "id", "expected", "actual", "output_ids", "prompt_tokens", "token_sha256",
               "http_status", "prefix_reuse_verified", "prefix_reuse_gate_passed", "ttft_ms", "output_arrival_ms",
               "generated", "finish", "eligible_for_speed_comparison", "all_distinct_slots_mostly_new_and_completed",
               "small_shared_prefix_allowance_tokens", "peak_live_slots", "peak_scope", "normal_gate_preserved",
               "readiness_override_used", "production_gate_unchanged", "fixture_token_sha256", "native_fixture_token_sha256",
               "requested_streams", "requested_max_output_tokens_each", "system_instruction_replaced", "long_prose_directive_appended",
               "configured_EOS_logit_bias", "quality_scope", "minimum_output_scope", "peak_live_running", "peak_active_slots",
               "peak_decoding_slots", "eight_concurrent_deep_streams_observed", "identity_preserved", "closed_our_SSE_handles",
               "our_requests_drained", "postcancel_idle_held_tokens_each", "retention_scope", "usage_note", "source_token_sha256",
               "load_token_sha256", "content_truncated", "headers_received", "finished", "usage", "finish_reason",
               "visible_retokenized_tokens", "content_prefix", "helper_source_sha256", "EOS_suppression_used",
               "harness_correction", "workload", "independent_retrieval_JSON_verified_each", "repeat_cached_tokens",
               "services_changed"}
NATIVE_KEYS = {"generated", "prompt_tokens", "prompt_ms", "decode_ms", "finish", "drafts_accepted", "drafts_offered",
               "reused_tokens", "read_tokens", "raw"}


class Refusal(RuntimeError):
    pass


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_normalizer(path):
    if path is None:
        options = [ROOT.parent / name / "publication/stage_publication.py" for name in
                   ["strata-0156-tuning-20261103", "strata-0156-tuning"]]
        path = next((p for p in options if p.is_file()), None)
    if path is None or not path.is_file():
        raise Refusal("supply the prior stock package's publication normalizer")
    spec = importlib.util.spec_from_file_location("stock_publication_normalizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def select(source):
    """Select only named paths/globs from the reviewed manifest; never walk home."""
    result = {}
    for entry in source["files"]:
        pattern = entry["source"]
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise Refusal("selection paths must be project-relative")
        paths = sorted(ROOT.glob(pattern))
        if not paths and entry.get("required"):
            raise Refusal("required selected artifact is missing: " + pattern)
        for path in paths:
            if not path.is_file() or path.is_symlink():
                raise Refusal("selected artifact is not a regular file")
            relative = path.relative_to(ROOT).as_posix()
            if path.name in {"initial-state.json", "candidate-private.json", "capture_initial_state.py"} or any(
                    word in relative.lower() for word in ["honcho", "wiki", "__pycache__", ".stages"]):
                raise Refusal("private/operational artifact entered the publication selection")
            result[relative] = {**entry, "path": path}
    return result


def response_projection(value):
    out = {k: value[k] for k in ["usage", "timings"] if k in value}
    choices = []
    for choice in value.get("choices", []):
        item = {k: choice[k] for k in ["index", "finish_reason"] if k in choice}
        message = choice.get("message", {})
        item["message"] = {k: message[k] for k in ["role", "content", "reasoning_content"] if k in message}
        if "tool_calls" in message:
            item["message"]["tool_calls"] = [{"type": call.get("type"),
                "function": {k: call.get("function", {})[k] for k in ["name", "arguments"] if k in call.get("function", {})}}
                for call in message["tool_calls"]]
        choices.append(item)
    if choices:
        out["choices"] = choices
    return out


def telemetry_projection(value):
    out = {k: value[k] for k in ["captured_at", "phase", "mem_available_kib", "swap_free_kib"] if k in value}
    if "csv" in value and "fields" in value:
        row = next(csv.reader([value["csv"]]))
        out["gpu"] = {key: item.strip() for key, item in zip(value["fields"], row) if key != "uuid"}
    return out


def metrics_projection(value):
    allowed = {"observed_output_tokens", "whole_request_seconds", "whole_request_output_tokens_per_second",
               "per_stream_output_tokens", "per_stream_ttft_ms", "valid", "speed_eligible", "reason", "trim_seconds",
               "start_ms", "end_ms", "seconds", "aggregate_output_tokens_per_second", "per_stream_output_tokens_per_second",
               "jain_fairness", "minimum_over_maximum_stream_rate", "per_stream_max_intertoken_gap_ms",
               "minimum_seconds", "minimum_tokens_each"}
    out = {k: child for k, child in value.items() if k in allowed}
    for key in ["common_window", "speed_requirements"]:
        if isinstance(value.get(key), dict):
            out[key] = metrics_projection(value[key])
    return out


def live_projection(value):
    out = {k: value[k] for k in ["running", "queued", "waiting", "admitting", "admission_inflight", "state"] if k in value}
    if "slots" in value:
        out["slots"] = [{k: row[k] for k in ["slot", "state", "prompt_tokens", "generated", "held_tokens"] if k in row}
                        for row in value["slots"]]
    return out


def sample_projection(value):
    out = {k: value[k] for k in ["elapsed_ms", "running", "active_slots", "decoding_slots", "visible_retokenized_tokens_each",
                               "independent_retrieval_JSON_verified_each"] if k in value}
    if "slots" in value:
        out["slots"] = live_projection({"slots": value["slots"]})["slots"]
    return out


def report_projection(value):
    """Keep measured synthetic evidence, omit real identities/config arrays."""
    if not isinstance(value, dict):
        raise Refusal("report/plan projection expects an object")
    out = {k: value[k] for k in SIMPLE_KEYS if k in value}
    for key in ["info", "engine_capacity"]:
        if isinstance(value.get(key), dict):
            out[key] = {k: v for k, v in value[key].items() if k in INFO_KEYS}
    for key in ["native", "native_admission"]:
        if isinstance(value.get(key), dict):
            out[key] = {k: v for k, v in value[key].items() if k in NATIVE_KEYS}
    for key in ["arms", "cases"]:
        if isinstance(value.get(key), list):
            out[key] = [report_projection(row) for row in value[key]]
    if isinstance(value.get("slots"), list):
        out["slots"] = [dict(report_projection(row), **({"slot": row["slot"]} if "slot" in row else {})) for row in value["slots"]]
    if isinstance(value.get("deep_slots"), dict):
        out["deep_slots"] = report_projection(value["deep_slots"])
    if isinstance(value.get("deep_concurrency"), dict):
        out["deep_concurrency"] = report_projection(value["deep_concurrency"])
    if isinstance(value.get("short_recovery"), dict):
        out["short_recovery"] = report_projection(value["short_recovery"])
    if isinstance(value.get("initial_capacity"), dict):
        out["initial_capacity"] = {k: v for k, v in value["initial_capacity"].items() if k in INFO_KEYS}
    if isinstance(value.get("streams"), list):
        out["streams"] = [report_projection(row) for row in value["streams"]]
    if isinstance(value.get("samples"), list):
        out["samples"] = [sample_projection(row) for row in value["samples"]]
    if isinstance(value.get("observation"), dict):
        out["observation"] = sample_projection(value["observation"])
    if isinstance(value.get("drained_metrics_live"), dict):
        out["drained_metrics_live"] = live_projection(value["drained_metrics_live"])
    if isinstance(value.get("primed_live"), dict):
        out["primed_live"] = live_projection(value["primed_live"])
    if isinstance(value.get("priming"), list):
        out["priming"] = [report_projection(row) for row in value["priming"]]
    if isinstance(value.get("response"), dict):
        out["response"] = response_projection(value["response"])
    for key in ["initial_metrics", "final_metrics"]:
        if isinstance(value.get(key), dict):
            metrics = value[key]
            out[key] = {"engine": {k: v for k, v in metrics.get("engine", {}).items() if k in INFO_KEYS}}
    for key in ["initial_health", "final_health"]:
        if isinstance(value.get(key), dict):
            out[key] = {k: value[key][k] for k in ["status", "service", "loaded", "qualified", "max_context", "images", "api_key"] if k in value[key]}
    if isinstance(value.get("telemetry"), dict):
        out["telemetry"] = {k: value["telemetry"][k] for k in ["interval_seconds", "samples", "minimum_mem_available_kib", "minimum_gpu_free_mib", "errors"] if k in value["telemetry"]}
    for key in ["output_metrics", "metrics"]:
        if isinstance(value.get(key), dict):
            out[key] = metrics_projection(value[key])
    if isinstance(value.get("error"), str):
        out["error"] = value["error"]
    if "schedule" in value:
        out["schedule"] = [{k: row[k] for k in ["slots", "prefill"] if k in row} for row in value["schedule"]]
    if "fixture_cases" in value:
        out["fixture_cases"] = [{k: row[k] for k in ["id", "prompt_tokens", "token_sha256"] if k in row} for row in value["fixture_cases"]]
    return out


def local_private_pin_redactions():
    """Read local-only receipts solely to remove private config fingerprints."""
    found = {}
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if (isinstance(child, str) and re.fullmatch(r"[0-9a-f]{64}", child)
                        and ((key.startswith("/") and Path(key).suffix == ".json")
                             or key in {"configuration_sha256", "contract_sha256", "server_config_sha256"})):
                    found[child] = "<private-configuration-sha256>"
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for relative in ["initial-state.json", "promotion262/deployment-result.json", "promotion262/independent-live-audit.json"]:
        path = ROOT / relative
        if path.is_file():
            visit(json.loads(path.read_bytes()))
    return found


def promotion_projection(value, kind):
    if kind == "promotion":
        keys = {"version", "at", "status", "context", "slots", "prefill", "engine", "normal_gate", "normal_production_canaries",
                "near_capacity_prompt_tokens", "repeat_cached_tokens", "independent_live_audit", "eight_deep_simultaneous_generation_qualified",
                "maximum_deep_decoders_observed_in_research", "client_onyx_flash_context", "client_other_values_unchanged",
                "existing_client_sessions_may_cache_old_context", "Sunshine"}
        return {k: value[k] for k in keys if k in value}
    if kind == "live-audit":
        keys = {"kind", "at", "status", "checks", "candidate_private_files_remaining", "scope_limit", "completion_requests_sent",
                "service_or_configuration_actions_performed"}
        out = {k: value[k] for k in keys if k in value}
        if isinstance(value.get("capacity"), dict):
            out["capacity"] = {k: v for k, v in value["capacity"].items() if k in INFO_KEYS}
        if isinstance(value.get("sunshine_user_unit"), dict):
            out["sunshine_user_unit"] = {k: value["sunshine_user_unit"][k] for k in ["ActiveState", "SubState", "UnitFileState"] if k in value["sunshine_user_unit"]}
        return out
    if kind == "deployment":
        keys = {"status", "at", "version", "context", "slots", "prefill", "runtime_wrapper_changed", "unit_changed", "Sunshine_modified",
                "state_history_modified", "rollback_server_and_helper_saved", "eight_deep_simultaneous_generation_qualified"}
    else:
        keys = {"status", "approved_contexts", "running_contract_still131072", "actual_engine_context_must_equal_sealed_contract",
                "fallback_or_minimum_capacity_match_added", "changed_source_files", "changed_logic_lines", "unchanged_consumers",
                "changes", "unchanged", "runtime_gate", "CPU_tests", "protected_mutations_performed", "before_sha256", "draft_sha256"}
    return {k: value[k] for k in keys if k in value}


def encode(normalizer, raw, entry):
    kind = entry["kind"]
    if kind == "binary":
        return raw, {"kind": "unchanged synthetic image"}
    if kind in {"report", "plan", "review", "live-audit", "deployment", "promotion"}:
        value = json.loads(raw)
        if value.get("status") in {"running", "starting"} and not value.get("finished_at"):
            raise Refusal("selected report is not finalized")
        projection = report_projection(value) if kind in {"report", "plan"} else promotion_projection(value, kind)
        raw = (json.dumps(projection, indent=2, allow_nan=False) + "\n").encode()
    elif kind == "telemetry":
        raw = ("\n".join(json.dumps(telemetry_projection(json.loads(line))) for line in raw.decode().splitlines() if line.strip()) + "\n").encode()
    suffix = ".jsonl" if kind == "telemetry" else Path(entry["source"]).suffix
    public, counts = normalizer.normalize(raw, suffix)
    text = public.decode()
    text, n = OWNER.subn("<maintenance-owner>", text)
    if n:
        counts["context_maintenance_owner"] = n
    # Python source includes synthetic process IDs in mock defaults/assignments.
    # Replacing numeric literals there changes semantics and can break syntax.
    if not is_python_source(entry["source"]):
        text, n = PROCESS_ID.subn(lambda m: m.group(1) + "=<redacted-id>", text)
        if n:
            counts["process_identifier"] = n
    for private, role in PRIVATE_PINS.items():
        n = text.count(private)
        if n:
            counts[role.strip("<>")] = n
            text = text.replace(private, role)
    return text.encode(), {"kind": kind, "redactions": counts}


def deterministic_gzip(raw):
    stream = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0, compresslevel=9) as zipped:
        zipped.write(raw)
    return stream.getvalue()


def is_python_source(relative):
    name = Path(relative).name
    return name.endswith(".py") or ".py." in name


def strict_json(raw):
    def invalid_constant(value):
        raise Refusal("nonfinite JSON constant: " + value)
    return json.loads(raw, parse_constant=invalid_constant)


def validate_tokens(value):
    """Validate any full fixture token vector without importing model code."""
    checked = 0
    if isinstance(value, dict):
        if "tokens" in value and "token_sha256" in value and "prompt_tokens" in value:
            tokens = value["tokens"]
            if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
                raise Refusal("published fixture has invalid token IDs")
            if type(value["prompt_tokens"]) is not int or len(tokens) != value["prompt_tokens"]:
                raise Refusal("published fixture token count differs")
            digest = sha((",".join(map(str, tokens)) + "\n").encode())
            if digest != value["token_sha256"]:
                raise Refusal("published fixture token hash differs")
            checked += 1
        for child in value.values():
            checked += validate_tokens(child)
    elif isinstance(value, list):
        for child in value:
            checked += validate_tokens(child)
    return checked


def validate_published(relative, raw, expected_uncompressed_sha256=None):
    compressed = relative.endswith(".gz")
    if compressed:
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError, zlib.error) as error:
            raise Refusal("published gzip does not decompress: " + relative) from error
        relative = relative[:-3]
    if expected_uncompressed_sha256 is not None and sha(raw) != expected_uncompressed_sha256:
        raise Refusal("published decompressed content hash differs: " + relative)
    result = {"python_sources": 0, "json_documents": 0, "fixture_token_vectors": 0, "gzip_archives": int(compressed)}
    if is_python_source(relative):
        try:
            ast.parse(raw.decode(), filename=relative)
        except SyntaxError as error:
            raise Refusal("published Python source is invalid: " + relative) from error
        result["python_sources"] = 1
    elif relative.endswith(".json"):
        value = strict_json(raw)
        result["json_documents"] = 1
        result["fixture_token_vectors"] = validate_tokens(value)
    elif relative.endswith(".jsonl"):
        for line in raw.decode().splitlines():
            if line.strip():
                result["fixture_token_vectors"] += validate_tokens(strict_json(line))
                result["json_documents"] += 1
    return result


def execute(args, normalizer, normalizer_path, selection):
    if not args.root_authorized or args.attestation is None:
        raise Refusal("execution requires the root's explicit final production-readiness/publication attestation")
    authority_raw = args.attestation.read_bytes()
    authority = json.loads(authority_raw)
    for key in ["publication_authorized", "private_experiments_finished", "production_final_ready", "sunshine_inactive", "user_authorized_promotion"]:
        if authority.get(key) is not True:
            raise Refusal("publication attestation is incomplete: " + key)
    if (authority.get("production_context"), authority.get("production_slots"), authority.get("production_prefill")) != (262144, 8, 8192):
        raise Refusal("final attestation does not match the user's authorized 262K/8-slot/8K profile")
    if type(authority.get("private_configs_remaining")) is not int or authority["private_configs_remaining"] != 0:
        raise Refusal("publication attestation has remaining private configurations")
    for path, expected in {**GUARDS, **MEASURED_HELPERS}.items():
        if sha((ROOT / path).read_bytes()) != expected:
            raise Refusal("guard provenance source differs: " + path)
    chosen = select(selection)
    PRIVATE_PINS.update(local_private_pin_redactions())
    destination = ROOT / "publication/.stages" / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") / DESTINATION
    destination.mkdir(parents=True, exist_ok=False)
    records, snapshots = [], {}
    validation = {"python_sources": 0, "json_documents": 0, "fixture_token_vectors": 0, "gzip_archives": 0}
    for relative, entry in chosen.items():
        raw = entry["path"].read_bytes()
        snapshots[relative] = sha(raw)
        public, policy = encode(normalizer, raw, entry)
        unpacked_sha = sha(public)
        target = relative + (".gz" if entry.get("gzip") else "")
        if entry.get("gzip"):
            public = deterministic_gzip(public)
        output = destination / target
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(public)
        checked = validate_published(target, output.read_bytes(), unpacked_sha)
        for key, count in checked.items():
            validation[key] += count
        records.append({"source_relative_path": relative, "source_sha256": sha(raw), "source_bytes": len(raw),
                        "published_relative_path": target, "published_sha256": sha(public), "published_bytes": len(public),
                        "normalized_uncompressed_sha256": unpacked_sha, "gzip_mtime": 0 if entry.get("gzip") else None,
                        "projection_and_redaction": policy})
    if sha(args.attestation.read_bytes()) != sha(authority_raw):
        raise Refusal("publication authority changed while staging")
    if set(select(selection)) != set(chosen):
        raise Refusal("selected source set changed while staging")
    for relative, digest in snapshots.items():
        if sha((ROOT / relative).read_bytes()) != digest:
            raise Refusal("a source artifact changed while staging: " + relative)
    manifest = {"schema_version": 1, "prepared_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "destination_relative_path": str(DESTINATION), "sources": records,
                "selection_sha256": sha(args.selection.read_bytes()), "normalizer_source_sha256": sha(normalizer_path.read_bytes()),
                "root_final_production_attestation_sha256": sha(authority_raw),
                "guard_source_provenance": GUARDS, "guard_difference": "Two refusal-message spacing changes only; numerical thresholds/control flow unchanged. Both original source identities retained; public copies are redacted.",
                "measured_API_helper_source_provenance": MEASURED_HELPERS,
                "published_artifact_validation": {"status": "passed", **validation},
                "gzip": {"mtime": 0, "filename": "", "compresslevel": 9, "python": sys.version.split()[0], "zlib": zlib.ZLIB_VERSION},
                "omitted": selection["omitted"], "repository_mutated": False, "commit_or_push": False,
                "evidence_limits": "A completed short JSON response can precede later deep-slot admissions. Report actual peak live slots; do not infer all eight simultaneously generating or preserved-history reuse solely from eight individual deep fills."}
    (destination / "PUBLICATION-MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checksum = [sha(path.read_bytes()) + "  " + path.relative_to(destination).as_posix() for path in sorted(destination.rglob("*")) if path.is_file()]
    (destination / "SHA256SUMS.txt").write_text("\n".join(checksum) + "\n")
    print(json.dumps({"status": "staged_local_only", "destination": str(destination), "files": len(records), "repository_mutated": False}))


def self_test(normalizer):
    private = "-".join(["codex", "context262k", "20261103", "example"])
    source = {"status": "passed", "slots": 8, "identity": {"pid": 123, "secret": "not public"},
              "initial_health": {"api_key": True, "loaded": True, "host": "not public"},
              "cases": [{"id": "synthetic", "status": "passed", "response": {"id": "not public", "usage": {"completion_tokens": 3},
                         "choices": [{"message": {"content": "OK", "tool_calls": [{"id": "not public", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}}]}}]}
    projected = report_projection(source)
    assert "identity" not in projected and projected["initial_health"]["api_key"] is True
    assert "not public" not in json.dumps(projected)
    deep = report_projection({"deep_concurrency": {"eight_concurrent_deep_streams_observed": True,
        "configured_EOS_logit_bias": {"123": -100}, "observation": {"active_slots": 8, "slots": [{"slot": 0, "prompt_tokens": 252000, "request_id": "not public", "pid": 123}]},
        "streams": [{"id": "synthetic_slot", "prompt_tokens": 252000, "pid": 123}], "identity": {"pid": 123}}})
    assert deep["deep_concurrency"]["eight_concurrent_deep_streams_observed"] is True
    assert "not public" not in json.dumps(deep) and '"pid"' not in json.dumps(deep)
    data, _ = encode(normalizer, ("owner=" + private + "\r\n").encode(), {"kind": "text", "source": "x.txt"})
    assert private.encode() not in data and b"\r" not in data
    promotion_owner = "-".join(["codex", "context262k", "promote", "20261103", "example"])
    promotion_source = ("OWNER = " + repr(promotion_owner) + "\n").encode()
    promotion_public, policy = encode(normalizer, promotion_source, {"kind": "text", "source": "promotion.py"})
    assert promotion_owner.encode() not in promotion_public
    assert policy["redactions"]["context_maintenance_owner"] == 1
    assert ast.literal_eval(ast.parse(promotion_public.decode()).body[0].value) == "<maintenance-owner>"
    assert deterministic_gzip(data) == deterministic_gzip(data)
    assert gzip.decompress(deterministic_gzip(data)) == data
    own, _ = encode(normalizer, Path(__file__).read_bytes(), {"kind": "text", "source": "x.py"})
    ast.parse(own.decode())
    literal = b"def identity(self, pid=500):\n    pgid=500\n    return pid, pgid\n"
    public, policy = encode(normalizer, literal, {"kind": "text", "source": "mock.py"})
    assert public == literal and "process_identifier" not in policy["redactions"]
    assert validate_published("mock.py", public)["python_sources"] == 1
    log, _ = encode(normalizer, b"pid=500 pgid:501\n", {"kind": "text", "source": "mock.log"})
    assert b"500" not in log and b"501" not in log
    fixture = {"tokens": [1, 500, 3], "prompt_tokens": 3, "token_sha256": sha(b"1,500,3\n")}
    fixture_raw = json.dumps(fixture).encode()
    checked = validate_published("fixture.json.gz", deterministic_gzip(fixture_raw), sha(fixture_raw))
    assert checked["fixture_token_vectors"] == 1 and checked["gzip_archives"] == 1
    for relative, raw in [("broken.py", b"def identity(pid=<redacted-id>): pass"),
                          ("broken.json", b'{"x":NaN}'),
                          ("broken.json.gz", deterministic_gzip(b'{"tokens":[1],"prompt_tokens":2,"token_sha256":"bad"}'))]:
        try:
            validate_published(relative, raw)
        except Refusal:
            pass
        else:
            raise AssertionError("invalid public artifact was accepted")
    source_count = 0
    for relative, entry in select(strict_json((ROOT / "publication/SELECTION.json").read_bytes())).items():
        if is_python_source(relative):
            public, _ = encode(normalizer, entry["path"].read_bytes(), entry)
            source_count += validate_published(relative, public)["python_sources"]
    print(json.dumps({"status": "self_test_passed", "python_sources_validated": source_count,
                      "receipts_read": False, "GPU_or_API_work": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    parser.add_argument("--selection", type=Path, default=ROOT / "publication/SELECTION.json")
    parser.add_argument("--normalizer", type=Path)
    parser.add_argument("--root-authorized", action="store_true")
    parser.add_argument("--attestation", type=Path)
    args = parser.parse_args()
    normalizer, path = load_normalizer(args.normalizer)
    if args.self_test:
        self_test(normalizer)
    elif not args.execute:
        selected = select(json.loads(args.selection.read_text()))
        print(json.dumps({"status": "plan_only", "selected": [{"source": key, "kind": value["kind"], "gzip": bool(value.get("gzip"))} for key, value in selected.items()],
                          "repository_mutated": False, "requires_root_final_production_attestation": True}, indent=2))
    else:
        execute(args, normalizer, path, json.loads(args.selection.read_text()))


if __name__ == "__main__":
    try:
        main()
    except (Refusal, OSError, UnicodeError, ValueError) as error:
        print(json.dumps({"status": "refused", "error": str(error) if isinstance(error, Refusal) else type(error).__name__}), file=sys.stderr)
        raise SystemExit(2)
