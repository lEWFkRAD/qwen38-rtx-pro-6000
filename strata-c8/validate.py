#!/usr/bin/env python3
"""Validate the published recipe offline; no downloads, GPU or server access."""
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def near(actual, expected, tolerance, message):
    check(abs(actual - expected) <= tolerance, message)


def main():
    manifest = {}
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        check(name not in manifest, "Duplicate manifest path")
        path = (ROOT / name).resolve()
        check(path.is_relative_to(ROOT) and path.is_file(), "Invalid manifest path: " + name)
        check(re.fullmatch(r"[0-9a-f]{64}", expected), "Malformed digest")
        check(digest(path) == expected, "Hash mismatch: " + name)
        manifest[name] = expected
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
              if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts}
    check(set(manifest) == actual, "Manifest does not cover the complete publication")

    result = read("evidence/results-2026-10-22.json")
    steady = result["steady_c8"]
    check(steady["concurrency"] == 8 and steady["qualified_common_windows_per_arm"] == 2,
          "C8 sample scope changed")
    check(steady["output_token_limit_per_request"] == 512, "Output length changed")
    for arm in (steady["baseline"], steady["candidate"]):
        check(len(arm["rounds"]) == 2, "Expected two windows per arm")
        for row in arm["rounds"]:
            rate = row["output_tokens_per_session_in_window"] / row["window_duration_s"]
            near(row["per_session_output_tokens_per_s"], rate, .001, "Session rate arithmetic")
            near(row["aggregate_output_tokens_per_s"], 8 * rate, .002, "Aggregate rate arithmetic")
        for kind in ("aggregate", "per_session"):
            field = kind + "_output_tokens_per_s"
            near(arm["median_" + field], statistics.median(r[field] for r in arm["rounds"]),
                 1e-8, "Round median mismatch")
    gain = 100 * (steady["candidate"]["median_per_session_output_tokens_per_s"] /
                  steady["baseline"]["median_per_session_output_tokens_per_s"] - 1)
    near(steady["matched_steady_c8_gain_percent"], gain, .001, "Steady gain arithmetic")
    pairs = steady["paired_output_sha256"]
    check(len(pairs) == len({(x["round"], x["request"]) for x in pairs}) == 16,
          "Expected sixteen distinct paired output hashes")
    for pair in pairs:
        check(re.fullmatch(r"[0-9a-f]{64}", pair["baseline_sha256"]), "Malformed output hash")
        check(pair["baseline_sha256"] == pair["candidate_sha256"] and pair["completion_tokens"] == 512,
              "Output parity mismatch")
    check(steady["full_generated_output_sha256_matches"] == len(pairs), "Parity summary mismatch")
    whole = result["candidate_request_throughput"]
    for row in whole["rounds"]:
        near(row["aggregate_output_tokens_per_s"], whole["output_tokens_per_round"] /
             row["request_wall_elapsed_s"], .001, "Whole-request rate arithmetic")
    near(whole["median_aggregate_output_tokens_per_s"],
         statistics.median(r["aggregate_output_tokens_per_s"] for r in whole["rounds"]),
         1e-8, "Whole-request median mismatch")
    mixed = result["optional_separate_mixed_workload"]
    near(mixed["gain_percent"], 100 * (mixed["candidate_combined_output_tokens_per_s_during_cold_prefill"] /
         mixed["baseline_combined_output_tokens_per_s_during_cold_prefill"] - 1), 1e-8, "Mixed gain arithmetic")
    check(mixed["generating_sessions"] == 7 and mixed["qualified_intervals_per_arm"] == 1,
          "Mixed workload scope mismatch")
    for source in result["source_receipts"]:
        check(Path(source["name"]).name == source["name"] and
              re.fullmatch(r"[0-9a-f]{64}", source["sha256"]), "Invalid source identity")

    cfg = read("config/strata-server.example.json")
    # Preserve the measured October 22 inputs; keep later saved settings dated.
    current = read("config/strata-server-2026-10-25.example.json")
    expected_current = json.loads(json.dumps(cfg))
    expected_current["args"][expected_current["args"].index("--prefill") + 1] = "32768"
    expected_current["fit_max_tokens"] = True
    check(current == expected_current, "October 25 config differs beyond the two documented settings")
    check((ROOT / "CURRENT-CONFIG-2026-10-25.md").is_file(), "Missing dated configuration context")
    profile = result["current_saved_recipe"]
    check(cfg["parallel"] == profile["resident_batch_slots"] == 8, "Slot count mismatch")
    check(cfg["env"] == profile["environment"], "Child engine environment mismatch")
    check(cfg["host"] == "127.0.0.1" and cfg["api_key"] == "" and
          cfg["compatibility_api_keys"] == [], "Published example has deployment credentials or binding")
    for flag, value in (("--max-context", "131072"), ("--kv", "int8"),
                        ("--kv-resident", "131072"), ("--prefill", "16384"),
                        ("--spec", "4"), ("--spec-min-p", "0.5"),
                        ("--ple-io", "ram"), ("--pcie-frac", "0")):
        check(cfg["args"][cfg["args"].index(flag) + 1] == value, "Argument mismatch: " + flag)
    check(cfg["sampling"]["experimental_speed_projection"] is True, "Variant projection missing")
    parent = (ROOT / "config/service-environment.example.conf").read_text()
    for assignment in ("STRATA_VERIFY_ALL_RESIDENT=1", "STRATA_ARENA_MMAP=1",
                       "STRATA_ADMIT_ROTATE=1", "STRATA_ADMIT_ROTATE_QUANTUM=32768",
                       "STRATA_SLOT_QUEUE=1"):
        check("Environment=" + assignment in parent, "Missing parent setting: " + assignment)
    reproduction = result["reproduction"]
    check(digest(ROOT / reproduction["harness"]) == reproduction["harness_sha256"], "Harness binding mismatch")
    check((ROOT / reproduction["note"]).is_file(), "Missing benchmark method")
    runtime = read("evidence/runtime-provenance.json")
    check(digest(ROOT / "patches/runtime-0142.patch") == runtime["patch_sha256"], "Runtime patch binding")
    engine = read("evidence/engine-release.json")
    check(digest(ROOT / "evidence/engine-component.json") == engine["manifest_sha256"], "Signed manifest bytes")
    signed = read("evidence/engine-component.json")
    check({k: v for k, v in signed.items() if k != "signature"} == engine["manifest"], "Engine selection mismatch")
    check(signed["version"] == profile["engine_version"] == "0.1.46", "Engine version mismatch")
    model = read("evidence/model-assets.json")
    check(len(model["verified_download_files"]) == 4, "Model asset scope mismatch")
    for name, item in model["verified_download_files"].items():
        check(item["sha256"] == model["publisher_manifest_sha256"][name] and item["matches_hf_lfs"],
              "Model publisher binding mismatch")

    for path in [REPO / "README.md", *ROOT.rglob("*.md")]:
        text = path.read_text(encoding="utf-8")
        links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
        links += re.findall(r'(?:src|href)="([^"]+)"', text)
        for link in links:
            if "://" in link or link.startswith("#"):
                continue
            target = (path.parent / link.split("#", 1)[0]).resolve()
            check(target.is_relative_to(REPO) and target.exists(), "Broken local link: " + link)
    for path in ROOT.rglob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    image = (ROOT / "assets/qwen-c8-firm-throughput-v2.png").read_bytes()
    check(image[:8] == b"\x89PNG\r\n\x1a\n", "Schematic is not PNG")
    width, height = struct.unpack(">II", image[16:24])
    check(width >= 1000 and height >= 1250, "Schematic resolution changed")
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in (".md", ".json", ".patch", ".conf"):
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in (r"/home/[^ /]+", r"(?i)tail[0-9a-f]+\.ts\.net",
                        r"\b100\.(?:\d{1,3}\.){2}\d{1,3}\b", r"-----BEGIN .*PRIVATE KEY-----",
                        r"\bgh[pousr]_[A-Za-z0-9]{20,}", r"\bsk-[A-Za-z0-9]{20,}"):
            check(not re.search(pattern, text), "Private data pattern: " + path.name)
    print(f"PASS: {len(manifest)} recipe hashes; metric arithmetic; 16 paired hashes; config, source and asset bindings; links; syntax; image; privacy")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, IndexError) as exc:
        print("FAIL:", exc, file=sys.stderr)
        sys.exit(1)
