#!/usr/bin/env python3
"""Audit completed fictional C8 receipts without contacting the model server."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def distribution(values):
    return {"min": min(values), "median": statistics.median(values),
            "max": max(values), "mean": statistics.mean(values)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze(directory):
    directory = Path(directory)
    raw = (directory / "report.json").read_bytes()
    report = json.loads(raw)
    fixtures = json.loads((directory / "fixtures.json").read_bytes())
    fixture_hash = hashlib.sha256(json.dumps(
        fixtures, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(fixture_hash == report["fixture_sha256"], "fixture hash differs")
    require(report["status"] == "passed" and len(report["rounds"]) == 2,
            "two completed passing rounds required")
    cases = {case["id"]: case for case in fixtures["cases"]}
    require(len(cases) == 8, "eight distinct fictional cases required")
    rows = []
    for r in report["rounds"]:
        require(r["status"] == "passed" and r["eligible_for_speed"]
                and not r["contaminated"] and r["owned_http_threads_drained"],
                "round eligibility failed")
        requests = r["requests"]
        require(len(requests) == 8 and r["concurrent_clients"] == 8,
                "eight completed requests required")
        require(len({q["response_id"] for q in requests}) == 8,
                "response identities must be independent")
        require({q["id"] for q in requests} == set(cases), "case identities differ")
        rates, outputs, reused = [], [], []
        for q in requests:
            usage = q["usage"]
            require(usage["prompt_tokens"] == cases[q["id"]]["prompt_tokens"],
                    "original prompt usage differs")
            require(usage["completion_tokens"] == 1024, "output cap not reached")
            outputs.append(usage["completion_tokens"])
            rates.append(usage["completion_tokens"] * 1000 / q["wall_ms"])
            reused.append((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
        require(all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(
            rates, r["per_stream_whole_request_tokens_s"])), "per-session rates differ")
        aggregate = sum(outputs) / r["whole_request_seconds"]
        require(math.isclose(aggregate, r["whole_request_output_tokens_s"], rel_tol=1e-12),
                "aggregate accounting differs")
        totals = r["solo_observation"]
        require(totals["totals_requests_delta"] == 8
                and totals["totals_output_delta"] == sum(outputs), "ledger differs")
        samples = [s for s in r["samples"] if s["decoding_slots"] == 8]
        require(len(samples) >= 2 and r["peak_observed_decoding_slots"] == 8,
                "actual eight-way decoding not observed")
        span = (samples[-1]["elapsed_ms"] - samples[0]["elapsed_ms"]) / 1000
        require(math.isclose(span, r["all_stream_decoding_samples_span_seconds"],
                             rel_tol=1e-12) and span >= .5, "decode overlap differs")
        progresses = []
        for slot in range(8):
            first = next(s for s in samples[0]["slots"] if s["slot"] == slot)
            last = next(s for s in samples[-1]["slots"] if s["slot"] == slot)
            progress = last["generated"] - first["generated"]
            require(progress > 0, "a decode slot made no observed output progress")
            progresses.append(progress)
        rows.append({
            "label": r["label"], "status": "passed", "actual_sessions": 8,
            "output_tokens_each": outputs, "output_tokens_total": sum(outputs),
            "per_session_end_to_end_tokens_s": rates,
            "per_session_end_to_end_summary": distribution(rates),
            "aggregate_end_to_end_tokens_s": aggregate,
            "whole_request_seconds": r["whole_request_seconds"],
            "ttft_ms": distribution(r["per_stream_ttft_ms"]),
            "cached_prompt_tokens_each": reused,
            "eight_decoding_sample_count": len(samples),
            "eight_decoding_observed_span_seconds": span,
            "observed_generated_progress_each_slot": progresses,
            "common_visible_output_interval_seconds": r["common_visible_output_interval_seconds"],
            "fairness": r["jain_fairness_from_whole_request_rates"],
            "exact_common_window_tokens_s": None,
            "count_scope": "Final SSE usage tokens; SSE chunks are never treated as tokens.",
        })
    return {
        "status": "passed", "source": "receipts/live-C8-r1/report.json",
        "source_report_sha256": hashlib.sha256(raw).hexdigest(),
        "fixture_sha256": fixture_hash, "started_at": report["started_at"],
        "finished_at": report["finished_at"], "context_allocation": 262144,
        "resident_slots": 8, "prefill": 8192,
        "input_tokens_each": report["synthetic_prompt_tokens"], "rounds": rows,
        "scope": "Two unchanged-production eight-session short-prompt rounds, with 1024 output tokens per session.",
        "eight_near_capacity_histories_qualified": False,
        "tuning_gain_established": False, "configuration_changed": False,
        "comparison_limit": "The historical approximately 28 tokens/s/session API test used roughly 4K prompts and 256 output tokens. Different input, output and cache conditions prevent a speedup claim.",
        "rate_limit": "Whole-request rates include prompt processing and admission. No exact steady token-arrival rate is inferred from buffered SSE delivery.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory), indent=2, allow_nan=False))
