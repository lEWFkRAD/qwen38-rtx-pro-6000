#!/usr/bin/env python3
"""Validate the sanitized cached NEXTN + ReplaySSM public receipts."""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"

SOURCES = {
    "cache-nextn-control-tps-20260901.json": (
        "control-tps-20260901T102100Z.json",
        "42c422bb1ba7348cb74166fd86cd5722c4c92359e53bbaf440ed5fd00bfbbdd7",
    ),
    "cache-nextn-replayssm-candidate-tps-20260901.json": (
        "candidate-tps.json",
        "161b70dec88d658eb412fe31cfeac92748d7fa19e28b9bce1149d49fc1e03fa1",
    ),
    "cache-nextn-replayssm-candidate-ready-20260901.json": (
        "candidate-ready.json",
        "ccfe4ee70460691bcd5df7ded4c0a4d30eee516bdf5d19a081f9539c3c28cc17",
    ),
    "cache-nextn-replayssm-candidate-cache-20260901.json": (
        "candidate-cache-acceptance.json",
        "a902066ed5692111b50559f901920622189c68b82e52b2c6a5b69772c0861ac0",
    ),
    "cache-nextn-replayssm-candidate-branch-matrix-20260901.json": (
        "candidate-cache-branch-matrix.json",
        "bc5b9008abe5a04112bf3d59a82a186336b8bf00b2c1c3e8ba4a4b524a47b39d",
    ),
    "cache-nextn-replayssm-candidate-workload-20260901.json": (
        "candidate-workload.json",
        "093756efbfe894af6ce470c66a7ecfbc8e475b461f806235dab09d2b249cd171",
    ),
    "cache-nextn-replayssm-promotion-ready-20260901.json": (
        "promoted-ready.json",
        "e4bba0eeec3e7fe87b76628199993153e0ae4157dec268da7088ba981a19792c",
    ),
    "cache-nextn-replayssm-promotion-cache-20260901.json": (
        "post-promotion-cache-acceptance.json",
        "45220d386b356dd2c581cf41a738ec17e84ec8ef0942cfdabb48d3f80cec22f2",
    ),
    "cache-nextn-replayssm-promotion-workload-20260901.json": (
        "post-promotion-workload.json",
        "0ccccd7ad83819bfafd56309cce6fb811c712f631878d9bd553a4c91cb9437be",
    ),
}

TPS_REMOVED = [
    "base_url",
    "idle_samples",
    "started_utc",
    "ended_utc",
    "started_monotonic_s",
    "first_token_monotonic_s",
    "last_token_monotonic_s",
    "ended_monotonic_s",
]

REMOVED_FIELDS = {
    "cache-nextn-control-tps-20260901.json": TPS_REMOVED,
    "cache-nextn-replayssm-candidate-tps-20260901.json": TPS_REMOVED,
    "cache-nextn-replayssm-candidate-ready-20260901.json": [
        "container_id",
        "speculative_draft_model_path",
    ],
    "cache-nextn-replayssm-candidate-cache-20260901.json": ["schema"],
    "cache-nextn-replayssm-candidate-branch-matrix-20260901.json": ["schema"],
    "cache-nextn-replayssm-candidate-workload-20260901.json": [],
    "cache-nextn-replayssm-promotion-ready-20260901.json": [
        "container_id",
        "speculative_draft_model_path",
    ],
    "cache-nextn-replayssm-promotion-cache-20260901.json": ["schema"],
    "cache-nextn-replayssm-promotion-workload-20260901.json": [],
}


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_wrapper(name: str) -> dict[str, Any]:
    data = (EVIDENCE / name).read_bytes()
    assert data.endswith(b"\n")
    assert not data.startswith(b"\xef\xbb\xbf")
    wrapper = json.loads(
        data,
        parse_constant=reject_constant,
        object_pairs_hook=no_duplicate_pairs,
    )
    assert set(wrapper) == {"publication", "receipt"}
    publication = wrapper["publication"]
    expected_source, expected_sha = SOURCES[name]
    assert publication["schema"] == "qwen38.public-receipt-wrapper/v1"
    assert publication["source_receipt"] == expected_source
    assert publication["source_sha256"] == expected_sha
    assert publication["removed_fields"] == REMOVED_FIELDS[name]
    return wrapper


def walk(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)
    elif isinstance(value, str):
        yield value


def validate_privacy(receipt: dict[str, Any]) -> None:
    forbidden_keys = {
        "base_url",
        "idle_samples",
        "container_id",
        "speculative_draft_model_path",
        "started_utc",
        "ended_utc",
        "started_monotonic_s",
        "first_token_monotonic_s",
        "last_token_monotonic_s",
        "ended_monotonic_s",
        "messages",
        "content",
        "prompt",
        "response",
    }
    strings = []
    for value in walk(receipt):
        if isinstance(value, str):
            strings.append(value)
    assert forbidden_keys.isdisjoint(strings)
    joined = "\n".join(strings).lower()
    for fragment in (
        "/home/",
        "c:\\users\\",
        "administrator",
        "jeffrey",
        "watts",
        "onyx.",
        "192.168.",
        "tail3bf",
        "authorization:",
        "bearer ",
        "api_key",
        "password",
    ):
        assert fragment not in joined
    assert re.search(r"(?:^|[^a-z])[a-z]:\\", joined) is None
    cgnat = r"(?<!\d)100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}(?!\d)"
    assert re.search(cgnat, joined) is None
    assert re.search(r"https?://", joined) is None


def measured_cells(receipt: dict[str, Any], concurrency: int) -> list[dict[str, Any]]:
    return [
        cell
        for cell in receipt["cells"]
        if cell["concurrency"] == concurrency and cell["measured"]
    ]


def validate_tps(receipt: dict[str, Any], medians: dict[int, float]) -> None:
    assert receipt["schema"] == "qwen38-flash-next-tps-ab-v2"
    assert receipt["model"] == "qwen3.8-flash-next"
    assert receipt["settings"]["concurrency"] == [1, 2, 4]
    assert receipt["settings"]["warmups"] == 1
    assert receipt["settings"]["repeats"] == 6
    assert receipt["settings"]["max_tokens"] == 512
    assert len(receipt["cells"]) == 21
    summary = {item["concurrency"]: item for item in receipt["summary"]}
    for concurrency, expected in medians.items():
        cells = measured_cells(receipt, concurrency)
        assert len(cells) == 6
        observed = statistics.median(
            cell["aggregate_decode_tok_s"] for cell in cells
        )
        assert math.isclose(observed, expected, abs_tol=0.0005)
        assert summary[concurrency]["median_aggregate_decode_tok_s"] == expected
        assert summary[concurrency]["all_ok"] is True


def validate_ready(receipt: dict[str, Any], pool: int) -> None:
    assert receipt["model"] == "qwen3.8-flash-next"
    assert receipt["kv_cache_token_pools"] == [pool, pool]
    assert receipt["resolved_max_total_num_tokens"] == pool
    assert receipt["speculative_algorithm"] == "EAGLE"
    assert receipt["speculative_num_steps"] == 1
    assert receipt["speculative_eagle_topk"] == 1
    assert receipt["speculative_num_draft_tokens"] == 2
    assert receipt["speculative_use_rejection_sampling"] is True
    assert receipt["enable_linear_replayssm_spec"] is True
    assert receipt["linear_replayssm_cache_len"] == 16
    assert receipt["disable_radix_cache"] is False
    assert receipt["mamba_max_states_per_path"] == 1
    assert receipt["cuda_graph_config"]["decode"]["bs"] == [1, 2, 3, 4]
    assert receipt["cuda_graph_config"]["prefill"]["backend"] == "disabled"


def validate_cache(receipt: dict[str, Any], pool: int) -> None:
    assert receipt["passed"] is True
    assert receipt["runtime_start"]["max_total_num_tokens"] == pool
    assert receipt["realistic_repeat"]["hot"]["prompt_tokens"] == 54_028
    assert receipt["realistic_repeat"]["hot"]["cached_tokens"] == 54_016
    branches = receipt["five_branch_retention"]
    assert len(branches) == 5
    assert [item["hot"]["prompt_tokens"] for item in branches] == [11_028] * 5
    assert [item["hot"]["cached_tokens"] for item in branches] == [11_008] * 5


def validate_workload(receipt: dict[str, Any], pool: int, acceptance: float) -> None:
    assert receipt["passed"] is True
    assert receipt["campaign_passed"] is True
    assert receipt["attempts"] == 34
    assert receipt["warmup_attempts"] == 2
    assert receipt["measured_attempts"] == 32
    assert receipt["failed_count"] == 0
    assert receipt["missing_metrics_count"] == 1
    assert receipt["runtime_info_start"]["max_total_num_tokens"] == pool
    assert receipt["summary"]["all"]["weighted_accept_rate"] == acceptance
    long_results = {
        item["case_id"]: item
        for item in receipt["results"]
        if item["workload_class"] == "long_retrieval"
    }
    assert long_results["long-119k"]["metrics"]["prompt_tokens"] == 119_735
    assert long_results["long-119k"]["metrics"]["completion_tokens"] == 512
    assert long_results["long-130k"]["metrics"]["prompt_tokens"] == 130_831
    assert long_results["long-overlimit"]["status_code"] == 400


def test_public_receipts() -> None:
    wrappers = {name: load_wrapper(name) for name in SOURCES}
    for wrapper in wrappers.values():
        validate_privacy(wrapper["receipt"])

    control = wrappers["cache-nextn-control-tps-20260901.json"]["receipt"]
    candidate = wrappers[
        "cache-nextn-replayssm-candidate-tps-20260901.json"
    ]["receipt"]
    validate_tps(control, {1: 97.882, 2: 185.666, 4: 332.794})
    validate_tps(candidate, {1: 144.563, 2: 255.986, 4: 434.883})
    assert control["settings"]["require_idle_seconds"] == 60
    assert candidate["settings"]["require_idle_seconds"] == 0
    for concurrency, expected in {1: 47.690, 2: 37.874, 4: 30.676}.items():
        before = statistics.median(
            cell["aggregate_decode_tok_s"]
            for cell in measured_cells(control, concurrency)
        )
        after = statistics.median(
            cell["aggregate_decode_tok_s"]
            for cell in measured_cells(candidate, concurrency)
        )
        assert math.isclose((after / before - 1) * 100, expected, abs_tol=0.0005)
    for concurrency, expected in {1: -2.1710, 2: 15.0485, 4: 9.9665}.items():
        before = statistics.median(
            cell["max_ttft_s"] for cell in measured_cells(control, concurrency)
        )
        after = statistics.median(
            cell["max_ttft_s"] for cell in measured_cells(candidate, concurrency)
        )
        assert math.isclose((after - before) * 1000, expected, abs_tol=1e-9)

    validate_ready(
        wrappers["cache-nextn-replayssm-candidate-ready-20260901.json"][
            "receipt"
        ],
        151_040,
    )
    validate_ready(
        wrappers["cache-nextn-replayssm-promotion-ready-20260901.json"][
            "receipt"
        ],
        151_808,
    )
    validate_cache(
        wrappers["cache-nextn-replayssm-candidate-cache-20260901.json"][
            "receipt"
        ],
        151_040,
    )
    validate_cache(
        wrappers["cache-nextn-replayssm-promotion-cache-20260901.json"][
            "receipt"
        ],
        151_808,
    )

    branch_matrix = wrappers[
        "cache-nextn-replayssm-candidate-branch-matrix-20260901.json"
    ]["receipt"]
    assert branch_matrix["passed"] is True
    assert len(branch_matrix["scenarios"]) == 9
    assert all(item["passed"] for item in branch_matrix["scenarios"])

    validate_workload(
        wrappers["cache-nextn-replayssm-candidate-workload-20260901.json"][
            "receipt"
        ],
        151_040,
        0.8833046471600688,
    )
    validate_workload(
        wrappers["cache-nextn-replayssm-promotion-workload-20260901.json"][
            "receipt"
        ],
        151_808,
        0.8953931416695532,
    )


def main() -> None:
    test_public_receipts()
    print("PUBLIC_CACHE_NEXTN_RECEIPTS_OK")


if __name__ == "__main__":
    main()
