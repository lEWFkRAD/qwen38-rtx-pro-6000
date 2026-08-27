#!/usr/bin/env python3
"""Collect per-workload speculative-acceptance receipts from SGLang.

The client is intentionally serialized and non-streaming because the pinned
SGLang OpenAI endpoint exposes per-request speculative counters only when
``return_meta_info=true`` and ``stream=false``. Response text is graded but is
not copied into the receipt; only its character count is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


RECEIPT_SCHEMA = "qwen38.mtp-workload-acceptance/v2"
ROLLING_SCHEMA = "qwen38.mtp-workload-histogram/v2"
WORKLOAD_CLASSES = ("chat", "tool_json", "math", "long_retrieval")
CAMPAIGN_MINIMUMS = {"chat": 9, "tool_json": 10, "math": 10, "long_retrieval": 2}
BIN_EDGES = tuple(round(1.0 + 0.25 * i, 2) for i in range(13))
ARM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
CASE_ID_RE = ARM_RE
EXPECT_KEYS = {
    "status_code",
    "finish_reason",
    "content_exact",
    "content_starts_with",
    "content_contains_all",
    "json_exact",
    "reasoning_empty",
    "tool_call",
    "error",
}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_temp_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp.open("xb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    return temp


def atomic_create_json(path: Path, value: Any) -> None:
    """Publish a complete receipt without ever replacing an existing path."""
    temp = _write_temp_json(path, value)
    try:
        os.link(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def atomic_replace_json(path: Path, value: Any) -> None:
    """Atomically replace the deliberately mutable rolling histogram."""
    temp = _write_temp_json(path, value)
    try:
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bucket_label(value: float) -> str:
    if value <= BIN_EDGES[0]:
        return f"<={BIN_EDGES[0]:.2f}"
    for lower, upper in zip(BIN_EDGES, BIN_EDGES[1:]):
        if value <= upper:
            return f"({lower:.2f},{upper:.2f}]"
    return f">{BIN_EDGES[-1]:.2f}"


def empty_histogram() -> dict[str, int]:
    labels = [f"<={BIN_EDGES[0]:.2f}"]
    labels.extend(
        f"({lower:.2f},{upper:.2f}]"
        for lower, upper in zip(BIN_EDGES, BIN_EDGES[1:])
    )
    labels.append(f">{BIN_EDGES[-1]:.2f}")
    return {label: 0 for label in labels}


def make_histogram(values: list[float]) -> dict[str, int]:
    histogram = empty_histogram()
    for value in values:
        histogram[bucket_label(value)] += 1
    return histogram


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r} is not allowed")
    return parsed


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r} is not allowed")
        result[key] = value
    return result


def strict_json_loads(value: str | bytes) -> Any:
    return json.loads(
        value,
        parse_constant=_reject_json_constant,
        parse_float=_parse_finite_float,
        object_pairs_hook=_reject_duplicate_object_pairs,
    )


def load_cases(path: Path, raw_bytes: bytes | None = None) -> list[dict[str, Any]]:
    """Parse and validate one immutable byte snapshot of a JSONL corpus."""
    if raw_bytes is None:
        raw_bytes = path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: corpus must be strict UTF-8: {exc}") from exc
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            case = strict_json_loads(line)
        except (json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(case, dict):
            raise ValueError(f"{path}:{line_number}: case must be an object")
        unknown_case_keys = set(case) - {
            "case_id",
            "workload_class",
            "request",
            "expect",
            "measured",
        }
        if unknown_case_keys:
            raise ValueError(
                f"{path}:{line_number}: unknown case keys: {sorted(unknown_case_keys)}"
            )
        case_id = case.get("case_id")
        workload_class = case.get("workload_class")
        request_body = case.get("request")
        expectation = case.get("expect", {})
        measured = case.get("measured", True)
        if not isinstance(case_id, str) or not CASE_ID_RE.fullmatch(case_id):
            raise ValueError(
                f"{path}:{line_number}: case_id must match {CASE_ID_RE.pattern!r}"
            )
        if case_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate case_id {case_id!r}")
        if workload_class not in WORKLOAD_CLASSES:
            raise ValueError(
                f"{path}:{line_number}: workload_class must be one of {WORKLOAD_CLASSES}"
            )
        if not isinstance(request_body, dict):
            raise ValueError(f"{path}:{line_number}: request must be an object")
        if not isinstance(expectation, dict):
            raise ValueError(f"{path}:{line_number}: expect must be an object")
        if not isinstance(measured, bool):
            raise ValueError(f"{path}:{line_number}: measured must be boolean")
        if request_body.get("stream") not in (None, False):
            raise ValueError(f"{path}:{line_number}: stream must be false")
        if request_body.get("return_meta_info") not in (None, True):
            raise ValueError(f"{path}:{line_number}: return_meta_info must be true")
        minimum = request_body.get("min_tokens")
        maximum = request_body.get("max_tokens")
        if minimum is not None and (not _is_int(minimum) or minimum < 0):
            raise ValueError(f"{path}:{line_number}: min_tokens must be nonnegative")
        if maximum is not None and (not _is_int(maximum) or maximum <= 0):
            raise ValueError(f"{path}:{line_number}: max_tokens must be positive")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"{path}:{line_number}: min_tokens exceeds max_tokens")
        if "ignore_eos" in request_body and not isinstance(
            request_body["ignore_eos"], bool
        ):
            raise ValueError(f"{path}:{line_number}: ignore_eos must be boolean")
        validate_expectation(expectation, path, line_number)
        try:
            canonical_json_bytes(case)
        except (RecursionError, TypeError, UnicodeEncodeError, ValueError) as exc:
            raise ValueError(
                f"{path}:{line_number}: case is not canonical JSON: {exc}"
            ) from exc
        seen_ids.add(case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path}: no workload cases found")
    return cases


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_positive(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        converted = float(value)
    except OverflowError:
        return False
    return math.isfinite(converted) and converted > 0


def validate_expectation(expectation: dict[str, Any], path: Path, line_number: int) -> None:
    unknown = set(expectation) - EXPECT_KEYS
    if unknown:
        raise ValueError(f"{path}:{line_number}: unknown expect keys: {sorted(unknown)}")
    status_code = expectation.get("status_code", 200)
    if not _is_int(status_code) or not 100 <= status_code <= 599:
        raise ValueError(f"{path}:{line_number}: status_code must be an HTTP integer")
    if "finish_reason" in expectation and not isinstance(
        expectation["finish_reason"], str
    ):
        raise ValueError(f"{path}:{line_number}: finish_reason must be a string")
    if "content_exact" in expectation and (
        not isinstance(expectation["content_exact"], str)
        or not expectation["content_exact"]
    ):
        raise ValueError(f"{path}:{line_number}: content_exact must be nonempty")
    if "content_starts_with" in expectation and (
        not isinstance(expectation["content_starts_with"], str)
        or not expectation["content_starts_with"]
    ):
        raise ValueError(
            f"{path}:{line_number}: content_starts_with must be a nonempty string"
        )
    if "content_contains_all" in expectation:
        markers = expectation["content_contains_all"]
        if (
            not isinstance(markers, list)
            or not markers
            or not all(isinstance(item, str) and item for item in markers)
        ):
            raise ValueError(
                f"{path}:{line_number}: content_contains_all must be nonempty strings"
            )
    if "json_exact" in expectation:
        canonical_json_bytes(expectation["json_exact"])
    if "reasoning_empty" in expectation and not isinstance(
        expectation["reasoning_empty"], bool
    ):
        raise ValueError(f"{path}:{line_number}: reasoning_empty must be boolean")
    tool_call = expectation.get("tool_call")
    if tool_call is not None:
        if not isinstance(tool_call, dict) or set(tool_call) - {"name", "arguments"}:
            raise ValueError(f"{path}:{line_number}: invalid tool_call expectation")
        if not isinstance(tool_call.get("name"), str) or not tool_call["name"]:
            raise ValueError(f"{path}:{line_number}: tool_call.name must be nonempty")
        if not isinstance(tool_call.get("arguments"), dict):
            raise ValueError(
                f"{path}:{line_number}: tool_call.arguments must be an exact object oracle"
            )
        canonical_json_bytes(tool_call["arguments"])
    error = expectation.get("error")
    if status_code != 200:
        if not isinstance(error, dict) or set(error) != {
            "type",
            "code",
            "message_contains_all",
        }:
            raise ValueError(
                f"{path}:{line_number}: non-200 cases require an exact error oracle"
            )
        if not isinstance(error["type"], str) or not error["type"]:
            raise ValueError(f"{path}:{line_number}: error.type must be nonempty")
        if not _is_int(error["code"]) or error["code"] != status_code:
            raise ValueError(
                f"{path}:{line_number}: error.code must equal status_code"
            )
        markers = error["message_contains_all"]
        if (
            not isinstance(markers, list)
            or not markers
            or not all(isinstance(item, str) and item for item in markers)
        ):
            raise ValueError(
                f"{path}:{line_number}: error.message_contains_all must be nonempty strings"
            )
    elif error is not None:
        raise ValueError(f"{path}:{line_number}: error oracle requires non-200 status")
    concrete_oracle = any(
        key in expectation
        for key in (
            "content_exact",
            "content_starts_with",
            "content_contains_all",
            "json_exact",
            "tool_call",
        )
    )
    if status_code == 200 and not concrete_oracle:
        raise ValueError(
            f"{path}:{line_number}: successful cases require an exact content/tool oracle"
        )


def validate_campaign_coverage(cases: list[dict[str, Any]]) -> None:
    eligible = [
        case
        for case in cases
        if case.get("measured", True)
        if case.get("expect", {}).get("status_code", 200) == 200
        if any(
            key in case.get("expect", {})
            for key in (
                "content_exact",
                "content_starts_with",
                "content_contains_all",
                "json_exact",
                "tool_call",
            )
        )
    ]
    counts = {
        workload_class: sum(
            case["workload_class"] == workload_class for case in eligible
        )
        for workload_class in WORKLOAD_CLASSES
    }
    short = {
        workload_class: {"found": counts[workload_class], "required": minimum}
        for workload_class, minimum in CAMPAIGN_MINIMUMS.items()
        if counts[workload_class] < minimum
    }
    if short:
        raise ValueError(
            "campaign successful-oracle workload coverage is incomplete: " f"{short}"
        )
    chat_witnesses = [
        case
        for case in eligible
        if case["workload_class"] == "chat"
        and case["request"].get("min_tokens") == 512
        and case["request"].get("max_tokens") == 512
        and case["request"].get("ignore_eos") is True
        and case["expect"].get("finish_reason") == "length"
    ]
    if len(chat_witnesses) < CAMPAIGN_MINIMUMS["chat"]:
        raise ValueError(
            "campaign requires nine forced 512-token measured chat witnesses"
        )

    def is_tool_witness(case: dict[str, Any]) -> bool:
        expected = case["expect"].get("tool_call")
        tools = case["request"].get("tools")
        choice = case["request"].get("tool_choice")
        if not isinstance(expected, dict) or not isinstance(tools, list) or len(tools) != 1:
            return False
        tool = tools[0]
        function = tool.get("function") if isinstance(tool, dict) else None
        if (
            not isinstance(tool, dict)
            or tool.get("type") != "function"
            or not isinstance(function, dict)
            or function.get("name") != expected.get("name")
        ):
            return False
        return choice == "required" or (
            isinstance(choice, dict)
            and choice.get("type") == "function"
            and isinstance(choice.get("function"), dict)
            and choice["function"].get("name") == expected.get("name")
        )

    def is_json_witness(case: dict[str, Any]) -> bool:
        response_format = case["request"].get("response_format")
        schema = (
            response_format.get("json_schema")
            if isinstance(response_format, dict)
            else None
        )
        return (
            "json_exact" in case["expect"]
            and isinstance(response_format, dict)
            and response_format.get("type") == "json_schema"
            and isinstance(schema, dict)
            and schema.get("strict") is True
        )

    tool_json_cases = [
        case for case in eligible if case["workload_class"] == "tool_json"
    ]
    if sum(is_tool_witness(case) for case in tool_json_cases) < 5:
        raise ValueError("campaign requires five forced exact tool-call witnesses")
    if sum(is_json_witness(case) for case in tool_json_cases) < 5:
        raise ValueError("campaign requires five strict JSON-schema witnesses")
    if not any(
        case.get("measured", True)
        and case["workload_class"] == "long_retrieval"
        and case.get("expect", {}).get("status_code") == 400
        for case in cases
    ):
        raise ValueError("campaign requires a measured long-retrieval HTTP 400 gate")


def validate_case_models(cases: list[dict[str, Any]], model: str) -> None:
    for case in cases:
        requested = case["request"].get("model")
        if requested not in (None, model):
            raise ValueError(
                f"{case['case_id']}: request model {requested!r} does not match {model!r}"
            )


def validate_campaign_results(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    expected_drafts_per_verify: int | None = None,
) -> list[str]:
    """Return promotion-blocking errors for a completed campaign-grade run."""
    errors: list[str] = []
    result_by_id: dict[str, dict[str, Any]] = {}
    for result in results:
        case_id = result.get("case_id")
        if not isinstance(case_id, str):
            errors.append("result has no valid case_id")
            continue
        if case_id in result_by_id:
            errors.append(f"duplicate result case_id: {case_id}")
            continue
        result_by_id[case_id] = result
    expected_ids = {case["case_id"] for case in cases}
    for case_id in sorted(set(result_by_id) - expected_ids):
        errors.append(f"unexpected result case_id: {case_id}")

    successful_results: list[dict[str, Any]] = []
    for case in cases:
        result = result_by_id.get(case["case_id"])
        if result is None:
            errors.append(f"missing result case_id: {case['case_id']}")
            continue
        if result.get("workload_class") != case["workload_class"]:
            errors.append(f"workload class mismatch: {case['case_id']}")
            continue
        if not case.get("measured", True):
            continue
        expected_status = case.get("expect", {}).get("status_code", 200)
        if expected_status != 200:
            if (
                result.get("passed") is not True
                or result.get("status_code") != expected_status
            ):
                errors.append(f"required error gate failed: {case['case_id']}")
            continue
        if result.get("passed") is not True or result.get("status_code") != 200:
            errors.append(f"successful-oracle case failed: {case['case_id']}")
            continue
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            errors.append(f"successful-oracle case lacks metrics: {case['case_id']}")
            continue
        if (
            not _is_int(metrics.get("prompt_tokens"))
            or metrics["prompt_tokens"] < 0
            or not _is_int(metrics.get("completion_tokens"))
            or metrics["completion_tokens"] <= 0
        ):
            errors.append(f"successful-oracle case has invalid token metrics: {case['case_id']}")
            continue
        latency = metrics.get("e2e_latency_s")
        throughput = metrics.get("e2e_output_tok_s")
        if not _is_finite_positive(latency) or not _is_finite_positive(throughput):
            errors.append(f"successful-oracle case lacks valid latency: {case['case_id']}")
            continue
        if case["workload_class"] == "chat" and metrics["completion_tokens"] < 512:
            errors.append(f"chat witness was shorter than 512 tokens: {case['case_id']}")
            continue
        if metrics.get("num_retractions") != 0:
            errors.append(f"successful-oracle case was retracted: {case['case_id']}")
            continue
        if (
            expected_drafts_per_verify is not None
            and metrics.get("drafts_per_verify") != expected_drafts_per_verify
        ):
            errors.append(
                f"unexpected draft depth for {case['case_id']}: "
                f"{metrics.get('drafts_per_verify')!r} != {expected_drafts_per_verify}"
            )
            continue
        successful_results.append(result)

    counts = {
        workload_class: sum(
            result["workload_class"] == workload_class
            for result in successful_results
        )
        for workload_class in WORKLOAD_CLASSES
    }
    for workload_class, minimum in CAMPAIGN_MINIMUMS.items():
        if counts[workload_class] < minimum:
            errors.append(
                f"metric-bearing passed {workload_class} cases: "
                f"{counts[workload_class]} < {minimum}"
            )

    gate_119_ids = {
        result["case_id"]
        for result in successful_results
        if result["workload_class"] == "long_retrieval"
        and result["metrics"]["prompt_tokens"] >= 119_424
        and result["metrics"]["prompt_tokens"] < 130_500
        and result["metrics"]["completion_tokens"] >= 512
    }
    gate_130_ids = {
        result["case_id"]
        for result in successful_results
        if result["workload_class"] == "long_retrieval"
        and result["metrics"]["prompt_tokens"] >= 130_500
    }
    if not gate_119_ids:
        errors.append("missing passed 119,424+ prompt / 512-token long-retrieval gate")
    if not gate_130_ids:
        errors.append("missing passed 130,500+ prompt long-retrieval gate")
    return errors


def _parse_tool_arguments(value: Any) -> Any:
    if isinstance(value, str):
        return strict_json_loads(value)
    return value


def grade_response(
    case: dict[str, Any], status_code: int, response: dict[str, Any] | None
) -> list[str]:
    expectation = case.get("expect", {})
    errors: list[str] = []
    expected_status = int(expectation.get("status_code", 200))
    if status_code != expected_status:
        errors.append(f"status {status_code}, expected {expected_status}")
        return errors
    if expected_status != 200:
        if not isinstance(response, dict):
            return ["error response is not a JSON object"]
        expected_error = expectation["error"]
        if response.get("type") != expected_error["type"]:
            errors.append("error type mismatch")
        if response.get("code") != expected_error["code"]:
            errors.append("error code mismatch")
        message = response.get("message")
        if not isinstance(message, str):
            errors.append("error message is not a string")
        else:
            for marker in expected_error["message_contains_all"]:
                if marker not in message:
                    errors.append(f"missing error message marker {marker!r}")
        return errors
    if not isinstance(response, dict):
        return ["response is not a JSON object"]
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return ["response must contain exactly one choice"]
    choice = choices[0]
    if not isinstance(choice, dict):
        return ["choice must be an object"]
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        return ["choice.message must be an object"]
    raw_content = message.get("content")
    if raw_content is not None and not isinstance(raw_content, str):
        errors.append("message content must be a string or null")
    content = raw_content if isinstance(raw_content, str) else ""
    if "finish_reason" in expectation and choice.get("finish_reason") != expectation["finish_reason"]:
        errors.append(
            f"finish_reason {choice.get('finish_reason')!r}, expected {expectation['finish_reason']!r}"
        )
    if "content_exact" in expectation and content != expectation["content_exact"]:
        errors.append("content_exact mismatch")
    if "content_starts_with" in expectation and not content.startswith(
        expectation["content_starts_with"]
    ):
        errors.append("content_starts_with mismatch")
    for marker in expectation.get("content_contains_all", []):
        if marker not in content:
            errors.append(f"missing content marker {marker!r}")
    if "json_exact" in expectation:
        try:
            parsed_content = strict_json_loads(content)
            parsed_bytes = canonical_json_bytes(parsed_content)
        except (
            json.JSONDecodeError,
            RecursionError,
            UnicodeEncodeError,
            ValueError,
        ):
            errors.append("content is not strict JSON")
        else:
            if parsed_bytes != canonical_json_bytes(expectation["json_exact"]):
                errors.append("json_exact mismatch")
    if expectation.get("reasoning_empty", False) and message.get("reasoning_content") not in (None, ""):
        errors.append("reasoning_content was not empty")
    expected_tool = expectation.get("tool_call")
    if expected_tool is not None:
        if content:
            errors.append("tool response contained unexpected visible content")
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            errors.append("tool_calls must be a list")
        elif len(tool_calls) != 1:
            errors.append(
                f"expected exactly one total tool call, got {len(tool_calls)}"
            )
        else:
            tool_item = tool_calls[0]
            if not isinstance(tool_item, dict) or tool_item.get("type") != "function":
                errors.append("tool call type mismatch")
            function = tool_item.get("function") if isinstance(tool_item, dict) else None
            function = function if isinstance(function, dict) else {}
            if function.get("name") != expected_tool.get("name"):
                errors.append("tool name mismatch")
            else:
                try:
                    actual_args = _parse_tool_arguments(function.get("arguments"))
                    actual_bytes = canonical_json_bytes(actual_args)
                except (
                    json.JSONDecodeError,
                    RecursionError,
                    TypeError,
                    UnicodeEncodeError,
                    ValueError,
                ):
                    errors.append("tool arguments are not strict JSON")
                else:
                    if not isinstance(actual_args, dict) or actual_bytes != canonical_json_bytes(
                        expected_tool["arguments"]
                    ):
                        errors.append("tool arguments mismatch")
    elif message.get("tool_calls"):
        errors.append("non-tool response contained unexpected tool calls")
    return errors


def _strict_counter(container: dict[str, Any], name: str, *, positive: bool) -> int:
    value = container.get(name)
    if not _is_int(value) or value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _finite_optional(container: dict[str, Any], name: str) -> float | None:
    value = container.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def extract_metrics(response: dict[str, Any]) -> dict[str, Any]:
    choice = response["choices"][0]
    meta = choice.get("meta_info") or {}
    usage = response.get("usage") or {}
    usage_completion = _strict_counter(usage, "completion_tokens", positive=True)
    meta_completion = _strict_counter(meta, "completion_tokens", positive=True)
    if usage_completion != meta_completion:
        raise ValueError("usage and meta completion token counts disagree")
    completion_tokens = usage_completion
    usage_prompt = _strict_counter(usage, "prompt_tokens", positive=False)
    meta_prompt = _strict_counter(meta, "prompt_tokens", positive=False)
    if usage_prompt != meta_prompt:
        raise ValueError("usage and meta prompt token counts disagree")
    prompt_tokens = usage_prompt
    num_retractions = _strict_counter(meta, "num_retractions", positive=False)
    if num_retractions != 0:
        raise ValueError("speculative telemetry is invalid after a scheduler retraction")
    verify_count = _strict_counter(meta, "spec_verify_ct", positive=True)
    correct_drafts = _strict_counter(meta, "spec_num_correct_drafts", positive=False)
    proposed_drafts = _strict_counter(meta, "spec_num_proposed_drafts", positive=True)
    if correct_drafts > proposed_drafts:
        raise ValueError("correct drafts exceed proposed drafts")
    if proposed_drafts % verify_count:
        raise ValueError("proposed drafts are inconsistent with verify count")
    # The output can contain one prefill token plus, for each verification,
    # one target/bonus token and the drafts that were actually accepted.
    # Grammar/EOS may shorten the run, so only this fail-closed upper bound is
    # enforced against accepted drafts.  A successful verification still emits
    # one target/bonus token, in addition to the initial prefill token.
    min_completion_tokens = 1 + verify_count
    max_completion_tokens = 1 + verify_count + correct_drafts
    if completion_tokens < min_completion_tokens:
        raise ValueError(
            "completion tokens are below the speculative verification lower bound"
        )
    if completion_tokens > max_completion_tokens:
        raise ValueError(
            "completion tokens exceed the speculative verification upper bound"
        )
    computed_length = completion_tokens / verify_count
    computed_rate = correct_drafts / proposed_drafts
    drafts_per_verify = proposed_drafts // verify_count
    raw_histogram = meta.get("spec_correct_drafts_histogram")
    correct_drafts_histogram = None
    if raw_histogram is not None:
        if not isinstance(raw_histogram, list) or not all(
            _is_int(item) and item >= 0 for item in raw_histogram
        ):
            raise ValueError("spec_correct_drafts_histogram must be nonnegative integers")
        if sum(raw_histogram) != verify_count:
            raise ValueError("speculative histogram count disagrees with verify count")
        if sum(index * count for index, count in enumerate(raw_histogram)) != correct_drafts:
            raise ValueError("speculative histogram weight disagrees with accepted drafts")
        if any(
            count and index > drafts_per_verify
            for index, count in enumerate(raw_histogram)
        ):
            raise ValueError("speculative histogram exceeds the configured draft depth")
        correct_drafts_histogram = raw_histogram
    if not math.isfinite(computed_length) or computed_length <= 0:
        raise ValueError("computed accept length is invalid")
    reported_length = _finite_optional(meta, "spec_accept_length")
    reported_rate = meta.get("spec_accept_rate")
    if reported_rate is not None:
        if isinstance(reported_rate, bool) or not isinstance(reported_rate, (int, float)):
            raise ValueError("spec_accept_rate must be numeric")
        reported_rate = float(reported_rate)
        if not math.isfinite(reported_rate) or not 0.0 <= reported_rate <= 1.0:
            raise ValueError("spec_accept_rate must be finite in [0,1]")
    if reported_length is not None and not math.isclose(
        reported_length, computed_length, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise ValueError("reported accept length disagrees with raw counters")
    if reported_rate is not None and not math.isclose(
        reported_rate, computed_rate, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise ValueError("reported accept rate disagrees with raw counters")
    e2e_latency = _finite_optional(meta, "e2e_latency")
    if e2e_latency is None:
        raise ValueError("e2e_latency is required for completed responses")
    e2e_output_tok_s = completion_tokens / e2e_latency
    if not math.isfinite(e2e_output_tok_s) or e2e_output_tok_s <= 0:
        raise ValueError("derived end-to-end throughput is not finite and positive")
    decode_throughput = _finite_optional(meta, "decode_throughput")
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "num_retractions": num_retractions,
        "spec_verify_ct": verify_count,
        "spec_num_correct_drafts": correct_drafts,
        "spec_num_proposed_drafts": proposed_drafts,
        "drafts_per_verify": drafts_per_verify,
        "spec_correct_drafts_histogram": correct_drafts_histogram,
        "spec_accept_length": computed_length,
        "spec_accept_rate": computed_rate,
        "e2e_latency_s": e2e_latency,
        "e2e_output_tok_s": e2e_output_tok_s,
        "decode_tok_s": decode_throughput,
    }


def validate_metrics_against_request(
    request_body: dict[str, Any], expectation: dict[str, Any], metrics: dict[str, Any]
) -> None:
    completion_tokens = metrics["completion_tokens"]
    minimum = request_body.get("min_tokens")
    maximum = request_body.get("max_tokens")
    if minimum is not None and completion_tokens < minimum:
        raise ValueError("completion tokens are below request min_tokens")
    if maximum is not None and completion_tokens > maximum:
        raise ValueError("completion tokens exceed request max_tokens")
    if (
        expectation.get("finish_reason") == "length"
        and maximum is not None
        and completion_tokens != maximum
    ):
        raise ValueError("length-finished response did not consume max_tokens")


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for workload_class in (*WORKLOAD_CLASSES, "all"):
        selected = [
            item
            for item in results
            if item.get("measured", True)
            and (workload_class == "all" or item["workload_class"] == workload_class)
        ]
        if not selected:
            continue
        metrics = [item["metrics"] for item in selected if item.get("metrics")]
        total_completion = sum(item["completion_tokens"] for item in metrics)
        total_verify = sum(item["spec_verify_ct"] for item in metrics)
        total_correct = sum(item["spec_num_correct_drafts"] for item in metrics)
        total_proposed = sum(item["spec_num_proposed_drafts"] for item in metrics)
        accept_lengths = [item["spec_accept_length"] for item in metrics]
        e2e_tps = [
            item["e2e_output_tok_s"]
            for item in metrics
            if item["e2e_output_tok_s"] is not None
        ]
        decode_tps = [
            item["decode_tok_s"] for item in metrics if item["decode_tok_s"] is not None
        ]
        summary[workload_class] = {
            "requests": len(selected),
            "passed": sum(bool(item["passed"]) for item in selected),
            "failed": sum(not bool(item["passed"]) for item in selected),
            "metrics_requests": len(metrics),
            "missing_metrics": len(selected) - len(metrics),
            "weighted_accept_length": total_completion / total_verify if total_verify else None,
            "weighted_accept_rate": total_correct / total_proposed if total_proposed else None,
            "accept_length_p10": percentile(accept_lengths, 0.10),
            "accept_length_p50": percentile(accept_lengths, 0.50),
            "accept_length_p90": percentile(accept_lengths, 0.90),
            "accept_length_histogram": make_histogram(accept_lengths),
            "e2e_output_tok_s_p50": percentile(e2e_tps, 0.50),
            "decode_tok_s_p50": percentile(decode_tps, 0.50),
            "completion_tokens": total_completion,
            "verify_count": total_verify,
            "correct_drafts": total_correct,
            "proposed_drafts": total_proposed,
        }
    return summary


def update_rolling(
    current: dict[str, Any] | None,
    arm: str,
    results: list[dict[str, Any]],
    updated_at: str,
) -> dict[str, Any]:
    if current is None:
        current = {"schema": ROLLING_SCHEMA, "arms": {}}
    if current.get("schema") != ROLLING_SCHEMA or not isinstance(current.get("arms"), dict):
        raise ValueError("rolling histogram has an unsupported schema")
    arm_state = current["arms"].setdefault(arm, {"workload_classes": {}})
    for result in results:
        if not result.get("measured", True):
            continue
        class_state = arm_state["workload_classes"].setdefault(
            result["workload_class"],
            {
                "requests": 0,
                "passed": 0,
                "failed": 0,
                "metrics_requests": 0,
                "missing_metrics": 0,
                "completion_tokens": 0,
                "verify_count": 0,
                "correct_drafts": 0,
                "proposed_drafts": 0,
                "accept_length_histogram": empty_histogram(),
                "recent": [],
            },
        )
        class_state["requests"] += 1
        class_state["passed"] += int(bool(result["passed"]))
        class_state["failed"] += int(not bool(result["passed"]))
        metrics = result.get("metrics")
        if not metrics:
            class_state["missing_metrics"] += 1
            class_state["recent"].append(
                {
                    "at": result["completed_at"],
                    "case_id": result["case_id"],
                    "passed": result["passed"],
                    "accept_length": None,
                    "accept_rate": None,
                    "e2e_output_tok_s": None,
                }
            )
            class_state["recent"] = class_state["recent"][-64:]
            continue
        class_state["metrics_requests"] += 1
        class_state["completion_tokens"] += metrics["completion_tokens"]
        class_state["verify_count"] += metrics["spec_verify_ct"]
        class_state["correct_drafts"] += metrics["spec_num_correct_drafts"]
        class_state["proposed_drafts"] += metrics["spec_num_proposed_drafts"]
        class_state["accept_length_histogram"][bucket_label(metrics["spec_accept_length"])] += 1
        class_state["recent"].append(
            {
                "at": result["completed_at"],
                "case_id": result["case_id"],
                "passed": result["passed"],
                "accept_length": metrics["spec_accept_length"],
                "accept_rate": metrics["spec_accept_rate"],
                "e2e_output_tok_s": metrics["e2e_output_tok_s"],
            }
        )
        class_state["recent"] = class_state["recent"][-64:]
        class_state["weighted_accept_length"] = (
            class_state["completion_tokens"] / class_state["verify_count"]
        )
        class_state["weighted_accept_rate"] = (
            class_state["correct_drafts"] / class_state["proposed_drafts"]
            if class_state["proposed_drafts"]
            else None
        )
    current["updated_at"] = updated_at
    return current


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, "redirect refused", headers, fp
        )


def build_http_opener():
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}), NoRedirect()
    )


def preflight_output_path(path: Path) -> None:
    """Prove sibling create/fsync/link/replace/read/unlink before inference."""
    probe = path.with_name(
        f".{path.name}.preflight.{os.getpid()}.{secrets.token_hex(8)}"
    )
    try:
        atomic_create_json(probe, {"phase": "create"})
        atomic_replace_json(probe, {"phase": "replace"})
        observed = strict_json_loads(probe.read_text(encoding="utf-8"))
        if observed != {"phase": "replace"}:
            raise ValueError(f"output preflight readback failed for {path}")
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass


def assert_loopback(base_url: str, allow_non_loopback: bool) -> None:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("base URL must use http or https")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain credentials, query, or fragment")
    hostname = parsed.hostname
    if hostname not in {"127.0.0.1", "localhost", "::1"} and not allow_non_loopback:
        raise ValueError("refusing non-loopback endpoint without --allow-non-loopback")


def fetch_server_info(
    base_url: str, timeout: float, api_key: str | None
) -> dict[str, Any]:
    parsed_url = urllib.parse.urlparse(base_url)
    info_url = urllib.parse.urlunparse(
        (parsed_url.scheme, parsed_url.netloc, "/get_server_info", "", "", "")
    )
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(info_url, headers=headers, method="GET")
    try:
        with build_http_opener().open(request, timeout=timeout) as response:
            if response.status != 200:
                raise ValueError(f"server-info status was {response.status}")
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise ValueError(f"server-info status was {exc.code}") from exc
    except (
        http.client.HTTPException,
        urllib.error.URLError,
        TimeoutError,
        OSError,
    ) as exc:
        raise ValueError(f"server-info transport failure: {type(exc).__name__}") from exc
    try:
        info = strict_json_loads(raw)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError(f"server-info response was not strict JSON: {exc}") from exc
    if not isinstance(info, dict):
        raise ValueError("server-info response was not an object")
    return info


def bind_runtime_info(
    info: dict[str, Any],
    *,
    model: str,
    expected_context_length: int,
    expected_max_total_num_tokens: int,
    expected_max_running_requests: int,
    expected_kv_cache_dtype: str,
    expected_draft_kv_cache_dtype: str,
    expected_drafts_per_verify: int,
) -> dict[str, Any]:
    selected = {
        "served_model_name": info.get("served_model_name"),
        "context_length": info.get("context_length"),
        "max_total_num_tokens": info.get("max_total_num_tokens"),
        "max_running_requests": info.get("max_running_requests"),
        "kv_cache_dtype": info.get("kv_cache_dtype"),
        "speculative_draft_kv_cache_dtype_raw": info.get(
            "speculative_draft_kv_cache_dtype"
        ),
        "speculative_algorithm": info.get("speculative_algorithm"),
        "speculative_num_steps": info.get("speculative_num_steps"),
        "speculative_eagle_topk": info.get("speculative_eagle_topk"),
        "speculative_num_draft_tokens": info.get("speculative_num_draft_tokens"),
        "speculative_use_rejection_sampling": info.get(
            "speculative_use_rejection_sampling"
        ),
        "startup_time": info.get("startup_time"),
    }
    selected["effective_draft_kv_cache_dtype"] = (
        selected["speculative_draft_kv_cache_dtype_raw"]
        or selected["kv_cache_dtype"]
    )
    expected = {
        "served_model_name": model,
        "context_length": expected_context_length,
        "max_total_num_tokens": expected_max_total_num_tokens,
        "max_running_requests": expected_max_running_requests,
        "kv_cache_dtype": expected_kv_cache_dtype,
        "effective_draft_kv_cache_dtype": expected_draft_kv_cache_dtype,
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": expected_drafts_per_verify,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": expected_drafts_per_verify + 1,
        "speculative_use_rejection_sampling": True,
    }
    mismatches = {
        key: {"expected": value, "actual": selected.get(key)}
        for key, value in expected.items()
        if selected.get(key) != value
    }
    if mismatches:
        raise ValueError(f"live runtime identity mismatch: {mismatches}")
    if not isinstance(selected["startup_time"], dict) or not selected["startup_time"]:
        raise ValueError("live runtime identity has no startup-time fingerprint")
    selected["identity_sha256"] = sha256_bytes(canonical_json_bytes(selected))
    return selected


def run_case(
    case: dict[str, Any],
    endpoint: str,
    model: str,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    body = dict(case["request"])
    if body.get("model") not in (None, model):
        raise ValueError(f"{case['case_id']}: request model does not match --model")
    body["model"] = model
    body["stream"] = False
    body["return_meta_info"] = True
    encoded = canonical_json_bytes(body)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, data=encoded, headers=headers, method="POST")
    started = time.monotonic()
    status_code = 0
    raw_response = b""
    opener = build_http_opener()
    transport_error = None
    try:
        with opener.open(request, timeout=timeout) as response:
            status_code = response.status
            raw_response = response.read()
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        try:
            raw_response = exc.read()
        except (
            http.client.HTTPException,
            TimeoutError,
            OSError,
        ) as read_exc:
            transport_error = f"transport failure: {type(read_exc).__name__}"
    except (http.client.HTTPException, urllib.error.URLError, TimeoutError, OSError) as exc:
        transport_error = f"transport failure: {type(exc).__name__}"
    elapsed = time.monotonic() - started
    response_json: dict[str, Any] | None = None
    if raw_response:
        try:
            parsed = strict_json_loads(raw_response)
            response_json = parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, RecursionError, ValueError):
            response_json = None
    errors = (
        [transport_error]
        if transport_error is not None
        else grade_response(case, status_code, response_json)
    )
    metrics = None
    if status_code == 200 and response_json is not None:
        try:
            candidate_metrics = extract_metrics(response_json)
            validate_metrics_against_request(
                body, case.get("expect", {}), candidate_metrics
            )
            metrics = candidate_metrics
        except (
            ArithmeticError,
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            errors.append(str(exc))
    choices = (response_json or {}).get("choices")
    choice = (
        choices[0]
        if isinstance(choices, list)
        and len(choices) == 1
        and isinstance(choices[0], dict)
        else {}
    )
    message = choice.get("message") or {}
    message = message if isinstance(message, dict) else {}
    content = message.get("content") or ""
    content = content if isinstance(content, str) else ""
    return {
        "case_id": case["case_id"],
        "workload_class": case["workload_class"],
        "measured": case.get("measured", True),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": not errors,
        "status_code": status_code,
        "elapsed_s": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "answer_chars": len(content),
        "metrics": metrics,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="JSONL workload cases")
    parser.add_argument("--output", type=Path, required=True, help="immutable run receipt")
    parser.add_argument("--rolling", type=Path, help="small cumulative histogram JSON")
    parser.add_argument("--arm", required=True, help="stable arm id, e.g. eagle-1-1-2-bf16kv")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--api-key-env",
        help="explicit environment variable containing this endpoint's API key",
    )
    parser.add_argument("--allow-non-loopback", action="store_true")
    parser.add_argument(
        "--expected-drafts-per-verify",
        type=int,
        choices=(1, 2, 3),
        help="bind campaign telemetry to A0/A1/A2 draft depth",
    )
    parser.add_argument(
        "--campaign-grade",
        action="store_true",
        help="require measured, metric-bearing promotion gates for all workloads",
    )
    parser.add_argument("--expected-context-length", type=int)
    parser.add_argument("--expected-max-total-num-tokens", type=int)
    parser.add_argument("--expected-max-running-requests", type=int)
    parser.add_argument(
        "--expected-kv-cache-dtype", choices=("bfloat16", "fp8_e4m3")
    )
    parser.add_argument(
        "--expected-draft-kv-cache-dtype", choices=("bfloat16", "fp8_e4m3")
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not ARM_RE.fullmatch(args.arm):
        raise ValueError("--arm contains unsupported characters")
    cases_bytes = args.cases.read_bytes()
    cases = load_cases(args.cases, cases_bytes)
    cases_sha256 = sha256_bytes(cases_bytes)
    validate_case_models(cases, args.model)
    if args.campaign_grade:
        required_runtime_args = {
            "--expected-drafts-per-verify": args.expected_drafts_per_verify,
            "--expected-context-length": args.expected_context_length,
            "--expected-max-total-num-tokens": args.expected_max_total_num_tokens,
            "--expected-max-running-requests": args.expected_max_running_requests,
            "--expected-kv-cache-dtype": args.expected_kv_cache_dtype,
            "--expected-draft-kv-cache-dtype": args.expected_draft_kv_cache_dtype,
        }
        missing_runtime_args = [
            name for name, value in required_runtime_args.items() if value is None
        ]
        if missing_runtime_args:
            raise ValueError(
                "--campaign-grade requires runtime identity arguments: "
                f"{missing_runtime_args}"
            )
        validate_campaign_coverage(cases)
    if args.validate_only:
        print(f"WORKLOAD_CASES_OK count={len(cases)}")
        return 0
    assert_loopback(args.base_url, args.allow_non_loopback)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.output}")
    preflight_output_path(args.output)
    if args.rolling:
        preflight_output_path(args.rolling)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(f"API key environment variable is unset: {args.api_key_env}")
    runtime_info_start = None
    if args.campaign_grade:
        runtime_info_start = bind_runtime_info(
            fetch_server_info(args.base_url, args.timeout, api_key),
            model=args.model,
            expected_context_length=args.expected_context_length,
            expected_max_total_num_tokens=args.expected_max_total_num_tokens,
            expected_max_running_requests=args.expected_max_running_requests,
            expected_kv_cache_dtype=args.expected_kv_cache_dtype,
            expected_draft_kv_cache_dtype=args.expected_draft_kv_cache_dtype,
            expected_drafts_per_verify=args.expected_drafts_per_verify,
        )
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results = [
        run_case(case, endpoint, args.model, args.timeout, api_key) for case in cases
    ]
    runtime_info_end = None
    runtime_identity_error = None
    if args.campaign_grade:
        try:
            runtime_info_end = bind_runtime_info(
                fetch_server_info(args.base_url, args.timeout, api_key),
                model=args.model,
                expected_context_length=args.expected_context_length,
                expected_max_total_num_tokens=args.expected_max_total_num_tokens,
                expected_max_running_requests=args.expected_max_running_requests,
                expected_kv_cache_dtype=args.expected_kv_cache_dtype,
                expected_draft_kv_cache_dtype=args.expected_draft_kv_cache_dtype,
                expected_drafts_per_verify=args.expected_drafts_per_verify,
            )
            if (
                runtime_info_end["identity_sha256"]
                != runtime_info_start["identity_sha256"]
            ):
                runtime_identity_error = "live runtime identity changed during campaign"
        except Exception as exc:
            runtime_identity_error = (
                f"post-run runtime identity failed: {type(exc).__name__}: {exc}"
            )
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "arm": args.arm,
        "model": args.model,
        "cases_sha256": cases_sha256,
        "campaign_grade": args.campaign_grade,
        "expected_drafts_per_verify": args.expected_drafts_per_verify,
        "runtime_info_start": runtime_info_start,
        "runtime_info_end": runtime_info_end,
        "runtime_identity_error": runtime_identity_error,
        "started_at": started_at,
        "completed_at": completed_at,
        "summary": summarize_results(results),
        "results": results,
    }
    campaign_errors = (
        validate_campaign_results(
            cases, results, args.expected_drafts_per_verify
        )
        if args.campaign_grade
        else None
    )
    if campaign_errors is not None and runtime_identity_error:
        campaign_errors.append(runtime_identity_error)
    receipt["campaign_errors"] = campaign_errors
    receipt["case_results_passed"] = all(result["passed"] for result in results)
    receipt["campaign_passed"] = (
        not campaign_errors if args.campaign_grade else None
    )
    receipt["passed"] = receipt["case_results_passed"] and (
        receipt["campaign_passed"] is not False
    )
    receipt["attempts"] = len(results)
    receipt["passed_count"] = sum(bool(result["passed"]) for result in results)
    receipt["failed_count"] = sum(not bool(result["passed"]) for result in results)
    receipt["missing_metrics_count"] = sum(
        result.get("metrics") is None for result in results
    )
    receipt["eligible_missing_metrics_count"] = sum(
        result.get("measured", True)
        and result.get("status_code") == 200
        and result.get("metrics") is None
        for result in results
    )
    receipt["measured_attempts"] = sum(
        result.get("measured", True) for result in results
    )
    receipt["warmup_attempts"] = sum(
        not result.get("measured", True) for result in results
    )
    receipt["endpoint_scope"] = (
        "loopback"
        if urllib.parse.urlparse(args.base_url).hostname in {"127.0.0.1", "localhost", "::1"}
        else "non-loopback"
    )
    atomic_create_json(args.output, receipt)
    if args.rolling:
        current = None
        if args.rolling.exists():
            current = strict_json_loads(args.rolling.read_text(encoding="utf-8"))
        rolling = update_rolling(current, args.arm, results, completed_at)
        atomic_replace_json(args.rolling, rolling)
    passed = bool(receipt["passed"])
    print(
        f"MTP_WORKLOAD_HISTOGRAM_{'OK' if passed else 'FAILED'} "
        f"passed={sum(result['passed'] for result in results)}/{len(results)} "
        f"receipt={args.output}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # fail closed with one concise diagnostic
        print(f"MTP_WORKLOAD_HISTOGRAM_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
