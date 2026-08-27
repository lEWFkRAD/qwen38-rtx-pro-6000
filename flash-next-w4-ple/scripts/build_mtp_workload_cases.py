#!/usr/bin/env python3
"""Build the deterministic MTP workload/Pareto JSONL corpus.

The two long-context cases are tokenizer-calibrated against the exact local
checkpoint.  The output is immutable: an existing path is never replaced.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


FILLER = (
    "Archive note: cedar crates arrived before noon and were counted beside the east window. "
    "The clerk checked the blue ledger, closed the brass latch, and continued to the next ordinary record.\n"
    "Field report: light rain crossed the valley while the survey team measured the old stone bridge. "
    "Nothing unusual was found, so the map was folded and returned to the cabinet.\n"
    "Workshop log: copper wire, linen cord, and three glass jars were placed on the center bench. "
    "After inspection, every item was labeled and the room was left in normal condition.\n"
)


def canonical_line(value: Any) -> bytes:
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


def atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def prompt_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, (list, tuple)):
        raise TypeError(
            f"unsupported chat-template token container: {type(encoded).__name__}"
        )
    if len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    if any(isinstance(item, (list, tuple)) for item in encoded):
        raise ValueError("chat-template token output has unexpected nested batches")
    return len(encoded)


def build_long_messages(
    tokenizer: Any,
    target: int,
    codes: tuple[str, str, str],
    *,
    long_decode: bool,
) -> tuple[list[dict[str, str]], int, str]:
    system = "Follow the archive audit instructions precisely. Do not reveal analysis."
    prefix = (
        "You are auditing a very long archive. Three records contain SECRET_CODE values. "
        "Ignore all ordinary records and retain each exact value.\n\n"
    )
    needles = [
        f"\nSPECIAL EARLY RECORD -- SECRET_CODE={codes[0]} -- retain this exact value.\n",
        f"\nSPECIAL MIDDLE RECORD -- SECRET_CODE={codes[1]} -- retain this exact value.\n",
        f"\nSPECIAL LATE RECORD -- SECRET_CODE={codes[2]} -- retain this exact value.\n",
    ]
    expected = "|".join(codes)
    if long_decode:
        suffix = (
            "\nEND OF ARCHIVE. Begin your answer with the early, middle, and late "
            "SECRET_CODE values in that order, separated by vertical bars. Then write a detailed "
            "neutral operational report about archive handling. Do not stop early."
        )
    else:
        suffix = (
            "\nEND OF ARCHIVE. Return the early, middle, and late SECRET_CODE values "
            "in that order separated by vertical bars. Return only the three codes."
        )

    def make(repeats: int) -> list[dict[str, str]]:
        early = max(1, int(repeats * 0.05))
        middle = max(1, int(repeats * 0.45))
        late = max(1, int(repeats * 0.45))
        tail = max(1, repeats - early - middle - late)
        content = (
            prefix
            + FILLER * early
            + needles[0]
            + FILLER * middle
            + needles[1]
            + FILLER * late
            + needles[2]
            + FILLER * tail
            + suffix
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    counts: dict[int, int] = {}

    def count(repeats: int) -> int:
        if repeats not in counts:
            counts[repeats] = prompt_tokens(tokenizer, make(repeats))
        return counts[repeats]

    low = 4
    low_count = count(low)
    if low_count > target:
        raise ValueError(f"minimum long prompt exceeds target {target}: {low_count}")
    sample = 8
    sample_count = count(sample)
    tokens_per_repeat = (sample_count - low_count) / (sample - low)
    if not math.isfinite(tokens_per_repeat) or tokens_per_repeat <= 0:
        raise ValueError("long-prompt token count did not increase monotonically")

    estimate = max(
        low + 1,
        int(low + (target - low_count) / tokens_per_repeat),
    )
    estimate_count = count(estimate)
    if estimate_count <= target:
        low = estimate
        projected = math.ceil((target - estimate_count) / tokens_per_repeat)
        high = max(low + 1, low + projected + 2)
        while count(high) <= target:
            low = high
            high = max(high + 4, high * 2)
    else:
        high = estimate
        projected = math.ceil((estimate_count - target) / tokens_per_repeat)
        low = max(4, estimate - projected - 2)
        while count(low) > target:
            high = low
            low = max(4, (4 + low) // 2)
            if low == 4 and count(low) > target:
                raise ValueError(f"cannot bracket long prompt below target {target}")

    while low + 1 < high:
        middle = (low + high) // 2
        if count(middle) <= target:
            low = middle
        else:
            high = middle
    messages = make(low)
    return messages, count(low), expected


def validate_long_prompt_window(case_id: str, count: int, max_tokens: int) -> None:
    if count + max_tokens > 131_072:
        raise ValueError(
            f"{case_id} prompt plus completion budget exceeds 131,072: "
            f"{count}+{max_tokens}"
        )
    if case_id == "long-119k" and not 119_424 <= count < 130_500:
        raise ValueError(f"119K case is outside the qualified window: {count}")
    if case_id == "long-130k" and count < 130_500:
        raise ValueError(f"130.5K case is below the qualified boundary: {count}")


def validate_overlimit_prompt(count: int) -> None:
    if count <= 131_072:
        raise ValueError(f"over-limit case did not exceed 131,072 tokens: {count}")


def common_request(messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    return {
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def build_short_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index in range(11):
        measured = index >= 2
        marker = f"MTPCHAT{index:02d}OK"
        request = common_request(
            [
                {
                    "role": "user",
                    "content": (
                        f"Begin with exactly {marker} on the first line. Then write a "
                        "neutral numbered sequence of concise field notes until generation stops."
                    ),
                }
            ],
            512,
        )
        request.update(
            {
                "min_tokens": 512,
                "ignore_eos": True,
                "seed": 910_000 + index,
            }
        )
        cases.append(
            {
                "case_id": f"chat-{index:02d}",
                "workload_class": "chat",
                "measured": measured,
                "request": request,
                "expect": {
                    "status_code": 200,
                    "finish_reason": "length",
                    "content_contains_all": [marker],
                    "reasoning_empty": True,
                },
            }
        )

    math_cases = [
        ("37 multiplied by 19", "703"),
        ("991 minus 347", "644"),
        ("125 plus 876", "1001"),
        ("144 divided by 12", "12"),
        ("17 squared", "289"),
        ("the remainder when 1000 is divided by 37", "1"),
        ("23 multiplied by 41", "943"),
        ("2048 divided by 16", "128"),
        ("the sum of 111, 222, and 333", "666"),
        ("4096 minus 1337", "2759"),
    ]
    for index, (expression, answer) in enumerate(math_cases):
        cases.append(
            {
                "case_id": f"math-{index:02d}",
                "workload_class": "math",
                "request": common_request(
                    [
                        {
                            "role": "user",
                            "content": f"Return only the integer result of {expression}.",
                        }
                    ],
                    32,
                ),
                "expect": {
                    "status_code": 200,
                    "finish_reason": "stop",
                    "content_exact": answer,
                    "reasoning_empty": True,
                },
            }
        )

    tool_cases = [
        ("multiply", "Multiply two integers", {"a": 37, "b": 19}),
        ("add", "Add two integers", {"a": 125, "b": 876}),
        ("subtract", "Subtract b from a", {"a": 991, "b": 347}),
        ("power", "Raise base to a nonnegative integer exponent", {"base": 17, "exponent": 2}),
        ("modulo", "Return the integer remainder a modulo b", {"a": 1000, "b": 37}),
    ]
    for index, (name, description, arguments) in enumerate(tool_cases):
        properties = {key: {"type": "integer"} for key in arguments}
        request = common_request(
            [
                {
                    "role": "user",
                    "content": f"Call {name} with exactly these integer arguments: {arguments}.",
                }
            ],
            128,
        )
        request.update(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": description,
                            "parameters": {
                                "type": "object",
                                "properties": properties,
                                "required": list(arguments),
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": name}},
            }
        )
        cases.append(
            {
                "case_id": f"tool-{index:02d}",
                "workload_class": "tool_json",
                "request": request,
                "expect": {
                    "status_code": 200,
                    "finish_reason": "tool_calls",
                    "tool_call": {"name": name, "arguments": arguments},
                    "reasoning_empty": True,
                },
            }
        )

    for index, value in enumerate((73, 144, 703, 1001, 2759)):
        expected = {"status": "ok", "value": value}
        request = common_request(
            [
                {
                    "role": "user",
                    "content": f"Return status ok and integer value {value} as JSON.",
                }
            ],
            64,
        )
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": f"result_{index}",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "const": "ok"},
                        "value": {"type": "integer", "const": value},
                    },
                    "required": ["status", "value"],
                    "additionalProperties": False,
                },
            },
        }
        cases.append(
            {
                "case_id": f"json-{index:02d}",
                "workload_class": "tool_json",
                "request": request,
                "expect": {
                    "status_code": 200,
                    "finish_reason": "stop",
                    "json_exact": expected,
                    "reasoning_empty": True,
                },
            }
        )
    return cases


def build_long_cases(tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    cases: list[dict[str, Any]] = []
    observed: dict[str, int] = {}
    specs = [
        (
            "long-119k",
            119_800,
            ("MTP119E7", "MTP119M9", "MTP119L3"),
            True,
        ),
        (
            "long-130k",
            130_900,
            ("MTP130E7", "MTP130M9", "MTP130L3"),
            False,
        ),
    ]
    for case_id, target, codes, long_decode in specs:
        messages, count, expected = build_long_messages(
            tokenizer, target, codes, long_decode=long_decode
        )
        observed[case_id] = count
        max_tokens = 512 if long_decode else 64
        validate_long_prompt_window(case_id, count, max_tokens)
        request = common_request(messages, max_tokens)
        expectation: dict[str, Any] = {
            "status_code": 200,
            "reasoning_empty": True,
        }
        if long_decode:
            request.update({"min_tokens": 512, "ignore_eos": True})
            expectation.update(
                {
                    "finish_reason": "length",
                    "content_starts_with": expected,
                }
            )
        else:
            expectation.update(
                {"finish_reason": "stop", "content_exact": expected}
            )
        cases.append(
            {
                "case_id": case_id,
                "workload_class": "long_retrieval",
                "request": request,
                "expect": expectation,
            }
        )

    messages, count, _ = build_long_messages(
        tokenizer,
        131_500,
        ("MTP131E7", "MTP131M9", "MTP131L3"),
        long_decode=False,
    )
    observed["long-overlimit"] = count
    validate_overlimit_prompt(count)
    cases.append(
        {
            "case_id": "long-overlimit",
            "workload_class": "long_retrieval",
            "request": common_request(messages, 1),
            "expect": {
                "status_code": 400,
                "error": {
                    "type": "BadRequestError",
                    "code": 400,
                    "message_contains_all": [
                        "longer than the model's context length",
                        "131072 tokens",
                    ],
                },
            },
        }
    )
    return cases, observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite corpus: {args.output}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    long_cases, observed = build_long_cases(tokenizer)
    cases = build_short_cases() + long_cases
    payload = b"".join(canonical_line(case) for case in cases)
    atomic_create(args.output, payload)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "bytes": len(payload),
                "long_prompt_tokens": observed,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
