#!/usr/bin/env python3
"""CPU-only unit tests for the MTP workload histogram collector."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mtp_workload_histogram.py"
SPEC = importlib.util.spec_from_file_location("mtp_workload_histogram", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
BUILDER_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_mtp_workload_cases.py"
)
BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_mtp_workload_cases", BUILDER_SCRIPT
)
assert BUILDER_SPEC and BUILDER_SPEC.loader
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(BUILDER)


def expect_raises(exception_type, callback) -> None:
    try:
        callback()
    except exception_type:
        return
    raise AssertionError(f"expected {exception_type.__name__}")


def overlimit_expectation() -> dict:
    return {
        "status_code": 400,
        "error": {
            "type": "BadRequestError",
            "code": 400,
            "message_contains_all": [
                "longer than the model's context length",
                "131072 tokens",
            ],
        },
    }


def server_info_fixture() -> dict:
    return {
        "served_model_name": "qwen3.8-flash-next",
        "context_length": 131_072,
        "max_total_num_tokens": 139_072,
        "max_running_requests": 4,
        "kv_cache_dtype": "bfloat16",
        "speculative_draft_kv_cache_dtype": None,
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 1,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 2,
        "speculative_use_rejection_sampling": True,
        "startup_time": {"scheduler_e2e": 205.5, "tokenizer_e2e": 211.5},
    }


def test_histogram_boundaries() -> None:
    histogram = MODULE.make_histogram([1.0, 1.01, 1.25, 1.26, 2.0, 4.01])
    assert histogram["<=1.00"] == 1
    assert histogram["(1.00,1.25]"] == 2
    assert histogram["(1.25,1.50]"] == 1
    assert histogram["(1.75,2.00]"] == 1
    assert histogram[">4.00"] == 1


def test_short_corpus_builder_has_exact_measured_coverage() -> None:
    class FakeTokenizer:
        kwargs = None

        def apply_chat_template(self, _messages, **kwargs):
            self.kwargs = kwargs
            return [1, 2, 3]

    fake = FakeTokenizer()
    assert BUILDER.prompt_tokens(fake, [{"role": "user", "content": "x"}]) == 3
    assert fake.kwargs["enable_thinking"] is False

    class TensorLike:
        @staticmethod
        def tolist():
            return [[1, 2, 3, 4]]

    class TensorTokenizer:
        @staticmethod
        def apply_chat_template(*_args, **_kwargs):
            return TensorLike()

    assert (
        BUILDER.prompt_tokens(
            TensorTokenizer(), [{"role": "user", "content": "x"}]
        )
        == 4
    )

    cases = BUILDER.build_short_cases()
    assert len(cases) == 31
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert sum(not case.get("measured", True) for case in cases) == 2
    measured_counts = {
        workload_class: sum(
            case["workload_class"] == workload_class
            and case.get("measured", True)
            for case in cases
        )
        for workload_class in MODULE.WORKLOAD_CLASSES
    }
    assert measured_counts == {
        "chat": 9,
        "tool_json": 10,
        "math": 10,
        "long_retrieval": 0,
    }
    for index, case in enumerate(cases, 1):
        MODULE.validate_expectation(case["expect"], Path("generated.jsonl"), index)


def test_long_corpus_builder_does_not_leak_combined_answer() -> None:
    class FakeTokenizer:
        template_kwargs = []

        @staticmethod
        def encode(text, **_kwargs):
            return text.split()

        def apply_chat_template(self, messages, **kwargs):
            self.template_kwargs.append(kwargs)
            return " ".join(item["content"] for item in messages).split()

    fake = FakeTokenizer()
    codes = ("EARLY7", "MIDDLE9", "LATE3")
    for long_decode in (False, True):
        messages, count, expected = BUILDER.build_long_messages(
            fake, 2_000, codes, long_decode=long_decode
        )
        user_content = messages[-1]["content"]
        assert count <= 2_000
        assert expected == "EARLY7|MIDDLE9|LATE3"
        assert expected not in user_content
        assert all(user_content.count(code) == 1 for code in codes)
    assert fake.template_kwargs
    assert all(
        kwargs["enable_thinking"] is False
        and kwargs["tokenize"] is True
        and kwargs["add_generation_prompt"] is True
        for kwargs in fake.template_kwargs
    )

    BUILDER.validate_long_prompt_window("long-119k", 119_424, 512)
    BUILDER.validate_long_prompt_window("long-130k", 130_500, 64)
    BUILDER.validate_long_prompt_window("long-130k", 131_008, 64)
    for case_id, count, max_tokens in (
        ("long-119k", 119_423, 512),
        ("long-119k", 130_500, 512),
        ("long-130k", 130_499, 64),
        ("long-130k", 131_009, 64),
    ):
        expect_raises(
            ValueError,
            lambda case_id=case_id, count=count, max_tokens=max_tokens: (
                BUILDER.validate_long_prompt_window(case_id, count, max_tokens)
            ),
        )
    expect_raises(ValueError, lambda: BUILDER.validate_overlimit_prompt(131_072))
    BUILDER.validate_overlimit_prompt(131_073)


def test_weighted_summary_is_not_mean_of_means() -> None:
    results = [
        {
            "workload_class": "chat",
            "passed": True,
            "metrics": {
                "completion_tokens": 100,
                "spec_verify_ct": 50,
                "spec_num_correct_drafts": 45,
                "spec_num_proposed_drafts": 50,
                "spec_accept_length": 2.0,
                "spec_accept_rate": 0.9,
                "e2e_output_tok_s": 120.0,
                "decode_tok_s": None,
            },
        },
        {
            "workload_class": "chat",
            "passed": True,
            "metrics": {
                "completion_tokens": 9,
                "spec_verify_ct": 9,
                "spec_num_correct_drafts": 0,
                "spec_num_proposed_drafts": 9,
                "spec_accept_length": 1.0,
                "spec_accept_rate": 0.0,
                "e2e_output_tok_s": 30.0,
                "decode_tok_s": None,
            },
        },
    ]
    summary = MODULE.summarize_results(results)["chat"]
    assert summary["weighted_accept_length"] == 109 / 59
    assert summary["weighted_accept_rate"] == 45 / 59
    assert summary["e2e_output_tok_s_p50"] == 75.0


def test_exact_tool_grade() -> None:
    case = {
        "expect": {
            "finish_reason": "tool_calls",
            "reasoning_empty": True,
            "tool_call": {"name": "multiply", "arguments": {"a": 37, "b": 19}},
        }
    }
    response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "reasoning_content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "multiply",
                                "arguments": json.dumps({"a": 37, "b": 19}),
                            }
                        }
                    ],
                },
            }
        ]
    }
    assert MODULE.grade_response(case, 200, response) == []

    response["choices"][0]["message"]["tool_calls"][0]["type"] = "custom"
    assert "tool call type mismatch" in MODULE.grade_response(case, 200, response)
    response["choices"][0]["message"]["tool_calls"][0]["type"] = "function"

    response["choices"][0]["message"]["tool_calls"].append(
        {"function": {"name": "delete_all", "arguments": "{}"}}
    )
    assert "expected exactly one total tool call" in MODULE.grade_response(
        case, 200, response
    )[0]


def test_tool_argument_comparison_is_type_strict() -> None:
    case = {"expect": {"tool_call": {"name": "set_value", "arguments": {"v": 1}}}}
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "set_value",
                                "arguments": '{"v":true}',
                            },
                        }
                    ]
                }
            }
        ]
    }
    assert MODULE.grade_response(case, 200, response) == ["tool arguments mismatch"]

    response["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ] = "NaN"
    assert MODULE.grade_response(case, 200, response) == [
        "tool arguments are not strict JSON"
    ]
    response["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ] = '{"v":0,"v":1}'
    assert MODULE.grade_response(case, 200, response) == [
        "tool arguments are not strict JSON"
    ]

    expect_raises(
        ValueError,
        lambda: MODULE.validate_expectation(
            {"tool_call": {"name": "danger"}}, Path("cases.jsonl"), 1
        ),
    )
    expect_raises(
        ValueError,
        lambda: MODULE.validate_expectation(
            {"content_starts_with": ""}, Path("cases.jsonl"), 1
        ),
    )


def test_strict_json_grade_is_semantic_and_type_strict() -> None:
    case = {"expect": {"json_exact": {"status": "ok", "value": 73}}}
    response = {
        "choices": [
            {
                "message": {
                    "content": '{ "value": 73, "status": "ok" }',
                    "tool_calls": [],
                }
            }
        ]
    }
    assert MODULE.grade_response(case, 200, response) == []
    response["choices"][0]["message"]["content"] = (
        '{"status":"ok","value":true}'
    )
    assert MODULE.grade_response(case, 200, response) == ["json_exact mismatch"]
    response["choices"][0]["message"]["content"] = (
        '{"status":"bad","status":"ok","value":73}'
    )
    assert MODULE.grade_response(case, 200, response) == [
        "content is not strict JSON"
    ]
    response["choices"][0]["message"]["content"] = '{"status":"ok","value":73}'
    response["choices"][0]["message"]["tool_calls"] = [
        {"function": {"name": "unexpected", "arguments": "{}"}}
    ]
    assert MODULE.grade_response(case, 200, response) == [
        "non-tool response contained unexpected tool calls"
    ]

    starts_case = {"expect": {"content_starts_with": "EARLY|MIDDLE|LATE"}}
    starts_response = {
        "choices": [
            {"message": {"content": "EARLY|MIDDLE|LATE\nreport", "tool_calls": []}}
        ]
    }
    assert MODULE.grade_response(starts_case, 200, starts_response) == []
    starts_response["choices"][0]["message"]["content"] = (
        "preamble EARLY|MIDDLE|LATE"
    )
    assert MODULE.grade_response(starts_case, 200, starts_response) == [
        "content_starts_with mismatch"
    ]

    surrogate_case = {"expect": {"json_exact": "safe"}}
    surrogate_response = {
        "choices": [{"message": {"content": '"\\ud800"'}}]
    }
    assert MODULE.grade_response(surrogate_case, 200, surrogate_response) == [
        "content is not strict JSON"
    ]


def metric_fixture() -> dict:
    return {
        "choices": [
            {
                "meta_info": {
                    "prompt_tokens": 5,
                    "completion_tokens": 20,
                    "num_retractions": 0,
                    "spec_verify_ct": 10,
                    "spec_num_correct_drafts": 9,
                    "spec_num_proposed_drafts": 10,
                    "spec_accept_length": 2.0,
                    "spec_accept_rate": 0.9,
                    "spec_correct_drafts_histogram": [1, 9],
                    "e2e_latency": 0.2,
                }
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 20},
    }


def test_real_shaped_metric_extraction_and_adversarial_values() -> None:
    metrics = MODULE.extract_metrics(metric_fixture())
    assert metrics["spec_accept_length"] == 2.0
    assert metrics["spec_accept_rate"] == 0.9
    assert metrics["e2e_output_tok_s"] == 100.0

    impossible = metric_fixture()
    impossible["choices"][0]["meta_info"]["spec_num_correct_drafts"] = 11
    expect_raises(ValueError, lambda: MODULE.extract_metrics(impossible))

    inconsistent = metric_fixture()
    inconsistent["choices"][0]["meta_info"]["spec_accept_length"] = 1.5
    expect_raises(ValueError, lambda: MODULE.extract_metrics(inconsistent))

    nonfinite = metric_fixture()
    nonfinite["choices"][0]["meta_info"]["spec_accept_length"] = math.nan
    expect_raises(ValueError, lambda: MODULE.extract_metrics(nonfinite))

    impossible_length = metric_fixture()
    impossible_meta = impossible_length["choices"][0]["meta_info"]
    impossible_length["usage"]["completion_tokens"] = 100
    impossible_meta.update(
        {
            "completion_tokens": 100,
            "spec_verify_ct": 1,
            "spec_num_correct_drafts": 1,
            "spec_num_proposed_drafts": 1,
            "spec_accept_length": 100.0,
            "spec_accept_rate": 1.0,
            "spec_correct_drafts_histogram": [0, 1],
        }
    )
    expect_raises(ValueError, lambda: MODULE.extract_metrics(impossible_length))

    too_many_for_accepts = metric_fixture()
    too_many_meta = too_many_for_accepts["choices"][0]["meta_info"]
    too_many_for_accepts["usage"]["completion_tokens"] = 3
    too_many_meta.update(
        {
            "completion_tokens": 3,
            "spec_verify_ct": 1,
            "spec_num_correct_drafts": 0,
            "spec_num_proposed_drafts": 1,
            "spec_accept_length": 3.0,
            "spec_accept_rate": 0.0,
            "spec_correct_drafts_histogram": [1],
        }
    )
    expect_raises(ValueError, lambda: MODULE.extract_metrics(too_many_for_accepts))

    too_few_for_verifies = metric_fixture()
    too_few_meta = too_few_for_verifies["choices"][0]["meta_info"]
    too_few_for_verifies["usage"]["completion_tokens"] = 1
    too_few_meta.update(
        {
            "completion_tokens": 1,
            "spec_verify_ct": 1,
            "spec_num_correct_drafts": 0,
            "spec_num_proposed_drafts": 1,
            "spec_accept_length": 1.0,
            "spec_accept_rate": 0.0,
            "spec_correct_drafts_histogram": [1],
        }
    )
    expect_raises(ValueError, lambda: MODULE.extract_metrics(too_few_for_verifies))

    def one_verify(completion_tokens: int, correct_drafts: int) -> dict:
        response = metric_fixture()
        response["usage"]["completion_tokens"] = completion_tokens
        meta = response["choices"][0]["meta_info"]
        meta.update(
            {
                "completion_tokens": completion_tokens,
                "spec_verify_ct": 1,
                "spec_num_correct_drafts": correct_drafts,
                "spec_num_proposed_drafts": 1,
                "spec_accept_length": float(completion_tokens),
                "spec_accept_rate": float(correct_drafts),
                "spec_correct_drafts_histogram": (
                    [1] if correct_drafts == 0 else [0, 1]
                ),
            }
        )
        return response

    assert MODULE.extract_metrics(one_verify(3, 1))["spec_accept_length"] == 3.0
    assert MODULE.extract_metrics(one_verify(2, 1))["spec_accept_length"] == 2.0
    assert MODULE.extract_metrics(one_verify(2, 0))["spec_accept_length"] == 2.0

    retracted = one_verify(2, 0)
    retracted["choices"][0]["meta_info"]["num_retractions"] = 1
    expect_raises(ValueError, lambda: MODULE.extract_metrics(retracted))

    usage_mismatch = one_verify(2, 0)
    usage_mismatch["usage"]["completion_tokens"] = 3
    expect_raises(ValueError, lambda: MODULE.extract_metrics(usage_mismatch))

    histogram_mismatch = one_verify(2, 0)
    histogram_mismatch["choices"][0]["meta_info"][
        "spec_correct_drafts_histogram"
    ] = [0, 1]
    expect_raises(ValueError, lambda: MODULE.extract_metrics(histogram_mismatch))

    missing_latency = one_verify(2, 0)
    del missing_latency["choices"][0]["meta_info"]["e2e_latency"]
    expect_raises(ValueError, lambda: MODULE.extract_metrics(missing_latency))

    overflow_latency = one_verify(2, 0)
    overflow_latency["choices"][0]["meta_info"]["e2e_latency"] = 10**400
    expect_raises(OverflowError, lambda: MODULE.extract_metrics(overflow_latency))

    tiny_latency = one_verify(2, 0)
    tiny_latency["choices"][0]["meta_info"]["e2e_latency"] = 5e-324
    expect_raises(ValueError, lambda: MODULE.extract_metrics(tiny_latency))

    MODULE.validate_metrics_against_request(
        {"min_tokens": 2, "max_tokens": 3}, {}, {"completion_tokens": 2}
    )
    expect_raises(
        ValueError,
        lambda: MODULE.validate_metrics_against_request(
            {"min_tokens": 3}, {}, {"completion_tokens": 2}
        ),
    )
    expect_raises(
        ValueError,
        lambda: MODULE.validate_metrics_against_request(
            {"max_tokens": 2}, {}, {"completion_tokens": 3}
        ),
    )
    expect_raises(
        ValueError,
        lambda: MODULE.validate_metrics_against_request(
            {"max_tokens": 3},
            {"finish_reason": "length"},
            {"completion_tokens": 2},
        ),
    )


def test_case_oracle_and_campaign_coverage_are_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cases.jsonl"
        path.write_text(
            json.dumps(
                {
                    "case_id": "chat-0",
                    "workload_class": "chat",
                    "request": {"messages": [{"role": "user", "content": "x"}]},
                    "expect": {},
                }
            ),
            encoding="utf-8",
        )
        expect_raises(ValueError, lambda: MODULE.load_cases(path))

    expect_raises(
        ValueError,
        lambda: MODULE.validate_expectation(
            {"status_code": 400}, Path("cases.jsonl"), 1
        ),
    )
    expect_raises(
        ValueError,
        lambda: MODULE.validate_expectation(
            {"content_exact": ""}, Path("cases.jsonl"), 1
        ),
    )
    error_case = {"expect": overlimit_expectation()}
    correct_error = {
        "type": "BadRequestError",
        "code": 400,
        "message": (
            "The input (131272 tokens) is longer than the model's context length "
            "(131072 tokens)."
        ),
    }
    assert MODULE.grade_response(error_case, 400, correct_error) == []
    wrong_error = dict(correct_error, message="request schema was invalid")
    assert any(
        "missing error message marker" in item
        for item in MODULE.grade_response(error_case, 400, wrong_error)
    )

    one_case = [
        {
            "case_id": "chat-0",
            "workload_class": "chat",
            "request": {},
            "expect": {"content_exact": "OK"},
        }
    ]
    expect_raises(ValueError, lambda: MODULE.validate_campaign_coverage(one_case))

    expected_errors = []
    for workload_class, minimum in MODULE.CAMPAIGN_MINIMUMS.items():
        expected_errors.extend(
            {
                "case_id": f"{workload_class}-{index}",
                "workload_class": workload_class,
                "request": {},
                "expect": {"status_code": 400},
            }
            for index in range(minimum)
        )
    expect_raises(
        ValueError, lambda: MODULE.validate_campaign_coverage(expected_errors)
    )

    model_cases = [
        {"case_id": "first", "request": {}},
        {"case_id": "late", "request": {"model": "wrong-model"}},
    ]
    expect_raises(
        ValueError,
        lambda: MODULE.validate_case_models(model_cases, "qwen3.8-flash-next"),
    )


def make_campaign_fixture() -> tuple[list[dict], list[dict]]:
    cases: list[dict] = [
        {
            "case_id": "chat-warmup",
            "workload_class": "chat",
            "measured": False,
            "request": {},
            "expect": {"content_exact": "OK"},
        }
    ]
    results: list[dict] = [
        {
            "case_id": "chat-warmup",
            "workload_class": "chat",
            "measured": False,
            "passed": True,
            "status_code": 200,
            "metrics": {
                "prompt_tokens": 8,
                "completion_tokens": 2,
                "num_retractions": 0,
                "spec_verify_ct": 1,
                "spec_num_correct_drafts": 0,
                "spec_num_proposed_drafts": 1,
                "drafts_per_verify": 1,
                "spec_accept_length": 2.0,
                "spec_accept_rate": 0.0,
                "e2e_output_tok_s": 50.0,
                "decode_tok_s": None,
            },
        }
    ]
    for workload_class, minimum in MODULE.CAMPAIGN_MINIMUMS.items():
        for index in range(minimum):
            case_id = f"{workload_class}-{index}"
            request = {}
            expectation = {"content_exact": "OK"}
            if workload_class == "chat":
                request = {
                    "min_tokens": 512,
                    "max_tokens": 512,
                    "ignore_eos": True,
                }
                expectation["finish_reason"] = "length"
            elif workload_class == "tool_json" and index < 5:
                name = f"tool_{index}"
                request = {
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": name, "parameters": {}},
                        }
                    ],
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": name},
                    },
                }
                expectation = {
                    "tool_call": {"name": name, "arguments": {}}
                }
            elif workload_class == "tool_json":
                request = {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"strict": True, "schema": {}},
                    }
                }
                expectation = {"json_exact": {"status": "ok"}}
            cases.append(
                {
                    "case_id": case_id,
                    "workload_class": workload_class,
                    "request": request,
                    "expect": expectation,
                }
            )
            prompt_tokens = 32
            completion_tokens = 20
            if workload_class == "chat":
                completion_tokens = 512
            elif workload_class == "long_retrieval" and index == 0:
                prompt_tokens = 119_424
                completion_tokens = 512
            elif workload_class == "long_retrieval" and index == 1:
                prompt_tokens = 130_500
            verify_count = 256 if completion_tokens == 512 else 10
            correct_drafts = 255 if completion_tokens == 512 else 9
            proposed_drafts = verify_count
            results.append(
                {
                    "case_id": case_id,
                    "workload_class": workload_class,
                    "measured": True,
                    "passed": True,
                    "status_code": 200,
                    "metrics": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "num_retractions": 0,
                        "spec_verify_ct": verify_count,
                        "spec_num_correct_drafts": correct_drafts,
                        "spec_num_proposed_drafts": proposed_drafts,
                        "drafts_per_verify": 1,
                        "spec_accept_length": completion_tokens / verify_count,
                        "spec_accept_rate": correct_drafts / proposed_drafts,
                        "e2e_latency_s": 1.0,
                        "e2e_output_tok_s": 100.0,
                        "decode_tok_s": None,
                    },
                }
            )
    cases.append(
        {
            "case_id": "long-overlimit",
            "workload_class": "long_retrieval",
            "request": {},
            "expect": overlimit_expectation(),
        }
    )
    results.append(
        {
            "case_id": "long-overlimit",
            "workload_class": "long_retrieval",
            "measured": True,
            "passed": True,
            "status_code": 400,
            "metrics": None,
        }
    )
    return cases, results


def test_campaign_results_require_metrics_and_real_long_context() -> None:
    cases, results = make_campaign_fixture()
    MODULE.validate_campaign_coverage(cases)
    assert MODULE.validate_campaign_results(cases, results) == []

    label_only = [
        {
            **case,
            "request": {},
            "expect": {"content_exact": "OK"},
        }
        if case["workload_class"] == "tool_json"
        else case
        for case in cases
    ]
    expect_raises(ValueError, lambda: MODULE.validate_campaign_coverage(label_only))

    missing_metrics = [dict(item) for item in results]
    missing_metrics[1]["metrics"] = None
    errors = MODULE.validate_campaign_results(cases, missing_metrics)
    assert any("lacks metrics" in item for item in errors)
    assert any("metric-bearing passed chat cases" in item for item in errors)

    short_chat = [
        {**item, "metrics": dict(item["metrics"]) if item["metrics"] else None}
        for item in results
    ]
    first_chat = next(
        item
        for item in short_chat
        if item["workload_class"] == "chat" and item.get("measured", True)
    )
    first_chat["metrics"]["completion_tokens"] = 511
    errors = MODULE.validate_campaign_results(cases, short_chat)
    assert any("shorter than 512" in item for item in errors)

    bad_error_gate = [dict(item) for item in results]
    bad_error_gate[-1]["passed"] = False
    errors = MODULE.validate_campaign_results(cases, bad_error_gate)
    assert any("required error gate failed" in item for item in errors)

    shallow_long = [
        {**item, "metrics": dict(item["metrics"]) if item["metrics"] else None}
        for item in results
    ]
    for item in shallow_long:
        if item["workload_class"] == "long_retrieval" and item["metrics"]:
            item["metrics"]["prompt_tokens"] = 118_999
            item["metrics"]["completion_tokens"] = 511
    errors = MODULE.validate_campaign_results(cases, shallow_long)
    assert any("119,424+" in item for item in errors)
    assert any("130,500+" in item for item in errors)

    one_witness = [
        {**item, "metrics": dict(item["metrics"]) if item["metrics"] else None}
        for item in results
    ]
    for item in one_witness:
        if item["workload_class"] == "long_retrieval" and item["metrics"]:
            item["metrics"]["prompt_tokens"] = 32
            item["metrics"]["completion_tokens"] = 20
    first_long = next(
        item
        for item in one_witness
        if item["workload_class"] == "long_retrieval" and item["metrics"]
    )
    first_long["metrics"]["prompt_tokens"] = 130_500
    first_long["metrics"]["completion_tokens"] = 512
    errors = MODULE.validate_campaign_results(cases, one_witness)
    assert any("119,424+" in item for item in errors)

    two_high = [
        {**item, "metrics": dict(item["metrics"]) if item["metrics"] else None}
        for item in results
    ]
    for item in two_high:
        if item["workload_class"] == "long_retrieval" and item["metrics"]:
            item["metrics"]["prompt_tokens"] = 130_600
            item["metrics"]["completion_tokens"] = 512
    errors = MODULE.validate_campaign_results(cases, two_high)
    assert any("119,424+" in item for item in errors)

    wrong_depth = [
        {**item, "metrics": dict(item["metrics"]) if item["metrics"] else None}
        for item in results
    ]
    errors = MODULE.validate_campaign_results(cases, wrong_depth, 2)
    assert any("unexpected draft depth" in item for item in errors)


def test_case_bytes_are_strict_and_snapshot_bound() -> None:
    valid = (
        json.dumps(
            {
                "case_id": "math-0",
                "workload_class": "math",
                "request": {"messages": [{"role": "user", "content": "1+1"}]},
                "expect": {"content_exact": "2"},
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cases.jsonl"
        path.write_bytes(valid)
        snapshot = path.read_bytes()
        path.write_bytes(valid.replace(b'"2"', b'"3"'))
        cases = MODULE.load_cases(path, snapshot)
        assert cases[0]["expect"]["content_exact"] == "2"
        assert MODULE.sha256_bytes(snapshot) != MODULE.sha256_bytes(path.read_bytes())

        invalid = valid.replace(b'"2"', b"NaN")
        expect_raises(ValueError, lambda: MODULE.load_cases(path, invalid))
        duplicate = valid.replace(
            b'"case_id":"math-0"',
            b'"case_id":"first","case_id":"math-0"',
        )
        expect_raises(ValueError, lambda: MODULE.load_cases(path, duplicate))
        expect_raises(ValueError, lambda: MODULE.strict_json_loads("1e400"))

        unknown = json.loads(valid)
        unknown["measure"] = False
        expect_raises(
            ValueError,
            lambda: MODULE.load_cases(
                path, (json.dumps(unknown, separators=(",", ":")) + "\n").encode()
            ),
        )
        lone_surrogate = valid.replace(b'"1+1"', b'"\\ud800"')
        expect_raises(ValueError, lambda: MODULE.load_cases(path, lone_surrogate))


def test_runtime_identity_binding_is_exact() -> None:
    selected = MODULE.bind_runtime_info(
        server_info_fixture(),
        model="qwen3.8-flash-next",
        expected_context_length=131_072,
        expected_max_total_num_tokens=139_072,
        expected_max_running_requests=4,
        expected_kv_cache_dtype="bfloat16",
        expected_draft_kv_cache_dtype="bfloat16",
        expected_drafts_per_verify=1,
    )
    assert selected["effective_draft_kv_cache_dtype"] == "bfloat16"
    assert len(selected["identity_sha256"]) == 64
    wrong = dict(server_info_fixture(), kv_cache_dtype="fp8_e4m3")
    expect_raises(
        ValueError,
        lambda: MODULE.bind_runtime_info(
            wrong,
            model="qwen3.8-flash-next",
            expected_context_length=131_072,
            expected_max_total_num_tokens=139_072,
            expected_max_running_requests=4,
            expected_kv_cache_dtype="bfloat16",
            expected_draft_kv_cache_dtype="bfloat16",
            expected_drafts_per_verify=1,
        ),
    )
    restarted = server_info_fixture()
    restarted["startup_time"] = {"scheduler_e2e": 206.0}
    restarted_selected = MODULE.bind_runtime_info(
        restarted,
        model="qwen3.8-flash-next",
        expected_context_length=131_072,
        expected_max_total_num_tokens=139_072,
        expected_max_running_requests=4,
        expected_kv_cache_dtype="bfloat16",
        expected_draft_kv_cache_dtype="bfloat16",
        expected_drafts_per_verify=1,
    )
    assert restarted_selected["identity_sha256"] != selected["identity_sha256"]


def test_main_writes_receipt_for_campaign_only_failure() -> None:
    cases, results = make_campaign_fixture()
    for result in results:
        if (
            result["workload_class"] == "long_retrieval"
            and result.get("metrics")
            and result["metrics"]["completion_tokens"] >= 512
        ):
            result["metrics"]["prompt_tokens"] = 119_423
    by_id = {result["case_id"]: result for result in results}

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "cases.jsonl"
        receipt_path = root / "receipt.json"
        corpus.write_text(
            "".join(json.dumps(case, separators=(",", ":")) + "\n" for case in cases),
            encoding="utf-8",
        )
        original_argv = sys.argv
        original_run_case = MODULE.run_case
        original_fetch_server_info = MODULE.fetch_server_info
        try:
            sys.argv = [
                str(SCRIPT),
                "--cases",
                str(corpus),
                "--output",
                str(receipt_path),
                "--arm",
                "test-arm",
                "--campaign-grade",
                "--expected-drafts-per-verify",
                "1",
                "--expected-context-length",
                "131072",
                "--expected-max-total-num-tokens",
                "139072",
                "--expected-max-running-requests",
                "4",
                "--expected-kv-cache-dtype",
                "bfloat16",
                "--expected-draft-kv-cache-dtype",
                "bfloat16",
            ]
            MODULE.run_case = lambda case, *_args: dict(by_id[case["case_id"]])
            MODULE.fetch_server_info = lambda *_args: server_info_fixture()
            assert MODULE.main() == 1
        finally:
            MODULE.run_case = original_run_case
            MODULE.fetch_server_info = original_fetch_server_info
            sys.argv = original_argv

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["schema"] == MODULE.RECEIPT_SCHEMA
        assert receipt["case_results_passed"] is True
        assert receipt["campaign_passed"] is False
        assert receipt["passed"] is False
        assert any("119,424+" in item for item in receipt["campaign_errors"])
        assert receipt["warmup_attempts"] == 1
        assert receipt["measured_attempts"] == len(results) - 1
        assert any(item["measured"] is False for item in receipt["results"])
        assert receipt["summary"]["chat"]["requests"] == 9


def test_rolling_histogram_keeps_raw_counters() -> None:
    result = {
        "case_id": "chat-0",
        "workload_class": "chat",
        "completed_at": "2026-08-27T00:00:00Z",
        "passed": True,
        "metrics": {
            "completion_tokens": 20,
            "spec_verify_ct": 10,
            "spec_num_correct_drafts": 9,
            "spec_num_proposed_drafts": 10,
            "spec_accept_length": 2.0,
            "spec_accept_rate": 0.9,
            "e2e_output_tok_s": 100.0,
        },
    }
    warmup = {**result, "case_id": "chat-warmup", "measured": False}
    rolling = MODULE.update_rolling(
        None, "eagle-1-1-2", [warmup, result], result["completed_at"]
    )
    state = rolling["arms"]["eagle-1-1-2"]["workload_classes"]["chat"]
    assert state["requests"] == 1
    assert state["weighted_accept_length"] == 2.0
    assert state["weighted_accept_rate"] == 0.9
    assert state["accept_length_histogram"]["(1.75,2.00]"] == 1


def test_failures_remain_visible_without_metrics() -> None:
    passed = {
        "case_id": "chat-ok",
        "workload_class": "chat",
        "completed_at": "2026-08-27T00:00:00Z",
        "passed": True,
        "metrics": {
            "completion_tokens": 20,
            "spec_verify_ct": 10,
            "spec_num_correct_drafts": 9,
            "spec_num_proposed_drafts": 10,
            "spec_accept_length": 2.0,
            "spec_accept_rate": 0.9,
            "e2e_output_tok_s": 100.0,
            "decode_tok_s": None,
        },
    }
    failed = {
        "case_id": "chat-fail",
        "workload_class": "chat",
        "completed_at": "2026-08-27T00:00:01Z",
        "passed": False,
        "metrics": None,
    }
    summary = MODULE.summarize_results([passed, failed])["chat"]
    assert summary["requests"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["missing_metrics"] == 1

    rolling = MODULE.update_rolling(None, "arm", [passed, failed], failed["completed_at"])
    state = rolling["arms"]["arm"]["workload_classes"]["chat"]
    assert state["requests"] == 2
    assert state["failed"] == 1
    assert state["missing_metrics"] == 1
    assert state["recent"][-1]["case_id"] == "chat-fail"


def test_transport_failure_is_persisted() -> None:
    case = {
        "case_id": "chat-transport",
        "workload_class": "chat",
        "request": {"messages": [{"role": "user", "content": "say OK"}]},
        "expect": {"content_exact": "OK"},
    }

    opened_requests = []

    class BrokenOpener:
        def open(self, request, **_kwargs):
            opened_requests.append(request)
            raise MODULE.urllib.error.URLError("offline")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "cases.jsonl"
        receipt_path = root / "receipt.json"
        corpus.write_text(json.dumps(case) + "\n", encoding="utf-8")
        original_argv = sys.argv
        original_build_opener = MODULE.urllib.request.build_opener
        opener_handlers = []
        try:
            sys.argv = [
                str(SCRIPT),
                "--cases",
                str(corpus),
                "--output",
                str(receipt_path),
                "--arm",
                "transport-test",
            ]
            def build_opener(*handlers):
                opener_handlers.extend(handlers)
                return BrokenOpener()

            MODULE.urllib.request.build_opener = build_opener
            assert MODULE.main() == 1
        finally:
            MODULE.urllib.request.build_opener = original_build_opener
            sys.argv = original_argv

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["passed"] is False
        assert receipt["campaign_passed"] is None
        assert receipt["attempts"] == 1
        assert receipt["results"][0]["status_code"] == 0
        assert receipt["results"][0]["errors"] == ["transport failure: URLError"]
        proxy_handlers = [
            item
            for item in opener_handlers
            if isinstance(item, MODULE.urllib.request.ProxyHandler)
        ]
        assert len(proxy_handlers) == 1
        assert proxy_handlers[0].proxies == {}
        assert len(opened_requests) == 1
        assert opened_requests[0].get_header("Authorization") is None


def test_malformed_http_200_becomes_failed_result() -> None:
    class StaticResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return b'{"choices":[],"usage":{}}'

    class StaticOpener:
        @staticmethod
        def open(*_args, **_kwargs):
            return StaticResponse()

    case = {
        "case_id": "malformed-200",
        "workload_class": "chat",
        "request": {"messages": [{"role": "user", "content": "say OK"}]},
        "expect": {"content_exact": "OK"},
    }
    original_build_opener = MODULE.urllib.request.build_opener
    try:
        MODULE.urllib.request.build_opener = lambda *_handlers: StaticOpener()
        result = MODULE.run_case(
            case,
            "http://127.0.0.1:8002/v1/chat/completions",
            "qwen3.8-flash-next",
            1.0,
            None,
        )
    finally:
        MODULE.urllib.request.build_opener = original_build_opener
    assert result["status_code"] == 200
    assert result["passed"] is False
    assert result["metrics"] is None
    assert any("exactly one choice" in item for item in result["errors"])


def test_truncated_http_body_becomes_failed_result() -> None:
    class TruncatedResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            raise MODULE.http.client.IncompleteRead(b"partial", 100)

    class TruncatedOpener:
        @staticmethod
        def open(*_args, **_kwargs):
            return TruncatedResponse()

    case = {
        "case_id": "truncated-200",
        "workload_class": "chat",
        "request": {"messages": [{"role": "user", "content": "say OK"}]},
        "expect": {"content_exact": "OK"},
    }
    original_build_opener = MODULE.urllib.request.build_opener
    try:
        MODULE.urllib.request.build_opener = lambda *_handlers: TruncatedOpener()
        result = MODULE.run_case(
            case,
            "http://127.0.0.1:8002/v1/chat/completions",
            "qwen3.8-flash-next",
            1.0,
            None,
        )
    finally:
        MODULE.urllib.request.build_opener = original_build_opener
    assert result["passed"] is False
    assert result["errors"] == ["transport failure: IncompleteRead"]


def test_truncated_http_error_body_becomes_failed_result() -> None:
    class BrokenHTTPError(MODULE.urllib.error.HTTPError):
        def read(self):
            raise MODULE.http.client.IncompleteRead(b"partial", 100)

    class BrokenOpener:
        @staticmethod
        def open(request, **_kwargs):
            raise BrokenHTTPError(request.full_url, 400, "bad", {}, None)

    case = {
        "case_id": "truncated-400",
        "workload_class": "long_retrieval",
        "request": {"messages": [{"role": "user", "content": "x"}]},
        "expect": overlimit_expectation(),
    }
    original_build_opener = MODULE.urllib.request.build_opener
    try:
        MODULE.urllib.request.build_opener = lambda *_handlers: BrokenOpener()
        result = MODULE.run_case(
            case,
            "http://127.0.0.1:8002/v1/chat/completions",
            "qwen3.8-flash-next",
            1.0,
            None,
        )
    finally:
        MODULE.urllib.request.build_opener = original_build_opener
    assert result["passed"] is False
    assert result["errors"] == ["transport failure: IncompleteRead"]

    class TimedOutHTTPError(MODULE.urllib.error.HTTPError):
        def read(self):
            raise TimeoutError("timed out while reading error body")

    class TimedOutOpener:
        @staticmethod
        def open(request, **_kwargs):
            raise TimedOutHTTPError(request.full_url, 400, "bad", {}, None)

    try:
        MODULE.urllib.request.build_opener = lambda *_handlers: TimedOutOpener()
        timed_out = MODULE.run_case(
            case,
            "http://127.0.0.1:8002/v1/chat/completions",
            "qwen3.8-flash-next",
            1.0,
            None,
        )
    finally:
        MODULE.urllib.request.build_opener = original_build_opener
    assert timed_out["passed"] is False
    assert timed_out["errors"] == ["transport failure: TimeoutError"]


def test_metric_overflow_becomes_failed_result() -> None:
    response = metric_fixture()
    response["choices"][0].update(
        {"finish_reason": "stop", "message": {"content": "OK"}}
    )
    response["choices"][0]["meta_info"]["e2e_latency"] = 10**400
    payload = json.dumps(response, separators=(",", ":")).encode("utf-8")

    class StaticResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return payload

    class StaticOpener:
        @staticmethod
        def open(*_args, **_kwargs):
            return StaticResponse()

    case = {
        "case_id": "overflow-200",
        "workload_class": "chat",
        "request": {"messages": [{"role": "user", "content": "say OK"}]},
        "expect": {"content_exact": "OK"},
    }
    original_build_opener = MODULE.urllib.request.build_opener
    try:
        MODULE.urllib.request.build_opener = lambda *_handlers: StaticOpener()
        result = MODULE.run_case(
            case,
            "http://127.0.0.1:8002/v1/chat/completions",
            "qwen3.8-flash-next",
            1.0,
            None,
        )
    finally:
        MODULE.urllib.request.build_opener = original_build_opener
    assert result["passed"] is False
    assert result["metrics"] is None
    assert any("too large" in item.lower() for item in result["errors"])


def test_request_bound_failure_drops_metrics() -> None:
    response = metric_fixture()
    response["choices"][0].update(
        {"finish_reason": "stop", "message": {"content": "OK"}}
    )
    payload = json.dumps(response, separators=(",", ":")).encode("utf-8")

    class StaticResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return payload

    class StaticOpener:
        @staticmethod
        def open(*_args, **_kwargs):
            return StaticResponse()

    case = {
        "case_id": "below-min",
        "workload_class": "chat",
        "request": {
            "messages": [{"role": "user", "content": "say OK"}],
            "min_tokens": 21,
            "max_tokens": 30,
        },
        "expect": {"content_exact": "OK"},
    }
    original_build_opener = MODULE.urllib.request.build_opener
    try:
        MODULE.urllib.request.build_opener = lambda *_handlers: StaticOpener()
        result = MODULE.run_case(
            case,
            "http://127.0.0.1:8002/v1/chat/completions",
            "qwen3.8-flash-next",
            1.0,
            None,
        )
    finally:
        MODULE.urllib.request.build_opener = original_build_opener
    assert result["passed"] is False
    assert result["metrics"] is None
    assert result["errors"] == ["completion tokens are below request min_tokens"]


def test_deep_response_json_becomes_failed_result() -> None:
    payload = b'{"nested":' + b"[" * 1_200 + b"0" + b"]" * 1_200 + b"}"

    class StaticResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return payload

    class StaticOpener:
        @staticmethod
        def open(*_args, **_kwargs):
            return StaticResponse()

    case = {
        "case_id": "deep-response",
        "workload_class": "chat",
        "request": {"messages": [{"role": "user", "content": "say OK"}]},
        "expect": {"content_exact": "OK"},
    }
    original_build_opener = MODULE.urllib.request.build_opener
    try:
        MODULE.urllib.request.build_opener = lambda *_handlers: StaticOpener()
        result = MODULE.run_case(
            case,
            "http://127.0.0.1:8002/v1/chat/completions",
            "qwen3.8-flash-next",
            1.0,
            None,
        )
    finally:
        MODULE.urllib.request.build_opener = original_build_opener
    assert result["passed"] is False
    assert result["metrics"] is None


def test_receipt_create_never_overwrites() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "receipt.json"
        path.write_text("original", encoding="utf-8")
        expect_raises(FileExistsError, lambda: MODULE.atomic_create_json(path, {"x": 1}))
        assert path.read_text(encoding="utf-8") == "original"


def main() -> None:
    test_histogram_boundaries()
    test_short_corpus_builder_has_exact_measured_coverage()
    test_long_corpus_builder_does_not_leak_combined_answer()
    test_weighted_summary_is_not_mean_of_means()
    test_exact_tool_grade()
    test_tool_argument_comparison_is_type_strict()
    test_strict_json_grade_is_semantic_and_type_strict()
    test_real_shaped_metric_extraction_and_adversarial_values()
    test_case_oracle_and_campaign_coverage_are_fail_closed()
    test_campaign_results_require_metrics_and_real_long_context()
    test_case_bytes_are_strict_and_snapshot_bound()
    test_runtime_identity_binding_is_exact()
    test_main_writes_receipt_for_campaign_only_failure()
    test_rolling_histogram_keeps_raw_counters()
    test_failures_remain_visible_without_metrics()
    test_transport_failure_is_persisted()
    test_malformed_http_200_becomes_failed_result()
    test_truncated_http_body_becomes_failed_result()
    test_truncated_http_error_body_becomes_failed_result()
    test_metric_overflow_becomes_failed_result()
    test_request_bound_failure_drops_metrics()
    test_deep_response_json_becomes_failed_result()
    test_receipt_create_never_overwrites()
    print("MTP_WORKLOAD_HISTOGRAM_TESTS_OK")


if __name__ == "__main__":
    main()
