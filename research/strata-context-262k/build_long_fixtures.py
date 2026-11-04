#!/usr/bin/env python3
"""CPU-only fictional long-context fixtures and an independent text oracle.

No model, CUDA, API, service or production configuration is loaded. Inputs use
the pinned pure Strata tokenizer/template and the supplied tokenizer asset path.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import re
import sys


PIN = "1aff6d14117b88171fcfcec8f2b55fdc47f2b165"
HELPERS = {"tools/strata_tokenizer.py": "6db1ea62d3edbaa64dce139088146ccf370933998d5f997cd5f41a39300075a6",
           "serve/frontend.py": "e0fded1ab8c856afa8cda7464b40d2753737de18dead42fc75381def13b1c6a1"}
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
ROW = re.compile(r"^Entry ([A-Z2-7]+): seal=([A-Z2-7]+);", re.MULTILINE)
FRACTIONS = [.035, .245, .535, .765, .975]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def opaque(rng, prefix, length):
    return prefix + "".join(rng.choice(ALPHABET) for _ in range(length))


def oracle_from_prompt(prompt):
    """Reference result parsed from complete visible records, never token IDs."""
    requests = re.findall(r"^Requested keys: (\[[^\n]+\])$", prompt, re.MULTILINE)
    if len(requests) != 1:
        raise ValueError("fixture must contain exactly one final key request")
    keys = json.loads(requests[0])
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate requested key")
    records = {}
    for key, value in ROW.findall(prompt):
        if key in records:
            raise ValueError("duplicate ledger key")
        records[key] = value
    if any(key not in records for key in keys):
        raise ValueError("requested key is absent from the full prompt")
    return {key: records[key] for key in keys}


def load_helpers(source, directory):
    for relative, expected in HELPERS.items():
        if sha((source / relative).read_bytes()) != expected:
            raise ValueError("pure helper differs from pinned upstream: " + relative)
    sys.path.insert(0, str(source / "tools"))
    sys.path.insert(0, str(source))
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    vocab = json.loads((directory / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    merges = (directory / "merges.txt").read_text().split("\n")
    options = json.loads((directory / "tokenizer.json").read_text())
    tok = Tokenizer(tokens, merges, json.loads((directory / "token_type.json").read_text()),
                    options.get("pre") or "qwen35", options.get("special_ids") or None)
    template = ChatTemplate(directory / "chat_template.jinja")
    hashes = {name: sha((directory / name).read_bytes()) for name in
              ["vocab.json", "merges.txt", "token_type.json", "tokenizer.json", "chat_template.jinja"]}
    return tok, template, hashes


def record(key, value, rng):
    shelf = opaque(rng, "S", 3)
    return f"Entry {key}: seal={value}; shelf={shelf}; tally={rng.randrange(100000):05d}.\n"


def make_pool(tok, seed, needle_count):
    rng = random.Random(seed)
    reserved_keys, reserved_values = set(), set()
    needles = []
    for _ in range(needle_count):
        key, value = opaque(rng, "L", 12), opaque(rng, "V", 20)
        reserved_keys.add(key); reserved_values.add(value)
        needles.append({"key": key, "value": value, "line": record(key, value, rng)})
    rows, prefix = [], [0]
    while prefix[-1] < 266000:
        key, value = opaque(rng, "L", 12), opaque(rng, "V", 20)
        if key in reserved_keys or value in reserved_values:
            continue
        reserved_keys.add(key); reserved_values.add(value)
        line = record(key, value, rng)
        rows.append(line)
        prefix.append(prefix[-1] + len(tok.encode(line)))
    query = [needle["key"] for needle in needles]
    rng.shuffle(query)
    return rows, prefix, needles, query


def make_case(tok, template, case_id, target_tokens, seed, pool, fractions):
    rows, prefix, needles, query = pool
    if len(needles) != len(fractions):
        raise ValueError("needle/fraction count differs")
    intro = (f"Fictional archive fixture {case_id}, seed tag {seed}.\n"
             "The ledger below contains unique entries. Keys and seals are opaque synthetic identifiers.\n"
             "All entries use the same format. Read the complete ledger and retain exact requested seals.\n")
    ending = ("\nRequested keys: " + json.dumps(query, separators=(",", ":")) + "\n"
              "Return exactly one JSON object mapping each requested key to the seal in its Entry. "
              "Include every requested key and no extra fields. Copy the seal values exactly; "
              "do not infer them from keys, explain, summarize, or use a code fence.")
    system = "Use only the supplied fictional records. Follow the requested JSON format exactly."
    body_budget = target_tokens - 1000
    case = None
    for _ in range(4):
        count = min(len(rows), bisect.bisect_left(prefix, body_budget))
        boundaries = [bisect.bisect_left(prefix, fraction * body_budget) for fraction in fractions]
        parts, at = [], 0
        for boundary, needle in zip(boundaries, needles):
            boundary = min(boundary, count)
            parts.extend(rows[at:boundary]); parts.append(needle["line"]); at = boundary
        parts.extend(rows[at:count])
        question = intro + "".join(parts) + ending
        messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        prompt = template.render(messages, enable_thinking=False)
        ids = tok.encode(prompt, parse_special=True)
        case = {"id": case_id, "messages": messages, "prompt": prompt, "tokens": ids,
                "prompt_tokens": len(ids), "target_prompt_tokens": target_tokens,
                "token_sha256": sha((",".join(map(str, ids)) + "\n").encode()), "max_new": 512,
                "checker": "json", "expected_json": {n["key"]: n["value"] for n in needles},
                "seed": seed, "whole_distractor_records": count, "fresh_synthetic": True}
        difference = target_tokens - len(ids)
        if abs(difference) <= 128:
            break
        body_budget += difference
    assert case is not None
    if abs(case["prompt_tokens"] - target_tokens) > 256:
        raise ValueError("fixture missed its bounded target")
    if case["prompt_tokens"] + case["max_new"] > 262144:
        raise ValueError("fixture lacks answer allowance")
    parsed = oracle_from_prompt(case["prompt"])
    if parsed != case["expected_json"]:
        raise ValueError("independent visible-record oracle differs")
    case["expected_json"] = parsed
    case["expected"] = json.dumps(parsed, separators=(",", ":"))
    case["needles"] = []
    for needle, fraction in zip(needles, fractions):
        key, value, line = needle["key"], needle["value"], needle["line"]
        if case["prompt"].count(value) != 1 or case["prompt"].count(key) != 2:
            raise ValueError("needle value leaked/repeated or key multiplicity differs")
        char_start = case["prompt"].index(line)
        char_end = char_start + len(line)
        begin = len(tok.encode(case["prompt"][:char_start], parse_special=True))
        end = len(tok.encode(case["prompt"][:char_end], parse_special=True))
        span = tok.decode(case["tokens"][max(0, begin - 2):end + 2])
        if key not in span or value not in span:
            raise ValueError("declared needle token span does not contain record")
        case["needles"].append({"key": key, "value": value, "target_fraction": fraction,
                                "line_start_token": begin, "line_end_token": end,
                                "actual_fraction": begin / len(ids), "value_occurrences_in_prompt": 1})
    case["oracle_source"] = "Independent parse of all visible Entry lines and the final requested-key list."
    return case


def write_exclusive(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "long-context-fixtures.json")
    parser.add_argument("--seed", type=int, default=20261103627)
    parser.add_argument("--include-short-control", action="store_true")
    parser.add_argument("--deep-slots", type=int, choices=range(0, 9), default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("fixture output already exists; originals are immutable")
    tok, template, hashes = load_helpers(args.source, args.tokenizer)
    cases = []
    single_pool = make_pool(tok, args.seed + 100003, 1)
    cases.append(make_case(tok, template, "cold_opaque_marker_260k", 260000, args.seed + 100003, single_pool, [.985]))
    print(json.dumps({"event": "case_prepared", "id": cases[-1]["id"], "prompt_tokens": cases[-1]["prompt_tokens"]}), flush=True)
    multi_pool = None
    for slot in range(args.deep_slots or 1):
        seed = args.seed + 104729 * (slot + 1)
        multi_pool = make_pool(tok, seed, 5)
        name = f"deep_slot_{slot}_multi_needle_252k" if args.deep_slots else "multi_needle_252k"
        cases.append(make_case(tok, template, name, 251800, seed, multi_pool, FRACTIONS))
        print(json.dumps({"event": "case_prepared", "id": cases[-1]["id"], "prompt_tokens": cases[-1]["prompt_tokens"]}), flush=True)
    control = make_case(tok, template, "multi_needle_124k_control", 124000, args.seed, multi_pool, FRACTIONS) if args.include_short_control else None
    for case in cases:
        for needle in case["needles"]:
            if any(needle["value"] in other["prompt"] for other in cases if other is not case):
                raise ValueError("a target seal appears in another slot document")
    payload = {"schema_version": 1, "kind": "independent-fictional-long-context-fixtures",
               "prepared_at": dt.datetime.now(dt.timezone.utc).isoformat(), "pure_helper_source_pin": PIN,
               "builder_sha256": sha(Path(__file__).read_bytes()), "deep_slots": args.deep_slots,
               "cross_document_target_values_disjoint": True,
               "helper_sha256": HELPERS, "tokenizer_sha256": hashes, "cases": cases,
               "optional_short_control": control,
               "limitations": ["Exact synthetic retrieval is not general model quality.",
                               "A configured context/slot count is not proof that every slot was filled near capacity.",
                               "The optional124K control distinguishes difficult-fixture failures from a long-context-only regression.",
                               "No model/GPU/API/service was started or changed by preparation."]}
    write_exclusive(args.output, payload)
    oracle = {"schema_version": 1, "fixture_sha256": sha(args.output.read_bytes()),
              "reference": "oracle_from_prompt parses literal records; target values are random and appear once.",
              "cases": [{key: c[key] for key in ["id", "prompt_tokens", "max_new", "token_sha256", "expected_json", "needles"]}
                        for c in [*cases, *([control] if control is not None else [])]]}
    oracle_path = args.output.with_name(args.output.stem + "-oracle.json")
    write_exclusive(oracle_path, oracle)
    print(json.dumps({"status": "prepared_cpu_only", "fixture": str(args.output), "oracle": str(oracle_path),
                      "sha256": oracle["fixture_sha256"], "cases": [{"id": c["id"], "prompt_tokens": c["prompt_tokens"],
                      "needle_token_positions": [n["line_start_token"] for n in c["needles"]]} for c in [*cases, *([control] if control is not None else [])]],
                      "gpu_work_started": False}), flush=True)


if __name__ == "__main__":
    main()
