#!/usr/bin/env python3
"""Compare two Strata Lens capture runs artifact by artifact.

Phase 2 repeatability tool. The gate for any analysis work is that the
same prompt, captured twice under the same profile, produces the same
routing. This module loads two run directories, validates both with the
harness's own validators, and then compares:

- token streams (positional comparison requires identical tokens)
- prefill routing: per-layer top-1 agreement, top-10 set overlap
  (Jaccard), and weight deltas over the union of routed experts
- final residuals: per-token cosine similarity and relative L2 distance

Verdicts:
- IDENTICAL   both artifact files are byte-identical
- EQUIVALENT  not byte-identical, but every positional metric is within
              BF16 tolerance (top-1 agreement 1.0, max weight delta
              <= 2^-8, min residual cosine >= 0.999999)
- DIVERGENT   anything else; the worst layers are listed

Exit codes: 0 IDENTICAL/EQUIVALENT, 1 DIVERGENT, 2 load/validation
error or runs that cannot be compared positionally.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase0_harness as harness

WEIGHT_TOLERANCE = 2 ** -8  # one BF16 ulp near 1.0
COSINE_TOLERANCE = 0.999999


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _derive_rows(routes_path: Path) -> int:
    size = routes_path.stat().st_size
    per_layer, remainder = divmod(size, harness.N_LAYERS)
    if remainder:
        raise harness.ValidationError(
            f"{routes_path.name}: size {size} is not divisible by {harness.N_LAYERS} layers"
        )
    body = per_layer - harness.PREFILL_ROUTE_HEADER_BYTES
    per_token = harness.TOP_K * 2 + harness.TOP_K * 2
    rows, remainder = divmod(body, per_token)
    if remainder or rows <= 0:
        raise harness.ValidationError(
            f"{routes_path.name}: per-layer body {body} does not hold a whole number of tokens"
        )
    return rows


def _read_words(path: Path) -> array.array:
    words = array.array("H")
    words.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        words.byteswap()
    return words


def parse_routes(path: Path, rows: int) -> list[dict[str, Any]]:
    """Return one entry per layer: {"ids": [[...] per token], "weights": [[...] per token]}."""
    layers: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for expected_layer in range(harness.N_LAYERS):
            header = struct.unpack("<4i", handle.read(16))
            if header != (expected_layer, 0, rows, 0):
                raise harness.ValidationError(
                    f"{path.name}: layer {expected_layer} header {header!r}, "
                    f"expected ({expected_layer}, 0, {rows}, 0)"
                )
            count = rows * harness.TOP_K
            id_words = array.array("H")
            id_words.frombytes(handle.read(count * 2))
            weight_words = array.array("H")
            weight_words.frombytes(handle.read(count * 2))
            if sys.byteorder != "little":
                id_words.byteswap()
                weight_words.byteswap()
            ids = [
                list(id_words[t * harness.TOP_K:(t + 1) * harness.TOP_K])
                for t in range(rows)
            ]
            weights = [
                [harness.bf16_to_float(w) for w in weight_words[t * harness.TOP_K:(t + 1) * harness.TOP_K]]
                for t in range(rows)
            ]
            layers.append({"ids": ids, "weights": weights})
    return layers


def parse_residual(path: Path, rows: int) -> list[list[float]]:
    words = _read_words(path)
    per_row = harness.FINAL_R_BYTES_PER_TOKEN // 2
    if len(words) != rows * per_row:
        raise harness.ValidationError(
            f"{path.name}: {len(words)} BF16 words, expected {rows * per_row}"
        )
    return [
        [harness.bf16_to_float(w) for w in words[t * per_row:(t + 1) * per_row]]
        for t in range(rows)
    ]


def load_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    routes_path = run_dir / harness.PREFILL_ROUTES_NAME
    residual_path = run_dir / harness.FINAL_R_NAME
    tokens_path = run_dir / "tokens.txt"
    for required in (routes_path, residual_path, tokens_path):
        if not required.exists():
            raise harness.ValidationError(f"missing required artifact: {required}")
    rows = _derive_rows(routes_path)
    harness.validate_artifacts(run_dir, rows, False)
    tokens = [int(word) for word in tokens_path.read_text(encoding="ascii").split()]
    return {
        "dir": str(run_dir),
        "rows": rows,
        "tokens": tokens,
        "routes": parse_routes(routes_path, rows),
        "residual": parse_residual(residual_path, rows),
        "sha256": {
            harness.PREFILL_ROUTES_NAME: _sha256(routes_path),
            harness.FINAL_R_NAME: _sha256(residual_path),
        },
    }


def _layer_metrics(a: dict[str, Any], b: dict[str, Any], rows: int) -> dict[str, Any]:
    top1_agree = 0
    jaccard_total = 0.0
    max_delta = 0.0
    for t in range(rows):
        ids_a, ids_b = a["ids"][t], b["ids"][t]
        if ids_a[0] == ids_b[0]:
            top1_agree += 1
        set_a, set_b = set(ids_a), set(ids_b)
        jaccard_total += len(set_a & set_b) / len(set_a | set_b)
        weights_a = dict(zip(ids_a, a["weights"][t]))
        weights_b = dict(zip(ids_b, b["weights"][t]))
        for expert in set_a | set_b:
            delta = abs(weights_a.get(expert, 0.0) - weights_b.get(expert, 0.0))
            if delta > max_delta:
                max_delta = delta
    return {
        "top1_agreement": top1_agree / rows,
        "mean_top10_jaccard": jaccard_total / rows,
        "max_weight_delta": max_delta,
    }


def _residual_metrics(a_rows: list[list[float]], b_rows: list[list[float]]) -> dict[str, Any]:
    cosines: list[float] = []
    rel_l2: list[float] = []
    for row_a, row_b in zip(a_rows, b_rows):
        dot = sum(x * y for x, y in zip(row_a, row_b))
        norm_a = math.sqrt(sum(x * x for x in row_a))
        norm_b = math.sqrt(sum(y * y for y in row_b))
        cosines.append(dot / (norm_a * norm_b) if norm_a and norm_b else 0.0)
        diff = math.sqrt(sum((x - y) ** 2 for x, y in zip(row_a, row_b)))
        rel_l2.append(diff / norm_a if norm_a else 0.0)
    return {
        "min_cosine": min(cosines),
        "mean_cosine": sum(cosines) / len(cosines),
        "max_relative_l2": max(rel_l2),
        "mean_relative_l2": sum(rel_l2) / len(rel_l2),
    }


def compare_runs(dir_a: Path, dir_b: Path) -> dict[str, Any]:
    run_a = load_run(Path(dir_a))
    run_b = load_run(Path(dir_b))
    identical_files = run_a["sha256"] == run_b["sha256"]
    tokens_equal = run_a["tokens"] == run_b["tokens"]
    same_shape = run_a["rows"] == run_b["rows"]
    report: dict[str, Any] = {
        "run_a": run_a["dir"],
        "run_b": run_b["dir"],
        "rows_a": run_a["rows"],
        "rows_b": run_b["rows"],
        "tokens_equal": tokens_equal,
        "files_identical": identical_files,
        "positional": tokens_equal and same_shape,
    }
    if not report["positional"]:
        report["verdict"] = "INCOMPARABLE"
        report["reason"] = (
            "token streams or row counts differ; positional comparison is not meaningful"
        )
        return report
    rows = run_a["rows"]
    layers = [
        _layer_metrics(run_a["routes"][layer], run_b["routes"][layer], rows)
        for layer in range(harness.N_LAYERS)
    ]
    residual = _residual_metrics(run_a["residual"], run_b["residual"])
    worst = sorted(range(harness.N_LAYERS), key=lambda i: layers[i]["top1_agreement"])[:5]
    report["layers"] = layers
    report["worst_layers_by_top1"] = [
        {"layer": i, **layers[i]} for i in worst if layers[i]["top1_agreement"] < 1.0
    ]
    report["residual"] = residual
    report["min_top1_agreement"] = min(m["top1_agreement"] for m in layers)
    report["max_weight_delta"] = max(m["max_weight_delta"] for m in layers)
    if identical_files:
        report["verdict"] = "IDENTICAL"
    elif (
        report["min_top1_agreement"] == 1.0
        and report["max_weight_delta"] <= WEIGHT_TOLERANCE
        and residual["min_cosine"] >= COSINE_TOLERANCE
    ):
        report["verdict"] = "EQUIVALENT"
    else:
        report["verdict"] = "DIVERGENT"
    return report


def _summarize(report: dict[str, Any]) -> str:
    lines = [f"verdict: {report['verdict']}"]
    lines.append(f"runs: {report['run_a']} vs {report['run_b']}")
    lines.append(
        f"rows: {report['rows_a']} vs {report['rows_b']}; tokens equal: {report['tokens_equal']}"
    )
    if report["verdict"] == "INCOMPARABLE":
        lines.append(report["reason"])
        return "\n".join(lines)
    lines.append(
        f"min top-1 agreement: {report['min_top1_agreement']:.4f}; "
        f"max weight delta: {report['max_weight_delta']:.6f}"
    )
    residual = report["residual"]
    lines.append(
        f"residual cosine min/mean: {residual['min_cosine']:.6f}/{residual['mean_cosine']:.6f}; "
        f"max relative L2: {residual['max_relative_l2']:.6f}"
    )
    for entry in report["worst_layers_by_top1"]:
        lines.append(
            f"layer {entry['layer']}: top-1 agreement {entry['top1_agreement']:.4f}, "
            f"mean top-10 Jaccard {entry['mean_top10_jaccard']:.4f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--json", action="store_true", help="emit the full report as JSON")
    args = parser.parse_args(argv)
    try:
        report = compare_runs(args.run_a, args.run_b)
    except (harness.ValidationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        printable = {k: v for k, v in report.items() if k != "layers"}
        printable["layers"] = report.get("layers")
        print(json.dumps(printable, indent=2))
    else:
        print(_summarize(report))
    if report["verdict"] in ("IDENTICAL", "EQUIVALENT"):
        return 0
    if report["verdict"] == "DIVERGENT":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
