#!/usr/bin/env python3
"""Synthetic round-trip and corruption tests for the PLE W4 sidecar."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

RELEASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_DIR / "src"))
sys.path.insert(0, str(RELEASE_DIR / "scripts"))

from build_ple_w4_sidecar import quantize_fp8_rows
from qwen4_ple_w4_sidecar import (
    FORMAT_NAME,
    FORMAT_VERSION,
    MANIFEST_DIGEST_NAME,
    MANIFEST_NAME,
    SCALE_FILE,
    WEIGHT_FILE,
    SidecarError,
    canonical_json_bytes,
    load_sidecar_into_tensors,
)


def write_manifest(root: Path, manifest: dict) -> None:
    data = canonical_json_bytes(manifest)
    (root / MANIFEST_NAME).write_bytes(data)
    (root / MANIFEST_DIGEST_NAME).write_text(
        f"{hashlib.sha256(data).hexdigest()}  {MANIFEST_NAME}\n", encoding="ascii"
    )


def make_artifact(root: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    # Values are first rounded into the checkpoint's actual FP8 domain.
    raw = torch.linspace(-11.5, 13.0, 128 * 32, dtype=torch.float32).reshape(128, 32)
    raw[0].zero_()
    source = raw.to(torch.float8_e4m3fn)
    packed, scales = quantize_fp8_rows(source)
    weight_data = packed.numpy().tobytes(order="C")
    scale_data = scales.view(torch.uint8).numpy().tobytes(order="C")
    (root / WEIGHT_FILE).write_bytes(weight_data)
    (root / SCALE_FILE).write_bytes(scale_data)
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "source": {
            "checkpoint_shards": 128,
            "rows_per_shard": 1,
            "model_revision": "synthetic-test",
            "hf_tree_receipt_sha256": "4" * 64,
            "ple_files": [
                {
                    "file": f"model-plefp8-{i:05d}.safetensors",
                    "bytes": 1,
                    "lfs_sha256": "5" * 64,
                }
                for i in range(10)
            ],
            "model_index_sha256": "0" * 64,
            "config_sha256": "1" * 64,
            "ple_raw_sha256": "2" * 64,
            "ple_shard_sha256": ["3" * 64] * 128,
            "module_prefix": "model.language_model.layers.1.ple.ple_embedding",
            "outer_weight_scale": 0.00019931793212890625,
        },
        "quantization": {
            "algorithm": "signed_uniform_int4",
            "group_size": 16,
            "scale_dtype": "float8_e4m3fn",
            "nibble_order": "even_low_odd_high_twos_complement",
            "rounding": "torch_round_nearest_even",
            "scale_rule": "max(max_positive/7,max_negative/8).clamp_min(2**-9)",
        },
        "layout": {
            "rows": 128,
            "embedding_dim": 32,
            "group_size": 16,
            "packed_dim": 16,
            "groups_per_row": 2,
        },
        "tensors": {
            "weight": {
                "file": WEIGHT_FILE,
                "dtype": "uint8",
                "shape": [128, 16],
                "bytes": len(weight_data),
                "sha256": hashlib.sha256(weight_data).hexdigest(),
            },
            "block_scales": {
                "file": SCALE_FILE,
                "dtype": "float8_e4m3fn",
                "shape": [128, 2],
                "bytes": len(scale_data),
                "sha256": hashlib.sha256(scale_data).hexdigest(),
            },
        },
    }
    write_manifest(root, manifest)
    return packed, scales, manifest


def expect_sidecar_error(label: str, callback) -> None:
    try:
        callback()
    except SidecarError:
        return
    raise AssertionError(f"{label} did not raise SidecarError")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ple-w4-sidecar-test-") as temporary:
        root = Path(temporary)
        expected_weight, expected_scales, manifest = make_artifact(root)
        weight = torch.empty_like(expected_weight)
        scales = torch.empty_like(expected_scales)
        expected_source = {
            "model_index_sha256": "0" * 64,
            "config_sha256": "1" * 64,
            "ple_raw_sha256": "2" * 64,
            "model_revision": "synthetic-test",
        }
        receipt = load_sidecar_into_tensors(
            root,
            weight,
            scales,
            chunk_bytes=4096,
            expected_source=expected_source,
        )
        assert torch.equal(weight, expected_weight)
        assert torch.equal(scales.view(torch.uint8), expected_scales.view(torch.uint8))
        assert receipt["outer_weight_scale"] == 0.00019931793212890625
        expect_sidecar_error(
            "external manifest binding",
            lambda: load_sidecar_into_tensors(
                root,
                torch.empty_like(weight),
                torch.empty_like(scales),
                chunk_bytes=4096,
                expected_manifest_sha256="f" * 64,
            ),
        )

        if torch.cuda.is_available():
            pinned_weight = torch.empty(
                expected_weight.shape, dtype=torch.uint8, pin_memory=True
            )
            pinned_scales = torch.empty(
                expected_scales.shape,
                dtype=torch.float8_e4m3fn,
                pin_memory=True,
            )
            assert pinned_weight.is_pinned() and pinned_scales.is_pinned()
            load_sidecar_into_tensors(
                root, pinned_weight, pinned_scales, chunk_bytes=4096
            )
            assert torch.equal(pinned_weight, expected_weight)
            assert torch.equal(
                pinned_scales.view(torch.uint8), expected_scales.view(torch.uint8)
            )

        for source_key in expected_source:
            wrong_source = dict(expected_source)
            wrong_source[source_key] = "wrong"
            expect_sidecar_error(
                f"source binding {source_key}",
                lambda wrong_source=wrong_source: load_sidecar_into_tensors(
                    root,
                    torch.empty_like(weight),
                    torch.empty_like(scales),
                    chunk_bytes=4096,
                    expected_source=wrong_source,
                ),
            )

        # A changed binary must fail its streaming digest check.
        weight_path = root / WEIGHT_FILE
        original = weight_path.read_bytes()
        corrupted = bytearray(original)
        corrupted[len(corrupted) // 2] ^= 0x80
        weight_path.write_bytes(corrupted)
        expect_sidecar_error(
            "binary corruption",
            lambda: load_sidecar_into_tensors(
                root, torch.empty_like(weight), torch.empty_like(scales), chunk_bytes=4096
            ),
        )
        weight_path.write_bytes(original)

        scale_path = root / SCALE_FILE
        original_scales = scale_path.read_bytes()
        corrupted_scales = bytearray(original_scales)
        corrupted_scales[-1] ^= 0x01
        scale_path.write_bytes(corrupted_scales)
        expect_sidecar_error(
            "scale corruption",
            lambda: load_sidecar_into_tensors(
                root, torch.empty_like(weight), torch.empty_like(scales), chunk_bytes=4096
            ),
        )
        scale_path.write_bytes(original_scales)

        weight_path.write_bytes(original[:-1])
        expect_sidecar_error(
            "truncated weight",
            lambda: load_sidecar_into_tensors(
                root, torch.empty_like(weight), torch.empty_like(scales), chunk_bytes=4096
            ),
        )
        weight_path.write_bytes(original + b"\x00")
        expect_sidecar_error(
            "appended weight",
            lambda: load_sidecar_into_tensors(
                root, torch.empty_like(weight), torch.empty_like(scales), chunk_bytes=4096
            ),
        )
        weight_path.write_bytes(original)

        # A changed manifest without a matching receipt must fail before loading.
        manifest_path = root / MANIFEST_NAME
        original_manifest = manifest_path.read_bytes()
        manifest_path.write_bytes(original_manifest + b" ")
        expect_sidecar_error(
            "manifest corruption",
            lambda: load_sidecar_into_tensors(
                root, torch.empty_like(weight), torch.empty_like(scales), chunk_bytes=4096
            ),
        )
        manifest_path.write_bytes(original_manifest)

        # Even a self-consistent malicious manifest cannot traverse the root.
        malicious = json.loads(original_manifest)
        malicious["tensors"]["weight"]["file"] = "../qweight.u8"
        write_manifest(root, malicious)
        expect_sidecar_error(
            "path traversal",
            lambda: load_sidecar_into_tensors(
                root, torch.empty_like(weight), torch.empty_like(scales), chunk_bytes=4096
            ),
        )
        write_manifest(root, manifest)

        wrong_rows = json.loads(original_manifest)
        wrong_rows["source"]["rows_per_shard"] = 2
        write_manifest(root, wrong_rows)
        expect_sidecar_error(
            "rows per shard",
            lambda: load_sidecar_into_tensors(
                root, torch.empty_like(weight), torch.empty_like(scales), chunk_bytes=4096
            ),
        )
        write_manifest(root, manifest)

        wrong_module = json.loads(original_manifest)
        wrong_module["source"]["module_prefix"] = "wrong.module"
        write_manifest(root, wrong_module)
        expect_sidecar_error(
            "module prefix",
            lambda: load_sidecar_into_tensors(
                root, torch.empty_like(weight), torch.empty_like(scales), chunk_bytes=4096
            ),
        )
        write_manifest(root, manifest)

        wrong_scale = json.loads(original_manifest)
        wrong_scale["source"]["outer_weight_scale"] = -1.0
        write_manifest(root, wrong_scale)
        expect_sidecar_error(
            "outer scale",
            lambda: load_sidecar_into_tensors(
                root, torch.empty_like(weight), torch.empty_like(scales), chunk_bytes=4096
            ),
        )
        write_manifest(root, manifest)

        # Shape mismatch must fail before any file bytes are trusted.
        expect_sidecar_error(
            "shape mismatch",
            lambda: load_sidecar_into_tensors(
                root,
                torch.empty((256, 16), dtype=torch.uint8),
                torch.empty((256, 2), dtype=torch.float8_e4m3fn),
                chunk_bytes=4096,
            ),
        )

    print("PLE_W4_SIDECAR_TESTS_OK")


if __name__ == "__main__":
    main()
