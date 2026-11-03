#!/usr/bin/env python3
"""Export the frozen Flash-Next pilot maps to Strata's CPU-readable format.

Explicit export only; does not load an engine or alter a service. The source
must be the complete FP32 384x64 pilot set. The accompanying JSON's temperatures
are stored separately from the map coefficients. Pin the exported SHA-256 when
passing the binary to a future opt-in engine adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

MAGIC = b"STLSAM01"
VERSION = 1
INPUT_DIM = 384
INDEX_DIM = 64
LAYERS = tuple(range(3, 48, 4))
HEADER = struct.Struct("<8sIIIIQ32s32s")
RECORD = struct.Struct("<IIfI")
RECORD_BYTES = RECORD.size + 2 * INPUT_DIM * INDEX_DIM * 4


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(weights: Path, metadata: Path, output: Path) -> dict:
    import torch

    torch.set_num_threads(2)
    if weights.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("weights file exceeds the fixed pilot export limit")
    if metadata.stat().st_size > 65536:
        raise ValueError("metadata file exceeds the fixed pilot export limit")
    source_digest, metadata_digest = digest(weights), digest(metadata)
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or meta.get("sha256") != source_digest:
        raise ValueError("metadata does not bind the supplied weights SHA-256")
    if meta.get("input") != "key_and_query" or meta.get("query_heads") != "all":
        raise ValueError("metadata is not the pilot key/query grouping contract")
    tau = meta.get("tau")
    if not isinstance(tau, dict) or set(tau) != {str(layer) for layer in LAYERS}:
        raise ValueError("metadata must provide exactly twelve layer temperatures")
    blob = torch.load(weights, map_location="cpu", weights_only=True)
    names = {f"{layer}.{kind}" for layer in LAYERS for kind in ("W_Q_idx", "W_K_idx")}
    if not isinstance(blob, dict) or set(blob) != names:
        raise ValueError("weights must contain exactly the twelve query/key pairs")
    payload = bytearray()
    source_tensors = {}
    for layer in LAYERS:
        value = tau[str(layer)]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"layer {layer} temperature must be positive and finite")
        try:
            encoded_tau = struct.pack("<f", value)
        except (OverflowError, struct.error) as exc:
            raise ValueError(f"layer {layer} temperature does not fit FP32") from exc
        native_tau = struct.unpack("<f", encoded_tau)[0]
        if not math.isfinite(native_tau) or native_tau <= 0:
            raise ValueError(f"layer {layer} temperature does not fit positive FP32")
        payload.extend(RECORD.pack(layer, 0, native_tau, 0))
        for kind in ("W_Q_idx", "W_K_idx"):
            name = f"{layer}.{kind}"
            tensor = blob[name]
            if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                    or tensor.dtype != torch.float32 or tuple(tensor.shape) != (INPUT_DIM, INDEX_DIM)):
                raise ValueError(f"{name} must be a dense FP32 [384,64] tensor")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} contains a nonfinite value")
            raw = tensor.detach().contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")
            source_tensors[name] = hashlib.sha256(raw).hexdigest()
            payload.extend(raw)
    if digest(weights) != source_digest or digest(metadata) != metadata_digest:
        raise ValueError("source changed during export")
    if len(payload) != len(LAYERS) * RECORD_BYTES:
        raise ValueError("internal payload size mismatch")
    data = HEADER.pack(MAGIC, VERSION, INPUT_DIM, INDEX_DIM, len(LAYERS), len(payload),
                       bytes.fromhex(source_digest), bytes.fromhex(metadata_digest)) + payload
    receipt = {
        "format": "STLSAM01", "version": VERSION,
        "scope": "frozen Flash-Next pilot maps; no serving or promotion claim",
        "input_dim": INPUT_DIM, "index_dim": INDEX_DIM, "layers": list(LAYERS),
        "matrix_layout": "row-major FP32 [384,64]; Q then K per layer",
        "temperature": "separate positive FP32 coefficient; not folded into maps",
        "source_weights": str(weights.resolve()), "source_weights_sha256": source_digest,
        "source_metadata": str(metadata.resolve()), "source_metadata_sha256": metadata_digest,
        "tensor_sha256": source_tensors, "output": str(output.resolve()),
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
    }
    receipt_path = Path(str(output) + ".json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("output or receipt already exists; use a fresh destination")
    # Build and validate the entire payload before writing. Exclusive creation
    # preserves prior immutable exports, including when two exporters race.
    with output.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = export(args.weights, args.metadata, args.output)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"export refused: {exc}\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
