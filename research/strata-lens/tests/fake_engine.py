#!/usr/bin/env python3
"""Small deterministic stand-in for Strata; used only by unit tests."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time


TOP_K = 10
N_LAYERS = 48
WIDTH = 4 * 2560


def bf16_word(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return (bits >> 16) & 0xFFFF


def option(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def read_tokens() -> list[int]:
    path = option("--tokens-file")
    if path is None:
        raise SystemExit("missing --tokens-file")
    return [int(piece) for piece in Path(path).read_text(encoding="ascii").split()]


def write_final_r(path: Path, rows: int, mode: str) -> None:
    finite = struct.pack("<H", bf16_word(0.5))
    expected_words = rows * WIDTH
    if mode == "missing":
        return
    if mode == "truncated":
        path.write_bytes(finite * max(0, expected_words - 1))
        return
    if mode == "nonfinite":
        path.write_bytes(struct.pack("<H", 0x7F80) + finite * (expected_words - 1))
        return
    if mode == "allzero":
        path.write_bytes(b"\x00\x00" * expected_words)
        return
    path.write_bytes(finite * expected_words)


def write_prefill_routes(path: Path, rows: int, mode: str) -> None:
    if mode == "missing_routes":
        return
    row_ids = list(range(TOP_K))
    if mode == "invalid_route_id":
        row_ids[0] = 512
    elif mode == "duplicate_route_id":
        row_ids[1] = row_ids[0]
    ids = b"".join(struct.pack("<H", rank) for _ in range(rows) for rank in row_ids)
    weight_value = 0.2 if mode == "nonnormalized_routes" else 0.1
    weight = struct.pack("<H", bf16_word(weight_value))
    weights = weight * (rows * TOP_K)
    with path.open("wb") as handle:
        for layer in range(N_LAYERS):
            recorded_layer = 99 if mode == "malformed_routes" and layer == 0 else layer
            handle.write(struct.pack("<4i", recorded_layer, 0, rows, 0))
            handle.write(ids)
            handle.write(weights)


def spawn_lingering_child() -> int:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pid_path = os.environ.get("FAKE_CHILD_PID_FILE")
    if pid_path:
        Path(pid_path).write_text(f"{child.pid}\n", encoding="ascii")
    return child.pid


def write_verifier_routes(path: Path) -> None:
    with path.open("wb") as handle:
        for layer in range(N_LAYERS):
            handle.write(struct.pack("<2i", layer, TOP_K))
            handle.write(struct.pack(f"<{TOP_K}i", *range(TOP_K)))
            handle.write(struct.pack(f"<{TOP_K}f", *([0.0] * TOP_K)))


def main() -> int:
    if "generate" in sys.argv:
        print("fake engine rejects the generate subcommand", file=sys.stderr)
        return 9
    mode = os.environ.get("FAKE_ENGINE_MODE", "success")
    if mode == "hang":
        time.sleep(60)
        return 0
    if mode == "hang_child":
        spawn_lingering_child()
        time.sleep(60)
        return 0
    if mode == "ignore_term_child":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        spawn_lingering_child()
        time.sleep(60)
        return 0
    if mode == "nonzero":
        print("injected nonzero failure", file=sys.stderr)
        return 7
    if mode == "nonzero_child":
        spawn_lingering_child()
        print("injected nonzero failure with descendant", file=sys.stderr)
        return 7

    tokens = read_tokens()
    if not tokens or tokens[-1] != 248046:
        print("missing sentinel", file=sys.stderr)
        return 8
    rows = len(tokens) - 1
    final_path = os.environ.get("STRATA_PREFILL_DUMP_R_ALL")
    routes_path = os.environ.get("STRATA_DBG_FEAT")
    if final_path is None or routes_path is None:
        print("missing controlled capture environment", file=sys.stderr)
        return 10
    if os.environ.get("STRATA_ARENA_MMAP") != "1":
        print("missing arena mmap control", file=sys.stderr)
        return 11
    if os.environ.get("STRATA_DBG_FEAT_LAYERS") != "":
        print("feature rows must be disabled", file=sys.stderr)
        return 12
    expected_env = {
        "STRATA_PF_FUSED": "0",
        "STRATA_BATCH_SELF_COMMIT": "1",
        "STRATA_QSA_SLOTS": "1",
        "STRATA_SLOT_COPY_DIRECT": "1",
        "STRATA_SLOT_COPY_BATCHSYNC": "1",
        "STRATA_BATCH_DECODE_SHARE": "0.75",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    for key, expected in expected_env.items():
        if os.environ.get(key) != expected:
            print(f"controlled environment mismatch for {key}", file=sys.stderr)
            return 13
    if "LD_PRELOAD" in os.environ:
        print("LD_PRELOAD must be absent", file=sys.stderr)
        return 14

    write_final_r(Path(final_path), rows, mode)
    write_prefill_routes(Path(routes_path), rows, mode)
    verifier_path = option("--dump-routing")
    if verifier_path is not None:
        write_verifier_routes(Path(verifier_path))
    if mode == "linger_success":
        spawn_lingering_child()
    print(f"fake engine completed {rows} real rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
