#!/usr/bin/env python3
"""Reproduce the sidecar-aware overlay from the sealed online-W4 base."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


EXPECTED_BASE_SHA256 = "95bb2059669f9a66abe1be8e271037eeaa11ea534ace808f1c171ca78fee101f"
EXPECTED_PATCH_SHA256 = "938c43e9d283883ffc285c4e5e3a03967ebc0167480767606131fe361f1d8644"
EXPECTED_OUTPUT_SHA256 = "b54de0a07a16a7a3070aabead3c53b80f108d89528c30763ca8e032f330d97ac"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require_digest(path: Path, expected: str, label: str) -> None:
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve(strict=True)
    patch = args.patch.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    require_digest(base, EXPECTED_BASE_SHA256, "base overlay")
    require_digest(patch, EXPECTED_PATCH_SHA256, "sidecar patch")

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        staged_base = temporary_dir / "qwen4_exp.w4-base.py"
        generated = temporary_dir / "qwen4_exp.sidecar.py"
        normalized_patch = temporary_dir / "sidecar.patch"
        shutil.copy2(base, staged_base)
        shutil.copy2(base, generated)
        patch_lines = patch.read_bytes().splitlines(keepends=True)
        if patch_lines[:2] != [
            b"--- qwen4_exp.w4-base.py\n",
            b"+++ qwen4_exp.sidecar.py\n",
        ]:
            raise RuntimeError("sidecar patch has unexpected file headers")
        patch_lines[0] = b"--- a/qwen4_exp.w4-base.py\n"
        patch_lines[1] = b"+++ b/qwen4_exp.sidecar.py\n"
        normalized_patch.write_bytes(b"".join(patch_lines))
        subprocess.run(
            [
                "git",
                "apply",
                "--unsafe-paths",
                f"--directory={temporary_dir}",
                "--whitespace=nowarn",
                str(normalized_patch),
            ],
            check=True,
        )
        require_digest(generated, EXPECTED_OUTPUT_SHA256, "generated overlay")
        os.replace(generated, output)
    print(f"SIDECAR_OVERLAY_REPRODUCED {EXPECTED_OUTPUT_SHA256} {output}")


if __name__ == "__main__":
    main()
