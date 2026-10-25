#!/usr/bin/env python3
"""Validate the public bundle without model assets or GPU access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import phase0_harness as harness


def main() -> int:
    errors: list[str] = []
    try:
        rows = (ROOT / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    listed: set[str] = set()
    for row in rows:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", row)
        if not match:
            errors.append("malformed checksum row")
            continue
        expected, name = match.groups()
        path = ROOT / name
        if (name in listed or Path(name).is_absolute() or ".." in Path(name).parts
                or name == "SHA256SUMS" or path.is_symlink() or not path.is_file()):
            errors.append(f"invalid checksum path: {name}")
            continue
        listed.add(name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"checksum mismatch: {name}")
    public_files = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in {"__pycache__", "runs", "local"} or part.endswith(".partial")
                    for part in path.relative_to(ROOT).parts)
        and path.name != "SHA256SUMS"
        and path.suffix not in {".pyc", ".log", ".bf16", ".bin"}
        and not path.name.endswith(".local.json")
    }
    if listed != public_files:
        errors.append(f"manifest membership mismatch: {sorted(listed ^ public_files)}")
    try:
        harness.configure_profile(harness.EXAMPLE_PROFILE)
        plan = harness.make_plan(harness.reference_prompt_tokens("numbers_01"), False)
        assert plan["engine"]["input_attestation_verified"] is False
        assert plan["expected_bytes"] == {
            "prefill_final_r.bf16": 2580480, "prefill_routes.bin": 242688}
        assert harness.PROFILE_ATTESTED is False
        contract = json.loads((ROOT / "input-contract.example.json").read_text())
        assert contract["files"] == []
    except (OSError, ValueError, AssertionError, harness.HarnessError) as exc:
        errors.append(f"example/reference invariant: {exc}")
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"Strata Lens publication valid: {len(listed)} files; reference plan; non-launchable templates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
