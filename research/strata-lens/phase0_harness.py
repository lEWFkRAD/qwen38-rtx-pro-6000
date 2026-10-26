#!/usr/bin/env python3
"""Safe Phase-0a planner, offline runner, and artifact validator.

This module never stops or starts production.  The default CLI actions are
read-only (``plan`` and ``validate``).  ``run-offline`` is deliberately gated
and refuses to launch while the production unit is active.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


WORKSPACE = Path(__file__).resolve().parent
BASE = Path("/srv/qwen")
CORPUS = WORKSPACE / "corpus/prompts.json"
TOKENIZER_TOOLS = BASE / "runtime-0.1.42/tools"

PINNED_ENGINE = Path("/srv/qwen/engine-0.1.46/strata")
PINNED_BUILD = Path("/srv/qwen/engine-0.1.46/BUILD.json")
PINNED_ENGINE_SHA256 = "e54498e95ee5974c5a2b92b430aa867861baf884d217e7761dd68e2d905a1e2e"
PINNED_VERSION = "0.1.46"
PINNED_SOURCE = "f-021-rel a7bdbfdabc9a"

PACK = BASE / "data/packs/iq3_s"
NATIVE = BASE / "models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
PLE = BASE / "models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"
PROFILE = BASE / "runtime-0.1.42/data/expert-profile.bin"
MTP = BASE / "data/mtp/rt"
CONTROL_VECTOR_FILE = BASE / "models/Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf"
CONTROL_VECTOR = f"{CONTROL_VECTOR_FILE}:1.0"

PRODUCTION_UNIT = "qwen-strata.service"
PRODUCTION_PORT = 8002
PRODUCTION_CONFIG = Path("/srv/qwen/config/strata-server.json")
RECOVERY_HOLD = Path("/srv/qwen/maintenance/HOLD")
SENTINEL_TOKEN = 248_046
VOCAB_SIZE = 248_320
N_LAYERS = 48
N_EXPERTS = 512
TOP_K = 10
HC = 4
N_EMBD = 2560
FINAL_R_WIDTH = HC * N_EMBD
PREFILL_CHUNK = 1024
MAX_CONTEXT = 1024
MAX_REAL_TOKENS = MAX_CONTEXT - 2  # one sentinel prompt token plus one generated token

FINAL_R_NAME = "prefill_final_r.bf16"
PREFILL_ROUTES_NAME = "prefill_routes.bin"
VERIFIER_ROUTING_NAME = "verifier_routing.bin"
TOKENS_NAME = "tokens.txt"
MANIFEST_NAME = "manifest.json"
STDOUT_NAME = "stdout.log"
STDERR_NAME = "stderr.log"
INPUT_CONTRACT_PATH = Path("/srv/qwen/config/strata-lens-input-contract.json")
PINNED_INPUT_CONTRACT_SHA256: str | None = None
OPERATOR_PROFILE: dict[str, Any] | None = None
PROFILE_SHA256: str | None = None
PROFILE_ATTESTED = False
EXAMPLE_PROFILE = WORKSPACE / "profile.example.json"
REFERENCE_TOKENS = WORKSPACE / "corpus/numbers_01.tokens.json"
MANIFEST_SCHEMA = 3
TEST_FAKE_ENGINE = WORKSPACE / "tests/fake_engine.py"
NUMBERS_01_REAL_TOKEN_COUNT = 126
NUMBERS_01_LAUNCHED_TOKEN_FILE_SHA256 = (
    "78e69e599855cda89b0bbd0a821c1f529c1ef2ad80223f441eb45f58f3db4d62"
)
HANDLED_LAUNCH_SIGNALS = (
    signal.SIGINT,
    signal.SIGTERM,
    signal.SIGHUP,
    signal.SIGQUIT,
)
CONTAINMENT_CONTRACT = {
    "scope": "original-pgid-only",
    "engine_process_model": "single-process-no-daemonization",
    "cgroup_containment": False,
}

# Exact arithmetic/path switches from the production Strata configuration on
# 2026-10-25.  STRATA_PF_FUSED is the one switch here that changes the native
# prompt expert implementation (and its rounding).  The remaining switches
# are verifier/slot/batch controls; most are dormant in this single one-shot
# launch, but pinning them prevents the offline invocation from silently
# taking a different verifier path when source defaults change.
PRODUCTION_PARITY_ENV = {
    "STRATA_PF_FUSED": "0",
    "STRATA_BATCH_SELF_COMMIT": "1",
    "STRATA_QSA_SLOTS": "1",
    "STRATA_SLOT_COPY_DIRECT": "1",
    "STRATA_SLOT_COPY_BATCHSYNC": "1",
    "STRATA_BATCH_DECODE_SHARE": "0.75",
}

FINAL_R_BYTES_PER_TOKEN = FINAL_R_WIDTH * 2
PREFILL_ROUTE_HEADER_BYTES = 4 * 4
PREFILL_ROUTE_BYTES_PER_LAYER = lambda n: PREFILL_ROUTE_HEADER_BYTES + n * TOP_K * 2 + n * TOP_K * 2
VERIFIER_ROUTING_BYTES = N_LAYERS * (2 * 4 + TOP_K * 4 + TOP_K * 4)

class HarnessError(RuntimeError):
    """Base class for an expected harness failure."""


class IdentityError(HarnessError):
    pass


class ValidationError(HarnessError):
    pass


class ExecutionError(HarnessError):
    pass


class ExecutionTimeout(ExecutionError):
    pass


class ExecutionSignal(ExecutionError):
    def __init__(self, signum: int):
        self.signum = int(signum)
        try:
            name = signal.Signals(self.signum).name
        except ValueError:
            name = f"signal-{self.signum}"
        self.signal_name = name
        self.received_epoch = time.time()
        super().__init__(f"received {name} ({self.signum}) during engine launch")


class _LaunchSignalGuard:
    """Temporary launch signal handlers with non-interruptible cleanup mode."""

    def __init__(self) -> None:
        self._previous: dict[int, Any] = {}
        self._installed = False
        self._defer_during_popen = False
        self._pending: ExecutionSignal | None = None

    def _set_ignored(self) -> None:
        if not self._installed:
            return
        for signum in HANDLED_LAUNCH_SIGNALS:
            signal.signal(signum, signal.SIG_IGN)

    def _handler(self, signum: int, frame: Any) -> None:
        del frame
        if self._pending is None:
            self._pending = ExecutionSignal(signum)
        # Never raise between Popen returning internally and Python assigning
        # the process object/its intended PGID.  Do not use pthread_sigmask:
        # an exec'd child would inherit that mask and ignore cleanup signals.
        if self._defer_during_popen:
            return
        self._set_ignored()
        raise self._pending

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            raise HarnessError("run_once launch signal containment requires the main Python thread")
        installed: list[int] = []
        try:
            for signum in HANDLED_LAUNCH_SIGNALS:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handler)
                installed.append(signum)
        except BaseException:
            for signum in reversed(installed):
                signal.signal(signum, self._previous[signum])
            raise
        self._installed = True

    def defer_for_popen(self) -> None:
        self._defer_during_popen = True

    def finish_popen(self) -> None:
        self._defer_during_popen = False
        if self._pending is not None:
            self._set_ignored()
            raise self._pending

    def begin_cleanup(self) -> None:
        self._set_ignored()

    def restore(self) -> None:
        if not self._installed:
            return
        # Signals were ignored during cleanup, so restoring one handler at a
        # time cannot interrupt teardown (which is already complete here).
        for signum in HANDLED_LAUNCH_SIGNALS:
            signal.signal(signum, self._previous[signum])
        self._installed = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_profile(path: Path, expected_sha256: str | None = None) -> None:
    """Load explicit operator choices; trust requires an out-of-band reviewed pin.

    This is a single-profile CLI/library process. Do not switch profiles while a
    run is in progress. No values are discovered from a live installation.
    """
    global PINNED_ENGINE, PINNED_BUILD, PACK, NATIVE, PLE, PROFILE, MTP
    global CONTROL_VECTOR_FILE, CONTROL_VECTOR, TOKENIZER_TOOLS
    global PRODUCTION_UNIT, PRODUCTION_PORT, PRODUCTION_CONFIG, RECOVERY_HOLD
    global INPUT_CONTRACT_PATH, PINNED_INPUT_CONTRACT_SHA256
    global OPERATOR_PROFILE, PROFILE_SHA256, PROFILE_ATTESTED
    raw = path.read_bytes()
    data = json.loads(raw)
    required = {"schema", "state", "deployment_id", "paths", "production",
                "input_contract", "maintenance_protocol"}
    if not isinstance(data, dict) or set(data) != required or data["schema"] != 1:
        raise IdentityError("operator profile has an unsupported shape/schema")
    if data["state"] not in ("template", "attested"):
        raise IdentityError("profile state must be template or attested")
    if not isinstance(data["deployment_id"], str) or not data["deployment_id"].strip():
        raise IdentityError("profile deployment_id must be nonempty")
    if data["maintenance_protocol"] != "owned-hold-inhibits-all-restart-paths":
        raise IdentityError("profile must explicitly attest the owned HOLD maintenance protocol")
    paths = data["paths"]
    names = {"engine", "build", "pack", "native", "ple", "expert_profile", "mtp",
             "control_vector", "tokenizer_tools", "production_config", "recovery_hold"}
    if not isinstance(paths, dict) or set(paths) != names:
        raise IdentityError("profile paths must contain every required path and no extras")
    for value in paths.values():
        if (not isinstance(value, str) or not Path(value).is_absolute()
                or ".." in Path(value).parts or "\n" in value or "\r" in value):
            raise IdentityError("profile paths must be normalized absolute paths")
    if Path(paths["build"]).parent != Path(paths["engine"]).parent:
        raise IdentityError("BUILD.json must share the pinned engine directory")
    production = data["production"]
    if (not isinstance(production, dict) or set(production) != {"unit", "port"}
            or not isinstance(production["unit"], str)
            or not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", production["unit"])
            or type(production["port"]) is not int or not 1 <= production["port"] <= 65535):
        raise IdentityError("profile production unit/port is invalid")
    contract = data["input_contract"]
    if (not isinstance(contract, dict) or set(contract) != {"path", "sha256"}
            or not isinstance(contract["path"], str) or not Path(contract["path"]).is_absolute()):
        raise IdentityError("profile requires an absolute input-contract path")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and (
            not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or digest != expected_sha256):
        raise IdentityError("operator profile SHA-256 does not match the supplied reviewed pin")
    attested = data["state"] == "attested" and expected_sha256 is not None
    if data["state"] == "attested" and not (
            isinstance(contract["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", contract["sha256"])):
        raise IdentityError("attested profile requires a reviewed input-contract SHA-256")
    PINNED_ENGINE, PINNED_BUILD = Path(paths["engine"]), Path(paths["build"])
    PACK, NATIVE, PLE = Path(paths["pack"]), Path(paths["native"]), Path(paths["ple"])
    PROFILE, MTP = Path(paths["expert_profile"]), Path(paths["mtp"])
    CONTROL_VECTOR_FILE = Path(paths["control_vector"])
    CONTROL_VECTOR = f"{CONTROL_VECTOR_FILE}:1.0"
    TOKENIZER_TOOLS = Path(paths["tokenizer_tools"])
    PRODUCTION_CONFIG, RECOVERY_HOLD = Path(paths["production_config"]), Path(paths["recovery_hold"])
    PRODUCTION_UNIT, PRODUCTION_PORT = production["unit"], production["port"]
    INPUT_CONTRACT_PATH = Path(contract["path"])
    PINNED_INPUT_CONTRACT_SHA256 = contract["sha256"]
    OPERATOR_PROFILE, PROFILE_SHA256, PROFILE_ATTESTED = data, digest, attested


def _load_input_contract() -> tuple[dict[str, Any], str]:
    if not PROFILE_ATTESTED:
        raise IdentityError("real identity requires an attested operator profile and separately supplied --profile-sha256")
    try:
        raw = INPUT_CONTRACT_PATH.read_bytes()
        contract = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentityError(f"cannot load reviewed input contract {INPUT_CONTRACT_PATH}: {exc}") from exc
    if not isinstance(contract, dict) or contract.get("schema") != 1:
        raise IdentityError("input contract has an unsupported schema")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PINNED_INPUT_CONTRACT_SHA256:
        raise IdentityError("input contract SHA-256 does not match the reviewed operator profile")
    return contract, digest


def _contract_entries(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = contract.get("files")
    if not isinstance(entries, list) or not entries:
        raise IdentityError("input contract requires independently attested files")
    paths: set[str] = set()
    for entry in entries:
        if (not isinstance(entry, dict) or set(entry) != {"path", "size", "mtime_ns", "sha256"}
                or not isinstance(entry["path"], str) or not Path(entry["path"]).is_absolute()
                or type(entry["size"]) is not int or entry["size"] < 0
                or type(entry["mtime_ns"]) is not int or entry["mtime_ns"] < 0
                or not isinstance(entry["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
            raise IdentityError("every input needs exact path/size/mtime_ns and a reviewed SHA-256; stat-only trust is unsupported")
        if entry["path"] in paths:
            raise IdentityError("input contract contains duplicate file paths")
        paths.add(entry["path"])
    required = {
        str(PINNED_BUILD), str(NATIVE), str(PLE), str(PROFILE),
        str(CONTROL_VECTOR_FILE), str(CORPUS), str(PRODUCTION_CONFIG),
        str(TOKENIZER_TOOLS / "strata_tokenizer.py"), str(TOKENIZER_TOOLS / "gguf_reader.py"),
        *(str(PINNED_ENGINE.parent / name) for name in (
            "SYSTEM-LIBS.txt", "engine-component.json", "libcublas.so.13",
            "libcublasLt.so.13", "libcudart.so.13", "libstrata_kq.so")),
    }
    if required - paths:
        raise IdentityError("input contract omits required path(s): " + ", ".join(sorted(required - paths)))
    directories = contract.get("directories")
    if (not isinstance(directories, list) or len(directories) != 2
            or any(not isinstance(row, dict) for row in directories)
            or {row.get("path") for row in directories} != {str(PACK), str(MTP)}):
        raise IdentityError("input contract needs exact PACK and MTP directory memberships")
    for row in directories:
        if set(row) != {"path", "files", "directories"}:
            raise IdentityError("directory contract has unexpected keys")
        for key in ("files", "directories"):
            if (not isinstance(row[key], list)
                    or any(not isinstance(v, str) or not v or Path(v).is_absolute()
                           or ".." in Path(v).parts for v in row[key])
                    or len(set(row[key])) != len(row[key])):
                raise IdentityError("directory membership must use unique safe relative paths")
        if not row["files"] or {str(Path(row["path"]) / v) for v in row["files"]} - paths:
            raise IdentityError("every directory member requires an attested file fingerprint")
    if contract.get("tokenizer_runtime") != {"python_major_minor": [3, 12], "regex_version": "2026.9.29"}:
        raise IdentityError("tokenizer runtime contract differs from the qualified pin")
    return entries


def _file_receipt(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {**entry, "method": "sha256+stat"}


def _verify_contract_file(entry: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(entry["path"])
    if path.is_symlink() or not path.is_file():
        raise IdentityError(f"input is missing, symlinked, or not a regular file: {path}")
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (entry["size"], entry["mtime_ns"]):
        raise IdentityError(f"input size/mtime_ns mismatch: {path}")
    if sha256_file(path) != entry["sha256"]:
        raise IdentityError(f"input SHA-256 mismatch: {path}")
    return _file_receipt(entry)


def _recorded_identity() -> dict[str, Any]:
    """Validate recorded input attestations without requiring model files."""
    contract, digest = _load_input_contract()
    entries = _contract_entries(contract)
    return {
        "path": str(PINNED_ENGINE), "build_path": str(PINNED_BUILD),
        "sha256": PINNED_ENGINE_SHA256, "version": PINNED_VERSION, "source": PINNED_SOURCE,
        "profile_sha256": PROFILE_SHA256,
        "input_contract": {
            "contract_path": str(INPUT_CONTRACT_PATH), "contract_sha256": digest,
            "contract_id": contract.get("contract_id"),
            "files": [_file_receipt(entry) for entry in entries],
            "tokenizer_runtime": contract["tokenizer_runtime"],
        },
    }


def reference_prompt_tokens(prompt_id: str) -> tuple[int, ...]:
    """Return checked-in reference IDs for an asset-free plan; never tokenize a run."""
    payload = json.loads(REFERENCE_TOKENS.read_text(encoding="utf-8"))
    if prompt_id != "numbers_01" or payload.get("prompt_id") != prompt_id:
        raise HarnessError("only numbers_01 has reference plan tokens; use explicit --token-ids")
    text = load_prompt_text(prompt_id)
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != payload["prompt_text_sha256"]:
        raise IdentityError("reference prompt text does not match its token fixture")
    ids = validate_token_ids(payload["real_token_ids"])
    digest = hashlib.sha256((" ".join(map(str, ids + (SENTINEL_TOKEN,))) + "\n").encode("ascii")).hexdigest()
    if len(ids) != NUMBERS_01_REAL_TOKEN_COUNT or digest != NUMBERS_01_LAUNCHED_TOKEN_FILE_SHA256:
        raise IdentityError("reference token fixture differs from pinned tokenization")
    return ids


def _verify_contract_directory(entry: Mapping[str, Any]) -> set[str]:
    path_value = entry.get("path")
    expected_files = entry.get("files")
    expected_directories = entry.get("directories", [])
    if not isinstance(path_value, str) or not path_value.startswith("/"):
        raise IdentityError(f"invalid directory path in input contract: {path_value!r}")
    if not isinstance(expected_files, list) or any(not isinstance(item, str) for item in expected_files):
        raise IdentityError(f"invalid file membership contract for {path_value}")
    if not isinstance(expected_directories, list) or any(
        not isinstance(item, str) for item in expected_directories
    ):
        raise IdentityError(f"invalid subdirectory membership contract for {path_value}")
    if len(set(expected_files)) != len(expected_files) or len(set(expected_directories)) != len(
        expected_directories
    ):
        raise IdentityError(f"duplicate directory membership in contract for {path_value}")
    path = Path(path_value)
    if path.is_symlink() or not path.is_dir():
        raise IdentityError(f"pinned input directory is missing, symlinked, or not a directory: {path}")
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for child in path.rglob("*"):
        relative = child.relative_to(path).as_posix()
        if child.is_symlink():
            raise IdentityError(f"symlink is forbidden in pinned input directory: {child}")
        if child.is_file():
            actual_files.add(relative)
        elif child.is_dir():
            actual_directories.add(relative)
        else:
            raise IdentityError(f"non-file input is forbidden in pinned directory: {child}")
    if actual_files != set(expected_files) or actual_directories != set(expected_directories):
        raise IdentityError(
            f"pinned directory membership mismatch for {path}: files={sorted(actual_files)!r}, "
            f"directories={sorted(actual_directories)!r}"
        )
    return {str(path / relative) for relative in expected_files}


def assert_model_inputs() -> dict[str, Any]:
    contract, digest = _load_input_contract()
    entries = _contract_entries(contract)
    for directory in contract["directories"]:
        _verify_contract_directory(directory)
    verified = [_verify_contract_file(entry) for entry in entries]
    try:
        import regex  # type: ignore
    except ImportError as exc:
        raise IdentityError("pinned tokenizer dependency regex is unavailable") from exc
    observed = {"python_major_minor": [sys.version_info.major, sys.version_info.minor],
                "regex_version": regex.__version__}
    if observed != contract["tokenizer_runtime"]:
        raise IdentityError(f"tokenizer runtime mismatch: {observed!r}")
    return {"contract_path": str(INPUT_CONTRACT_PATH), "contract_sha256": digest,
            "contract_id": contract.get("contract_id"), "files": verified,
            "tokenizer_runtime": observed}


def assert_pinned_engine() -> dict[str, Any]:
    _load_input_contract()  # fail before touching any example path
    if not PINNED_ENGINE.is_file():
        raise IdentityError(f"pinned engine is missing: {PINNED_ENGINE}")
    if not PINNED_BUILD.is_file():
        raise IdentityError(f"pinned BUILD.json is missing: {PINNED_BUILD}")
    actual_sha = sha256_file(PINNED_ENGINE)
    with PINNED_BUILD.open("r", encoding="utf-8") as handle:
        build = json.load(handle)
    problems: list[str] = []
    if actual_sha != PINNED_ENGINE_SHA256:
        problems.append(f"sha256 {actual_sha} != {PINNED_ENGINE_SHA256}")
    if build.get("version") != PINNED_VERSION:
        problems.append(f"version {build.get('version')!r} != {PINNED_VERSION!r}")
    if build.get("source") != PINNED_SOURCE:
        problems.append(f"source {build.get('source')!r} != {PINNED_SOURCE!r}")
    if problems:
        raise IdentityError("pinned engine identity mismatch: " + "; ".join(problems))
    inputs = assert_model_inputs()
    return {
        "path": str(PINNED_ENGINE),
        "build_path": str(PINNED_BUILD),
        "sha256": actual_sha,
        "version": build["version"],
        "source": build["source"],
        "input_contract": inputs,
        "profile_sha256": PROFILE_SHA256,
    }


def validate_token_ids(token_ids: Sequence[int]) -> tuple[int, ...]:
    ids = tuple(int(value) for value in token_ids)
    if not ids:
        raise HarnessError("at least one real prompt token is required")
    if len(ids) > MAX_REAL_TOKENS:
        raise HarnessError(
            f"this first slice supports at most {MAX_REAL_TOKENS} real tokens; got {len(ids)}"
        )
    for index, token in enumerate(ids):
        if token < 0 or token >= VOCAB_SIZE:
            raise HarnessError(f"token {index} is outside [0, {VOCAB_SIZE}): {token}")
    return ids


def parse_token_ids(text: str) -> tuple[int, ...]:
    pieces = [piece for piece in re.split(r"[\s,]+", text.strip()) if piece]
    try:
        return validate_token_ids(tuple(int(piece, 10) for piece in pieces))
    except ValueError as exc:
        raise HarnessError(f"invalid token-id list: {exc}") from exc


def load_prompt_text(prompt_id: str, corpus_path: Path | None = None) -> str:
    corpus_path = CORPUS if corpus_path is None else corpus_path
    with corpus_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    matches = [row for row in payload.get("prompts", []) if row.get("id") == prompt_id]
    if len(matches) != 1:
        raise HarnessError(f"expected exactly one corpus prompt {prompt_id!r}; found {len(matches)}")
    text = matches[0].get("text")
    if not isinstance(text, str) or not text:
        raise HarnessError(f"corpus prompt {prompt_id!r} has no text")
    return text


def tokenize_prompt(prompt_id: str = "numbers_01", corpus_path: Path | None = None) -> tuple[int, ...]:
    contract, _ = _load_input_contract()
    entries = _contract_entries(contract)
    by_path = {entry["path"]: entry for entry in entries}
    # Importing external Python executes it: verify both sources first.
    for source in (TOKENIZER_TOOLS / "strata_tokenizer.py", TOKENIZER_TOOLS / "gguf_reader.py"):
        _verify_contract_file(by_path[str(source)])
    corpus_path = CORPUS if corpus_path is None else corpus_path
    text = load_prompt_text(prompt_id, corpus_path)
    sys.path.insert(0, str(TOKENIZER_TOOLS))
    try:
        from strata_tokenizer import Tokenizer  # type: ignore

        tokenizer = Tokenizer.from_gguf(str(NATIVE))
        ids = tokenizer.encode(text)
    finally:
        try:
            sys.path.remove(str(TOKENIZER_TOOLS))
        except ValueError:
            pass
    validated = validate_token_ids(ids)
    if prompt_id == "numbers_01" and corpus_path.resolve() == CORPUS.resolve():
        launched = validated + (SENTINEL_TOKEN,)
        token_bytes = (" ".join(str(token) for token in launched) + "\n").encode("ascii")
        digest = hashlib.sha256(token_bytes).hexdigest()
        if len(validated) != NUMBERS_01_REAL_TOKEN_COUNT or digest != NUMBERS_01_LAUNCHED_TOKEN_FILE_SHA256:
            raise IdentityError(
                f"pinned numbers_01 tokenization drifted: count/SHA-256 {(len(validated), digest)!r}"
            )
    return validated


def expected_artifact_bytes(real_token_count: int, include_verifier_routing: bool) -> dict[str, int]:
    result = {
        FINAL_R_NAME: real_token_count * FINAL_R_BYTES_PER_TOKEN,
        PREFILL_ROUTES_NAME: N_LAYERS * PREFILL_ROUTE_BYTES_PER_LAYER(real_token_count),
    }
    if include_verifier_routing:
        result[VERIFIER_ROUTING_NAME] = VERIFIER_ROUTING_BYTES
    return result


def build_model_args(
    tokens_path: Path,
    verifier_routing_path: Path | None = None,
) -> list[str]:
    args = [
        "--pack", str(PACK),
        "--native", str(NATIVE),
        "--ple-gguf", str(PLE),
        "--expert-profile", str(PROFILE),
        "--expert-cache", "auto",
        "--prefill", str(PREFILL_CHUNK),
        "--spec", "4",
        "--spec-min-p", "0.5",
        "--mtp", str(MTP),
        "--max-context", str(MAX_CONTEXT),
        "--kv", "int8",
        "--vram-reserve-mib", "700",
        "--control-vector-scaled", CONTROL_VECTOR,
        "--control-vector-layer-range", "4", "44",
        "--cvec-mode", "project",
        "--cvec-dir", "per-layer",
        "--pool-workers", "12",
        "--pcie-frac", "0",
        "--no-prefill-borrow",
        "--ple-io", "ram",
        "--tokens-file", str(tokens_path),
        "--max-new", "1",
    ]
    if verifier_routing_path is not None:
        args.extend(["--dump-routing", str(verifier_routing_path)])
    if "generate" in args:
        raise AssertionError("the one-shot parser does not accept a generate subcommand")
    if "--mmap-experts" in args:
        raise AssertionError("the native pack must use --expert-cache auto, not --mmap-experts")
    return args


def build_child_env(
    run_dir: Path,
    base_env: Mapping[str, str] | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    env = dict(os.environ if base_env is None else base_env)
    # The research executable must not inherit invisible engine/loader/GPU
    # behavior from an operator shell.  Add back only the exact controls
    # recorded below.
    for key in list(env):
        if key.startswith((
            "STRATA_", "CUDA_", "LD_", "HIP_", "HSA_", "ROCR_", "NVIDIA_"
        )):
            env.pop(key, None)
    controlled = {
        **PRODUCTION_PARITY_ENV,
        "STRATA_ARENA_MMAP": "1",
        "STRATA_PREFILL_DUMP_R_ALL": str(run_dir / FINAL_R_NAME),
        "STRATA_DBG_FEAT": str(run_dir / PREFILL_ROUTES_NAME),
        "STRATA_DBG_FEAT_LAYERS": "",
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": str(PINNED_ENGINE.parent),
    }
    env.update(controlled)
    if extra_env:
        normalized = {str(key): str(value) for key, value in extra_env.items()}
        forbidden = sorted(
            key for key in normalized
            if key in controlled
            or key == "LD_PRELOAD"
            or key.startswith(("STRATA_", "CUDA_", "LD_", "HIP_", "HSA_", "ROCR_"))
        )
        if forbidden:
            raise HarnessError(
                "extra child environment may not override engine/loader/GPU controls: "
                + ", ".join(forbidden)
            )
        env.update(normalized)
    return env, controlled


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _mark_status(manifest: dict[str, Any], status: str, epoch: float | None = None) -> float:
    timestamp = time.time() if epoch is None else float(epoch)
    manifest["status"] = status
    history = manifest.setdefault("status_history", [])
    if not isinstance(history, list):
        raise ExecutionError("manifest status_history was corrupted in memory")
    history.append({"status": status, "epoch": timestamp})
    return timestamp


def _flush_and_fsync_handle(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ExecutionError(f"durability target is missing, symlinked, or not regular: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_tokens(path: Path, real_token_ids: Sequence[int]) -> tuple[int, ...]:
    launched = tuple(real_token_ids) + (SENTINEL_TOKEN,)
    with path.open("x", encoding="ascii") as handle:
        handle.write(" ".join(str(token) for token in launched))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return launched


def _read_u16_words(path: Path) -> array.array[int]:
    words = array.array("H")
    with path.open("rb") as handle:
        words.frombytes(handle.read())
    if sys.byteorder != "little":
        words.byteswap()
    return words


def bf16_to_float(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(word) << 16))[0]


def validate_final_r(path: Path, real_token_count: int) -> dict[str, Any]:
    expected = real_token_count * FINAL_R_BYTES_PER_TOKEN
    try:
        actual = path.stat().st_size
    except FileNotFoundError as exc:
        raise ValidationError(f"missing required artifact: {path}") from exc
    if actual != expected:
        raise ValidationError(f"{path.name}: expected {expected} bytes, found {actual}")

    word_offset = 0
    row_nonzero = [False] * real_token_count
    with path.open("rb") as handle:
        while True:
            data = handle.read(1024 * 1024)
            if not data:
                break
            if len(data) % 2:
                raise ValidationError(f"{path.name}: odd BF16 byte count")
            words = array.array("H")
            words.frombytes(data)
            if sys.byteorder != "little":
                words.byteswap()
            for local_index, word in enumerate(words):
                if word & 0x7F80 == 0x7F80:
                    index = word_offset + local_index
                    value = bf16_to_float(word)
                    raise ValidationError(
                        f"{path.name}: non-finite BF16 at scalar {index}: {value!r}"
                    )
                absolute_index = word_offset + local_index
                if word & 0x7FFF:
                    row_nonzero[absolute_index // FINAL_R_WIDTH] = True
            word_offset += len(words)
    zero_rows = [index for index, nonzero in enumerate(row_nonzero) if not nonzero]
    if zero_rows:
        preview = zero_rows[:8]
        suffix = "..." if len(zero_rows) > len(preview) else ""
        raise ValidationError(f"{path.name}: all-zero BF16 row(s): {preview}{suffix}")
    return {
        "path": path.name,
        "bytes": actual,
        "rows": real_token_count,
        "row_width": FINAL_R_WIDTH,
        "dtype": "bf16-le-raw",
        "all_finite": True,
        "all_rows_nonzero": True,
        "classification": "native-prefill final residual; structural validation only",
    }


def _read_exact(handle: Any, count: int, label: str) -> bytes:
    data = handle.read(count)
    if len(data) != count:
        raise ValidationError(f"prefill routing truncated while reading {label}: wanted {count}, got {len(data)}")
    return data


def validate_prefill_routes(path: Path, real_token_count: int) -> dict[str, Any]:
    expected = N_LAYERS * PREFILL_ROUTE_BYTES_PER_LAYER(real_token_count)
    try:
        actual = path.stat().st_size
    except FileNotFoundError as exc:
        raise ValidationError(f"missing required artifact: {path}") from exc
    if actual != expected:
        raise ValidationError(f"{path.name}: expected {expected} bytes, found {actual}")

    min_sum = math.inf
    max_sum = -math.inf
    with path.open("rb") as handle:
        for expected_layer in range(N_LAYERS):
            header = struct.unpack("<4i", _read_exact(handle, 16, f"layer {expected_layer} header"))
            layer, pos, rows, has_features = header
            if header != (expected_layer, 0, real_token_count, 0):
                raise ValidationError(
                    f"{path.name}: record {expected_layer} header {header!r}, expected "
                    f"({expected_layer}, 0, {real_token_count}, 0)"
                )
            count = rows * TOP_K
            ids = array.array("H")
            ids.frombytes(_read_exact(handle, count * 2, f"layer {layer} ids"))
            weights = array.array("H")
            weights.frombytes(_read_exact(handle, count * 2, f"layer {layer} weights"))
            if sys.byteorder != "little":
                ids.byteswap()
                weights.byteswap()
            for scalar_index, expert_id in enumerate(ids):
                if expert_id >= N_EXPERTS:
                    raise ValidationError(
                        f"{path.name}: layer {layer} expert id {expert_id} at scalar {scalar_index} is invalid"
                    )
            for token_index in range(rows):
                total = 0.0
                start = token_index * TOP_K
                token_ids = ids[start:start + TOP_K]
                if len(set(token_ids)) != TOP_K:
                    raise ValidationError(
                        f"{path.name}: layer {layer} token {token_index} has duplicate expert ids"
                    )
                for rank in range(TOP_K):
                    weight = bf16_to_float(weights[start + rank])
                    if not math.isfinite(weight):
                        raise ValidationError(
                            f"{path.name}: layer {layer} token {token_index} rank {rank} has non-finite weight"
                        )
                    if weight < 0.0 or weight > 1.01:
                        raise ValidationError(
                            f"{path.name}: layer {layer} token {token_index} rank {rank} weight {weight} is invalid"
                        )
                    total += weight
                if not 0.98 <= total <= 1.02:
                    raise ValidationError(
                        f"{path.name}: layer {layer} token {token_index} BF16 weight sum {total:.6f} is not normalized"
                    )
                min_sum = min(min_sum, total)
                max_sum = max(max_sum, total)
        if handle.read(1):
            raise ValidationError(f"{path.name}: trailing bytes after layer {N_LAYERS - 1}")

    return {
        "path": path.name,
        "bytes": actual,
        "records": N_LAYERS,
        "rows_per_record": real_token_count,
        "top_k": TOP_K,
        "weight_sum_range": [min_sum, max_sum],
        "classification": "native-prefill router trace; structurally valid, not a scientific conclusion",
    }


def validate_verifier_routing(path: Path) -> dict[str, Any]:
    try:
        actual = path.stat().st_size
    except FileNotFoundError as exc:
        raise ValidationError(f"missing optional verifier artifact: {path}") from exc
    if actual != VERIFIER_ROUTING_BYTES:
        raise ValidationError(
            f"{path.name}: expected {VERIFIER_ROUTING_BYTES} bytes, found {actual}"
        )
    with path.open("rb") as handle:
        for expected_layer in range(N_LAYERS):
            layer, k = struct.unpack("<2i", _read_exact(handle, 8, f"verifier layer {expected_layer} header"))
            if layer != expected_layer or k != TOP_K:
                raise ValidationError(
                    f"{path.name}: record {expected_layer} header {(layer, k)!r}, expected {(expected_layer, TOP_K)!r}"
                )
            ids = struct.unpack(f"<{TOP_K}i", _read_exact(handle, TOP_K * 4, f"verifier layer {layer} ids"))
            weights = struct.unpack(f"<{TOP_K}f", _read_exact(handle, TOP_K * 4, f"verifier layer {layer} weights"))
            if any(expert < 0 or expert >= N_EXPERTS for expert in ids):
                raise ValidationError(f"{path.name}: invalid expert id at layer {layer}")
            if any(not math.isfinite(weight) or weight != 0.0 for weight in weights):
                raise ValidationError(
                    f"{path.name}: native verifier weights must be finite zero placeholders at layer {layer}"
                )
        if handle.read(1):
            raise ValidationError(f"{path.name}: trailing bytes")
    return {
        "path": path.name,
        "bytes": actual,
        "records": N_LAYERS,
        "classification": "non-research verifier-sentinel plumbing; weights are zero placeholders",
    }


def validate_artifacts(
    run_dir: Path,
    real_token_count: int,
    include_verifier_routing: bool,
) -> dict[str, Any]:
    result = {
        "prefill_final_r": validate_final_r(run_dir / FINAL_R_NAME, real_token_count),
        "prefill_routes": validate_prefill_routes(run_dir / PREFILL_ROUTES_NAME, real_token_count),
    }
    if include_verifier_routing:
        result["verifier_routing"] = validate_verifier_routing(run_dir / VERIFIER_ROUTING_NAME)
    return result


def _artifact_receipts(run_dir: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for name in names:
        path = run_dir / name
        try:
            size = path.stat().st_size
        except FileNotFoundError as exc:
            raise ValidationError(f"cannot receipt missing artifact: {path}") from exc
        receipts[name] = {"bytes": size, "sha256": sha256_file(path)}
    return receipts


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _manifest_list_of_ints(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValidationError(f"manifest {label} must be a list of integers")
    return value


def _manifest_timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"manifest {label} must be a finite timestamp")
    return float(value)


def validate_run_dir(run_dir: Path, *, _allow_test_override: bool = False) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / MANIFEST_NAME
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except FileNotFoundError as exc:
        raise ValidationError(f"manifest missing: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValidationError("manifest root must be an object")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValidationError(f"manifest schema must be exactly {MANIFEST_SCHEMA}")
    if manifest.get("success") is not True or manifest.get("status") != "complete":
        raise ValidationError("manifest is not a completed successful run")
    if "error" in manifest:
        raise ValidationError("successful manifest unexpectedly contains an error")
    if "termination_signal" in manifest:
        raise ValidationError("successful manifest unexpectedly records a termination signal")
    if manifest.get("timed_out") is not False:
        raise ValidationError("manifest does not prove that the engine avoided timeout")
    if type(manifest.get("engine_return_code")) is not int or manifest["engine_return_code"] != 0:
        raise ValidationError("manifest engine return code is not exactly zero")

    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or run_dir.name != run_id:
        raise ValidationError("run directory name does not match manifest run_id")
    launch_dir = run_dir.parent / f"{run_id}.partial"
    if manifest.get("launch_directory") != str(launch_dir):
        raise ValidationError("manifest launch directory is missing or does not match the atomic staging path")

    engine = manifest.get("engine")
    if not isinstance(engine, dict):
        raise ValidationError("manifest engine identity is missing")
    is_test = engine.get("test_override") is True
    if is_test:
        if not _allow_test_override:
            raise ValidationError("test-only engine identity is never accepted by offline validation")
    else:
        try:
            current_identity = _recorded_identity()
        except IdentityError as exc:
            raise ValidationError(f"recorded engine/input attestation cannot be proven: {exc}") from exc
        if engine != current_identity:
            raise ValidationError("manifest engine/version/source/input identity does not match the exact pin")

    prompt = manifest.get("prompt", {})
    if not isinstance(prompt, dict):
        raise ValidationError("manifest prompt must be an object")
    real_ids = _manifest_list_of_ints(prompt.get("real_token_ids"), "real_token_ids")
    validated_ids = validate_token_ids(real_ids)
    if prompt.get("sentinel_token") != SENTINEL_TOKEN:
        raise ValidationError("manifest sentinel does not match the pinned sentinel")
    launched = list(validated_ids) + [SENTINEL_TOKEN]
    if prompt.get("real_token_count") != len(validated_ids):
        raise ValidationError("manifest real_token_count does not match real_token_ids")
    if prompt.get("launch_token_count") != len(launched):
        raise ValidationError("manifest launch_token_count does not include exactly one sentinel")
    if _manifest_list_of_ints(manifest.get("launched_token_ids"), "launched_token_ids") != launched:
        raise ValidationError("manifest launched_token_ids do not equal real ids plus sentinel")
    tokens_path = run_dir / TOKENS_NAME
    try:
        token_text = tokens_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"required tokens file is missing or unreadable: {exc}") from exc
    expected_token_text = " ".join(str(token) for token in launched) + "\n"
    if token_text != expected_token_text:
        raise ValidationError("tokens.txt does not exactly match manifest real ids plus sentinel")

    requested = manifest.get("requested_artifacts")
    if not isinstance(requested, dict) or set(requested) != {
        "prefill_final_r", "prefill_routes", "verifier_routing"
    }:
        raise ValidationError("manifest requested_artifacts has an unexpected shape")
    if requested.get("prefill_final_r") is not True or requested.get("prefill_routes") is not True:
        raise ValidationError("both native prefill artifacts must be requested")
    if type(requested.get("verifier_routing")) is not bool:
        raise ValidationError("manifest verifier_routing request must be boolean")
    include_verifier = requested["verifier_routing"]
    expected_bytes = expected_artifact_bytes(len(validated_ids), include_verifier)
    if manifest.get("expected_bytes") != expected_bytes:
        raise ValidationError("manifest expected byte counts do not match token count/request")

    verifier_path = launch_dir / VERIFIER_ROUTING_NAME if include_verifier else None
    if is_test:
        command_prefix = manifest.get("test_command_prefix")
        if not isinstance(command_prefix, list) or not command_prefix or any(
            not isinstance(item, str) for item in command_prefix
        ):
            raise ValidationError("test manifest command prefix is malformed")
    else:
        command_prefix = [str(PINNED_ENGINE)]
    expected_argv = command_prefix + build_model_args(launch_dir / TOKENS_NAME, verifier_path)
    if manifest.get("argv") != expected_argv:
        raise ValidationError("manifest argv is not the exact pinned one-shot invocation")
    _, expected_env = build_child_env(launch_dir, base_env={})
    if manifest.get("controlled_env") != expected_env:
        raise ValidationError("manifest controlled environment is not the exact pinned capture/parity contract")
    expected_sanitized = ["STRATA_*", "CUDA_*", "LD_*", "HIP_*", "HSA_*", "ROCR_*", "NVIDIA_*"]
    if manifest.get("sanitized_env_prefixes") != expected_sanitized:
        raise ValidationError("manifest does not record the exact sanitized environment contract")

    if manifest.get("containment") != CONTAINMENT_CONTRACT:
        raise ValidationError("manifest containment/process-model contract is not exact")
    gate = manifest.get("gate_evidence")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValidationError("manifest has no successful prelaunch gate evidence")
    if is_test:
        if set(gate) != {"mode", "passed", "callback_checked", "checked_epoch"} or gate.get(
            "mode"
        ) != "test-only" or type(gate.get("callback_checked")) is not bool:
            raise ValidationError("test manifest gate evidence is malformed")
    else:
        expected_gate = {
            "mode": "production-quiescence",
            "passed": True,
            "unit": PRODUCTION_UNIT,
            "load_state": "loaded",
            "active_state": "inactive",
            "sub_state": "dead",
            "main_pid": 0,
            "port": PRODUCTION_PORT,
            "listener_present": False,
            "hold_owner_confirmed": True,
        }
        if {key: gate.get(key) for key in expected_gate} != expected_gate or set(gate) != set(
            expected_gate
        ) | {"checked_epoch"}:
            raise ValidationError("production manifest gate evidence is malformed")

    group = manifest.get("process_group")
    if not isinstance(group, dict):
        raise ValidationError("manifest has no process-group cleanup receipt")
    expected_group_keys = {
        "child_pid", "pgid", "cleanup_started_epoch", "cleanup_finished_epoch",
        "leader_was_done", "group_existed_at_cleanup", "term_sent", "kill_sent",
        "lingering_after_parent", "gone_before_finalize", "containment_scope",
    }
    if set(group) != expected_group_keys:
        raise ValidationError("manifest process-group cleanup receipt has an unexpected shape")
    child_pid = manifest.get("child_pid")
    child_pgid = manifest.get("child_pgid")
    observed_pgid = manifest.get("observed_child_pgid")
    if any(type(value) is not int or value <= 0 for value in (child_pid, child_pgid, observed_pgid)):
        raise ValidationError("manifest child PID/PGID fields are invalid")
    if not (child_pid == child_pgid == observed_pgid == group.get("child_pid") == group.get("pgid")):
        raise ValidationError("manifest child PID, intended/observed PGID, and cleanup PGID disagree")
    if group.get("containment_scope") != CONTAINMENT_CONTRACT["scope"]:
        raise ValidationError("cleanup receipt overclaims its containment scope")
    for boolean_key in (
        "leader_was_done", "group_existed_at_cleanup", "term_sent", "kill_sent",
        "lingering_after_parent", "gone_before_finalize",
    ):
        if type(group.get(boolean_key)) is not bool:
            raise ValidationError(f"cleanup receipt {boolean_key} must be boolean")
    if group.get("gone_before_finalize") is not True:
        raise ValidationError("manifest does not prove process-group cleanup before finalization")
    if group.get("lingering_after_parent") is not False:
        raise ValidationError("successful run recorded lingering original-PGID members")
    if (
        group.get("leader_was_done") is not True
        or group.get("group_existed_at_cleanup") is not False
        or group.get("term_sent") is not False
        or group.get("kill_sent") is not False
    ):
        raise ValidationError(
            "successful single-process run required cleanup intervention or had a live original PGID"
        )

    expected_statuses = [
        "created", "launching", "gated", "running", "cleaned", "validating",
        "validated_pending_finalize", "complete",
    ]
    history = manifest.get("status_history")
    if not isinstance(history, list) or [row.get("status") for row in history if isinstance(row, dict)] != expected_statuses:
        raise ValidationError("manifest status history is not the exact successful lifecycle")
    if len(history) != len(expected_statuses) or any(not isinstance(row, dict) or set(row) != {"status", "epoch"} for row in history):
        raise ValidationError("manifest status history entries are malformed")
    history_times = [_manifest_timestamp(row["epoch"], f"status_history[{index}].epoch") for index, row in enumerate(history)]
    if history_times != sorted(history_times):
        raise ValidationError("manifest status history timestamps are out of order")

    durability = manifest.get("durability")
    durable_files = [TOKENS_NAME, STDOUT_NAME, STDERR_NAME, *expected_bytes.keys(), MANIFEST_NAME]
    if not isinstance(durability, dict) or durability.get("completed") is not True:
        raise ValidationError("manifest does not prove pre-rename file durability")
    if set(durability) != {"completed", "completed_epoch", "files"} or durability.get(
        "files"
    ) != durable_files:
        raise ValidationError("manifest durability receipt is malformed")

    started = _manifest_timestamp(manifest.get("started_epoch"), "started_epoch")
    gated = _manifest_timestamp(manifest.get("prelaunch_gate_epoch"), "prelaunch_gate_epoch")
    gate_checked = _manifest_timestamp(gate.get("checked_epoch"), "gate_evidence.checked_epoch")
    popen_epoch = _manifest_timestamp(manifest.get("popen_epoch"), "popen_epoch")
    cleanup_started = _manifest_timestamp(group.get("cleanup_started_epoch"), "cleanup_started_epoch")
    cleanup_finished = _manifest_timestamp(group.get("cleanup_finished_epoch"), "cleanup_finished_epoch")
    validation_epoch = _manifest_timestamp(manifest.get("validation_epoch"), "validation_epoch")
    durable_epoch = _manifest_timestamp(durability.get("completed_epoch"), "durability.completed_epoch")
    renamed_epoch = _manifest_timestamp(manifest.get("renamed_epoch"), "renamed_epoch")
    finished = _manifest_timestamp(manifest.get("finished_epoch"), "finished_epoch")
    if gate_checked != gated or not (
        started <= gated <= popen_epoch <= cleanup_started <= cleanup_finished
        <= validation_epoch <= durable_epoch <= renamed_epoch <= finished
    ):
        raise ValidationError("manifest lifecycle timestamps are inconsistent")
    exact_history_times = {
        0: started,
        2: gated,
        4: cleanup_finished,
        6: validation_epoch,
        7: finished,
    }
    if any(history_times[index] != epoch for index, epoch in exact_history_times.items()):
        raise ValidationError("manifest named timestamps do not match exact lifecycle status entries")

    validation = validate_artifacts(run_dir, len(validated_ids), include_verifier)
    if manifest.get("validation") != validation:
        raise ValidationError("recorded artifact validation does not match fresh validation")
    expected_names = list(expected_bytes)
    receipts = manifest.get("artifact_receipts")
    if not isinstance(receipts, dict) or set(receipts) != set(expected_names):
        raise ValidationError("manifest artifact receipts are missing or incomplete")
    current_receipts = _artifact_receipts(run_dir, expected_names)
    if receipts != current_receipts:
        raise ValidationError("artifact size/SHA-256 receipt mismatch")
    return validation


@dataclass(frozen=True)
class RunRequest:
    real_token_ids: tuple[int, ...]
    output_root: Path
    include_verifier_routing: bool = False
    timeout_seconds: float = 600.0
    kill_grace_seconds: float = 10.0
    command_prefix: tuple[str, ...] = field(default_factory=lambda: (str(PINNED_ENGINE),))
    engine_identity: Mapping[str, Any] | None = None
    extra_child_env: Mapping[str, str] = field(default_factory=dict)
    prelaunch_gate: Callable[[], None] | None = None
    wait_for_process: Callable[[subprocess.Popen[Any], float], int] | None = None
    test_only_allow_unpinned: bool = False
    production_confirmed: bool = False
    maintenance_owner: str | None = None
    run_id: str | None = None


def _process_group_exists(pgid: int) -> bool:
    if pgid <= 0 or pgid == os.getpgrp():
        raise ExecutionError(f"refusing unsafe process-group probe: {pgid}")
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_process_group_gone(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while _process_group_exists(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _terminate_process_group(
    process: subprocess.Popen[Any], pgid: int, grace_seconds: float
) -> dict[str, Any]:
    cleanup_started = time.time()
    # poll() reaps an exited leader before the group probe.  If the group still
    # exists after that, it necessarily contains an unexpected descendant.
    leader_was_done = process.poll() is not None
    existed = _process_group_exists(pgid)
    term_sent = False
    kill_sent = False
    if existed:
        try:
            os.killpg(pgid, signal.SIGTERM)
            term_sent = True
        except ProcessLookupError:
            pass
    if process.poll() is None:
        try:
            process.wait(timeout=max(0.0, grace_seconds))
        except subprocess.TimeoutExpired:
            pass
    gone = _wait_process_group_gone(pgid, grace_seconds)
    if not gone:
        try:
            os.killpg(pgid, signal.SIGKILL)
            kill_sent = True
        except ProcessLookupError:
            pass
        if process.poll() is None:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
        gone = _wait_process_group_gone(pgid, 5.0)
    if process.poll() is None:
        # This can only be a start_new_session race; never leave the leader.
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError("engine leader could not be reaped") from exc
    if not gone or _process_group_exists(pgid):
        raise ExecutionError(f"engine process group {pgid} survived TERM/KILL cleanup")
    return {
        "child_pid": process.pid,
        "pgid": pgid,
        "cleanup_started_epoch": cleanup_started,
        "cleanup_finished_epoch": time.time(),
        "leader_was_done": leader_was_done,
        "group_existed_at_cleanup": existed,
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "lingering_after_parent": bool(leader_was_done and existed),
        "gone_before_finalize": True,
        "containment_scope": CONTAINMENT_CONTRACT["scope"],
    }


def _new_run_id() -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"phase0a-{stamp}-{uuid.uuid4().hex[:8]}"


def _update_failure_manifest(run_dir: Path, manifest: dict[str, Any], exc: BaseException) -> None:
    finished = time.time()
    _mark_status(manifest, "failed", finished)
    manifest.update(
        success=False,
        finished_epoch=finished,
        error={"type": type(exc).__name__, "message": str(exc)},
    )
    try:
        write_json_atomic(run_dir / MANIFEST_NAME, manifest)
    except OSError:
        pass


def run_once(request: RunRequest) -> Path:
    real_ids = validate_token_ids(request.real_token_ids)
    if request.timeout_seconds <= 0:
        raise HarnessError("timeout must be positive")
    if request.kill_grace_seconds < 0:
        raise HarnessError("kill grace must be nonnegative")
    if not request.command_prefix:
        raise HarnessError("empty engine command")
    if request.test_only_allow_unpinned:
        if request.engine_identity is None or request.engine_identity.get("test_override") is not True:
            raise HarnessError("test-only unpinned execution requires an explicit test_override identity")
        if request.command_prefix != (sys.executable, str(TEST_FAKE_ENGINE)):
            raise HarnessError("test-only execution is restricted to the workspace fake engine")
        if any(not str(key).startswith("FAKE_") for key in request.extra_child_env):
            raise HarnessError("test-only child environment accepts FAKE_* keys only")
        identity = dict(request.engine_identity)
    else:
        if request.command_prefix != (str(PINNED_ENGINE),):
            raise HarnessError("production run_once forbids an alternate command prefix")
        if request.engine_identity is not None:
            raise HarnessError("production run_once derives identity itself; engine_identity overrides are forbidden")
        if request.extra_child_env:
            raise HarnessError("production run_once forbids extra child environment variables")
        if request.prelaunch_gate is not None or request.wait_for_process is not None:
            raise HarnessError("production run_once forbids test gate/wait injection seams")
        _require_production_inactive(request.production_confirmed, request.maintenance_owner)
        identity = assert_pinned_engine()

    output_root = request.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    budget = sum(expected_artifact_bytes(len(real_ids), request.include_verifier_routing).values())
    free = shutil.disk_usage(output_root).free
    if free < budget + 64 * 1024 * 1024:
        raise HarnessError(f"insufficient free space: need at least {budget + 64 * 1024 * 1024}, found {free}")

    run_id = request.run_id or _new_run_id()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise HarnessError(f"invalid run id: {run_id!r}")
    partial = output_root / f"{run_id}.partial"
    final = output_root / run_id
    if partial.exists() or final.exists():
        raise HarnessError(f"run destination already exists for {run_id}")
    partial.mkdir(mode=0o750)

    started_epoch = time.time()
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "success": False,
        "run_id": run_id,
        "launch_directory": str(partial),
        "started_epoch": started_epoch,
        "timed_out": False,
        "engine": identity,
        "containment": dict(CONTAINMENT_CONTRACT),
        "prompt": {
            "real_token_ids": list(real_ids),
            "real_token_count": len(real_ids),
            "sentinel_token": SENTINEL_TOKEN,
            "launch_token_count": len(real_ids) + 1,
        },
        "requested_artifacts": {
            "prefill_final_r": True,
            "prefill_routes": True,
            "verifier_routing": request.include_verifier_routing,
        },
        "expected_bytes": expected_artifact_bytes(len(real_ids), request.include_verifier_routing),
        "claims": {
            "validated": [
                "native prefill final residual BF16 structure",
                "native prefill router ids/normalized weights structure",
            ],
            "not_claimed": [
                "layer ladders",
                "logpos or token log-probabilities",
                "causal or scientific conclusions from router traces",
                "true verifier routing weights",
            ],
        },
    }
    _mark_status(manifest, "created", started_epoch)
    write_json_atomic(partial / MANIFEST_NAME, manifest)

    try:
        launched_ids = write_tokens(partial / TOKENS_NAME, real_ids)
        verifier_path = partial / VERIFIER_ROUTING_NAME if request.include_verifier_routing else None
        argv = list(request.command_prefix) + build_model_args(partial / TOKENS_NAME, verifier_path)
        env, controlled_env = build_child_env(partial, extra_env=request.extra_child_env)
        manifest.update(
            argv=argv,
            controlled_env=controlled_env,
            sanitized_env_prefixes=[
                "STRATA_*", "CUDA_*", "LD_*", "HIP_*", "HSA_*", "ROCR_*", "NVIDIA_*"
            ],
            launched_token_ids=list(launched_ids),
        )
        _mark_status(manifest, "launching")
        if request.test_only_allow_unpinned:
            manifest["test_command_prefix"] = list(request.command_prefix)
        write_json_atomic(partial / MANIFEST_NAME, manifest)

        # For the real CLI this repeats the fail-closed service/HOLD/port
        # check immediately before Popen, after slower tokenization and hashing.
        if request.test_only_allow_unpinned:
            if request.prelaunch_gate is not None:
                request.prelaunch_gate()
            gate_evidence: dict[str, Any] = {
                "mode": "test-only",
                "passed": True,
                "callback_checked": request.prelaunch_gate is not None,
            }
        else:
            gate_evidence = _require_production_inactive(
                request.production_confirmed, request.maintenance_owner
            )
        gate_epoch = time.time()
        gate_evidence["checked_epoch"] = gate_epoch
        manifest["prelaunch_gate_epoch"] = gate_epoch
        manifest["gate_evidence"] = gate_evidence
        _mark_status(manifest, "gated", gate_epoch)
        write_json_atomic(partial / MANIFEST_NAME, manifest)

        process: subprocess.Popen[Any] | None = None
        intended_pgid: int | None = None
        return_code: int | None = None
        primary_error: BaseException | None = None
        group_receipt: dict[str, Any] | None = None
        guard = _LaunchSignalGuard()
        with (partial / STDOUT_NAME).open("wb") as stdout_handle, (
            partial / STDERR_NAME
        ).open("wb") as stderr_handle:
            try:
                guard.install()
                try:
                    # Defer raising only across Popen and the non-fallible
                    # intended-PGID assignment.  Signals are not OS-masked, so
                    # the exec'd engine inherits normal signal semantics.
                    guard.defer_for_popen()
                    process = subprocess.Popen(
                        argv,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        start_new_session=True,
                    )
                    intended_pgid = process.pid
                    manifest.update(
                        child_pid=process.pid,
                        child_pgid=intended_pgid,
                        popen_epoch=time.time(),
                    )
                finally:
                    guard.finish_popen()

                # Everything after successful Popen is covered by the finally
                # below.  Observation or manifest-write failures therefore use
                # the intended PGID supplied by start_new_session=True.
                observed_pgid = os.getpgid(process.pid)
                manifest["observed_child_pgid"] = observed_pgid
                if observed_pgid != intended_pgid:
                    raise ExecutionError(
                        f"detached engine PGID mismatch: observed {observed_pgid}, "
                        f"expected {intended_pgid}"
                    )
                _mark_status(manifest, "running")
                write_json_atomic(partial / MANIFEST_NAME, manifest)
                waiter = request.wait_for_process
                if waiter is None:
                    return_code = process.wait(timeout=request.timeout_seconds)
                else:
                    return_code = waiter(process, request.timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                manifest["timed_out"] = True
                primary_error = ExecutionTimeout(
                    f"engine exceeded {request.timeout_seconds:.3f}s and was terminated"
                )
                primary_error.__cause__ = exc
            except BaseException as exc:
                primary_error = exc
                if isinstance(exc, ExecutionSignal):
                    manifest["termination_signal"] = {
                        "number": exc.signum,
                        "name": exc.signal_name,
                        "received_epoch": exc.received_epoch,
                    }
            finally:
                # Repeat INT/TERM/HUP/QUIT cannot interrupt teardown.  Restore
                # the caller's handlers only after cleanup and log durability.
                try:
                    guard.begin_cleanup()
                    if process is not None:
                        if intended_pgid is None:
                            # Popen succeeded, so this is the only safe fallback
                            # even if a later Python operation was interrupted.
                            intended_pgid = process.pid
                        try:
                            group_receipt = _terminate_process_group(
                                process, intended_pgid, request.kill_grace_seconds
                            )
                        except BaseException as cleanup_exc:
                            cleanup_finished = time.time()
                            group_receipt = {
                                "child_pid": process.pid,
                                "pgid": intended_pgid,
                                "cleanup_started_epoch": cleanup_finished,
                                "cleanup_finished_epoch": cleanup_finished,
                                "leader_was_done": process.poll() is not None,
                                "group_existed_at_cleanup": True,
                                "term_sent": False,
                                "kill_sent": False,
                                "lingering_after_parent": False,
                                "gone_before_finalize": False,
                                "containment_scope": CONTAINMENT_CONTRACT["scope"],
                                "cleanup_error": (
                                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                                ),
                            }
                            if primary_error is None:
                                primary_error = cleanup_exc
                        manifest["process_group"] = group_receipt
                        if return_code is not None:
                            manifest["engine_return_code"] = return_code
                        elif process.returncode is not None:
                            manifest["cleanup_return_code"] = process.returncode
                        if group_receipt.get("gone_before_finalize") is True:
                            _mark_status(
                                manifest,
                                "cleaned",
                                group_receipt["cleanup_finished_epoch"],
                            )
                    for handle in (stdout_handle, stderr_handle):
                        try:
                            _flush_and_fsync_handle(handle)
                        except BaseException as fsync_exc:
                            if primary_error is None:
                                primary_error = fsync_exc
                    try:
                        write_json_atomic(partial / MANIFEST_NAME, manifest)
                    except BaseException as manifest_exc:
                        if primary_error is None:
                            primary_error = manifest_exc
                finally:
                    try:
                        guard.restore()
                    except BaseException as restore_exc:
                        if primary_error is None:
                            primary_error = restore_exc

        if primary_error is not None:
            raise primary_error
        if process is None or intended_pgid is None or group_receipt is None:
            raise ExecutionError("engine launch completed without a cleanup receipt")
        if group_receipt.get("gone_before_finalize") is not True:
            raise ExecutionError("engine process group was not proven gone before validation")
        if return_code is None:
            raise ExecutionError("engine wait returned no exit status")
        if group_receipt.get("lingering_after_parent"):
            raise ExecutionError("engine parent exited while members remained in its original process group")
        if return_code != 0:
            raise ExecutionError(f"engine returned {return_code}")

        _mark_status(manifest, "validating")
        write_json_atomic(partial / MANIFEST_NAME, manifest)
        validation = validate_artifacts(partial, len(real_ids), request.include_verifier_routing)
        receipt_names = list(expected_artifact_bytes(len(real_ids), request.include_verifier_routing))
        receipts = _artifact_receipts(partial, receipt_names)
        validation_epoch = time.time()
        manifest.update(
            validation=validation,
            artifact_receipts=receipts,
            validation_epoch=validation_epoch,
        )
        _mark_status(manifest, "validated_pending_finalize", validation_epoch)

        durable_files = [
            TOKENS_NAME,
            STDOUT_NAME,
            STDERR_NAME,
            *receipt_names,
            MANIFEST_NAME,
        ]
        manifest["durability"] = {"completed": False, "files": durable_files}
        write_json_atomic(partial / MANIFEST_NAME, manifest)
        for name in durable_files:
            _fsync_regular_file(partial / name)
        _fsync_directory(partial)
        durability_epoch = time.time()
        manifest["durability"] = {
            "completed": True,
            "completed_epoch": durability_epoch,
            "files": durable_files,
        }
        write_json_atomic(partial / MANIFEST_NAME, manifest)
        _fsync_regular_file(partial / MANIFEST_NAME)
        _fsync_directory(partial)

        os.replace(partial, final)
        _fsync_directory(output_root)
        renamed_epoch = time.time()
        finished_epoch = time.time()
        manifest.update(success=True, renamed_epoch=renamed_epoch, finished_epoch=finished_epoch)
        _mark_status(manifest, "complete", finished_epoch)
        write_json_atomic(final / MANIFEST_NAME, manifest)
        _fsync_regular_file(final / MANIFEST_NAME)
        _fsync_directory(final)
        _fsync_directory(output_root)
        return final
    except BaseException as exc:
        run_dir = partial if partial.exists() else final
        _update_failure_manifest(run_dir, manifest, exc)
        raise


def make_plan(
    real_token_ids: Sequence[int],
    include_verifier_routing: bool,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ids = validate_token_ids(real_token_ids)
    engine_identity = dict(identity) if identity is not None else {
        "path": str(PINNED_ENGINE), "sha256": PINNED_ENGINE_SHA256,
        "version": PINNED_VERSION, "source": PINNED_SOURCE,
        "plan_only": True, "input_attestation_verified": False,
    }
    placeholder = Path("<unique-run>.partial")
    verifier = placeholder / VERIFIER_ROUTING_NAME if include_verifier_routing else None
    argv = [str(PINNED_ENGINE)] + build_model_args(placeholder / TOKENS_NAME, verifier)
    _, controlled_env = build_child_env(placeholder, base_env={})
    return {
        "engine": engine_identity,
        "containment": dict(CONTAINMENT_CONTRACT),
        "real_token_count": len(ids),
        "sentinel_token": SENTINEL_TOKEN,
        "launch_token_count": len(ids) + 1,
        "argv": argv,
        "controlled_env": controlled_env,
        "expected_bytes": expected_artifact_bytes(len(ids), include_verifier_routing),
        "production_lifecycle": (
            "not implemented; run-offline requires a matching pre-existing recovery HOLD and proves "
            f"loaded/inactive/dead, MainPID=0, and no TCP listener on port {PRODUCTION_PORT}"
        ),
        "not_collected": ["layer ladders", "logpos"],
        "verifier_routing_classification": (
            "non-research plumbing only" if include_verifier_routing else "not requested"
        ),
    }


def _add_token_source(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prompt-id", help="prompt id in the pinned corpus (default: numbers_01)")
    group.add_argument("--token-ids", help="comma/space-separated real prompt token ids")


def _resolve_cli_tokens(args: argparse.Namespace) -> tuple[int, ...]:
    if args.token_ids is not None:
        return parse_token_ids(args.token_ids)
    if args.command == "plan":
        return reference_prompt_tokens(args.prompt_id or "numbers_01")
    return tokenize_prompt(args.prompt_id or "numbers_01")


def _parse_systemd_show(output: str) -> dict[str, str]:
    properties: dict[str, str] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise HarnessError(f"cannot prove production is inactive: malformed systemd property {line!r}")
        key, value = line.split("=", 1)
        if key in properties:
            raise HarnessError(f"cannot prove production is inactive: duplicate systemd property {key!r}")
        properties[key] = value
    required = {"LoadState", "ActiveState", "SubState", "MainPID"}
    if set(properties) != required:
        raise HarnessError(
            "cannot prove production is inactive: systemd properties were "
            f"{sorted(properties)}, expected {sorted(required)}"
        )
    return properties


def _read_maintenance_owner(hold_path: Path) -> str:
    """Read an existing private structured lease without following links.

    The external controller, not this observation, owns synchronization and
    restoration.  Raw owner strings are not a substitute for its lease.
    """
    def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate maintenance lease field")
            result[key] = value
        return result

    try:
        fd = os.open(hold_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if (not stat.S_ISREG(before.st_mode)
                    or before.st_uid not in (0, os.geteuid())
                    or stat.S_IMODE(before.st_mode) & 0o077
                    or before.st_size > 4096):
                raise ValueError("HOLD must be a private regular lease owned by root or this operator")
            raw = stream.read(4097)
            after = os.fstat(stream.fileno())
            current = hold_path.stat(follow_symlinks=False)
            identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                      value.st_mtime_ns, value.st_ctime_ns)
            if len(raw) > 4096 or identity(before) != identity(after) or identity(after) != identity(current):
                raise ValueError("HOLD changed while its owner was being read")
        lease = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_fields)
        if (not isinstance(lease, dict)
                or type(lease.get("version")) is not int or lease["version"] != 1
                or lease.get("kind") != "owned-strata-maintenance"
                or not isinstance(lease.get("owner"), str)
                or not re.fullmatch(r"[A-Za-z0-9._:-]{1,100}", lease["owner"])):
            raise ValueError("HOLD is not a version-1 owned Strata maintenance lease")
        return lease["owner"]
    except (OSError, ValueError) as exc:
        raise HarnessError(f"cannot prove an owned maintenance window: {hold_path}: {exc}") from exc


def _require_production_inactive(
    confirmed: bool,
    maintenance_owner: str | None,
    *,
    command_runner: Any = subprocess.run,
    hold_path: Path | None = None,
) -> dict[str, Any]:
    hold_path = RECOVERY_HOLD if hold_path is None else hold_path
    if not confirmed:
        raise HarnessError(
            "real execution is disabled unless --confirm-production-stopped is explicitly supplied"
        )
    if not maintenance_owner:
        raise HarnessError(
            "real execution also requires --maintenance-owner matching a pre-existing recovery HOLD"
        )
    hold_owner = _read_maintenance_owner(hold_path)
    if hold_owner != maintenance_owner:
        raise HarnessError(
            f"maintenance HOLD owner mismatch: file has {hold_owner!r}, request has {maintenance_owner!r}"
        )

    try:
        status = command_runner(
            [
                "systemctl", "show", PRODUCTION_UNIT, "--no-pager",
                "--property=LoadState", "--property=ActiveState",
                "--property=SubState", "--property=MainPID",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HarnessError(f"cannot prove production is inactive: {exc}") from exc
    if status.returncode != 0:
        raise HarnessError(
            f"cannot prove production is inactive: systemctl show returned {status.returncode}"
        )
    if status.stderr.strip():
        raise HarnessError(
            f"cannot prove production is inactive: systemctl show wrote stderr: {status.stderr.strip()!r}"
        )
    properties = _parse_systemd_show(status.stdout)
    try:
        main_pid = int(properties["MainPID"], 10)
    except ValueError as exc:
        raise HarnessError(
            f"cannot prove production is inactive: invalid MainPID {properties['MainPID']!r}"
        ) from exc
    expected = {"LoadState": "loaded", "ActiveState": "inactive", "SubState": "dead"}
    wrong = {key: properties[key] for key, value in expected.items() if properties[key] != value}
    if wrong or main_pid != 0:
        raise HarnessError(
            "refusing offline launch: production is not exactly loaded/inactive/dead with MainPID=0; "
            f"observed {properties!r}"
        )

    try:
        listeners = command_runner(
            ["ss", "-H", "-ltn", f"sport = :{PRODUCTION_PORT}"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HarnessError(f"cannot prove port {PRODUCTION_PORT} is unowned: {exc}") from exc
    if listeners.returncode != 0:
        raise HarnessError(
            f"cannot prove port {PRODUCTION_PORT} is unowned: ss returned {listeners.returncode}"
        )
    if listeners.stderr.strip():
        raise HarnessError(
            f"cannot prove port {PRODUCTION_PORT} is unowned: ss wrote stderr: {listeners.stderr.strip()!r}"
        )
    if listeners.stdout.strip():
        raise HarnessError(f"refusing offline launch: TCP port {PRODUCTION_PORT} still has a listener")
    return {
        "mode": "production-quiescence",
        "passed": True,
        "unit": PRODUCTION_UNIT,
        "load_state": properties["LoadState"],
        "active_state": properties["ActiveState"],
        "sub_state": properties["SubState"],
        "main_pid": main_pid,
        "port": PRODUCTION_PORT,
        "listener_present": False,
        "hold_owner_confirmed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=EXAMPLE_PROFILE,
                        help="operator profile; the shipped template permits plans only")
    parser.add_argument("--profile-sha256", help="separately reviewed SHA-256 of an attested operator profile")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="print a non-mutating pinned launch plan")
    _add_token_source(plan)
    plan.add_argument("--include-verifier-routing", action="store_true")

    validate = sub.add_parser("validate", help="validate an existing run directory without modifying it")
    validate.add_argument("run_dir", type=Path)

    run = sub.add_parser("run-offline", help="explicitly launch once; never manages production")
    _add_token_source(run)
    run.add_argument("--output-root", type=Path, default=WORKSPACE / "runs")
    run.add_argument("--include-verifier-routing", action="store_true")
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--confirm-production-stopped", action="store_true")
    run.add_argument(
        "--maintenance-owner",
        help=f"must exactly match the existing owner in {RECOVERY_HOLD}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        configure_profile(args.profile, args.profile_sha256)
        if args.command == "plan":
            payload = make_plan(_resolve_cli_tokens(args), args.include_verifier_routing)
        elif args.command == "validate":
            payload = validate_run_dir(args.run_dir.resolve())
        elif args.command == "run-offline":
            _load_input_contract()  # refuse unreviewed profiles before tokenization
            real_token_ids = _resolve_cli_tokens(args)
            request = RunRequest(
                real_token_ids=real_token_ids,
                output_root=args.output_root,
                include_verifier_routing=args.include_verifier_routing,
                timeout_seconds=args.timeout,
                production_confirmed=args.confirm_production_stopped,
                maintenance_owner=args.maintenance_owner,
            )
            completed = run_once(request)
            payload = {"success": True, "run_dir": str(completed)}
        else:
            raise AssertionError(args.command)
    except ExecutionSignal as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 128 + exc.signum
    except (HarnessError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
