#!/usr/bin/env python3
"""Stage, validate, and transactionally install the pinned SGLang overlays."""

from __future__ import annotations

import argparse
import hashlib
import os
import py_compile
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable


EXPECTED_COMMIT = "73a255206f916366c8d26d4022f82ddfb0ab558d"
QWEN_REL = Path("python/sglang/srt/models/qwen4_exp.py")
QSA_REL = Path(
    "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
)
LOADER_REL = Path("python/sglang/srt/models/qwen4_ple_w4_sidecar.py")

EXPECTED_QWEN_BASE = "f406977eb2373937393241f453477867f7dc943bd4839216db8fe66fa9f921d8"
EXPECTED_QWEN_W4 = "95bb2059669f9a66abe1be8e271037eeaa11ea534ace808f1c171ca78fee101f"
EXPECTED_QWEN_FINAL = "b54de0a07a16a7a3070aabead3c53b80f108d89528c30763ca8e032f330d97ac"
EXPECTED_QSA_BASE = "c959835d05d0f395ad7eae4330cf264af9f6f7c1bff3d45a39bb953d2536f5f2"
EXPECTED_QSA_FINAL = "584e2acdce11c6e1e6dc50b9f61ca8018a3634e1bafb3d079b367fc30d1f7634"
EXPECTED_LOADER = "9dbace396f69ca2a319c7ae9cf74380549b62b6cdcc9f88135776552c245bb68"

EXPECTED_RELEASE_INPUTS = {
    Path("patches/0001-qwen4-exp-w4-ple.patch"): (
        "55a1cee2968c0938e79d4e4d552c35b310bfd62e66ed54ca7fc3070a7ce19032"
    ),
    Path("patches/0002-qwen4-exp-sidecar.patch"): (
        "938c43e9d283883ffc285c4e5e3a03967ebc0167480767606131fe361f1d8644"
    ),
    Path("patches/0003-qsa-sm120-xqa.patch"): (
        "ddcee6ebdb2f52653c3cc0f64a234e59bce6054b487467ff4edd4f69d4371f02"
    ),
    Path("scripts/build_sidecar_overlay.py"): (
        "c47a38ab2cf1b31934bf18bd8a367233164ecac1ffb2f80ee51884cbd6a6477e"
    ),
    Path("src/qwen4_ple_w4_sidecar.py"): EXPECTED_LOADER,
}

INJECT_ENV = "SGLANG_QWEN4_PLE_PATCH_TEST_FAIL_AFTER"
INJECT_STAGES = {"staged", "qwen", "qsa", "loader"}
ROLLBACK_INJECT_ENV = "SGLANG_QWEN4_PLE_PATCH_TEST_FAIL_ROLLBACK"
ROLLBACK_INJECT_STAGES = {
    "loader_unlink",
    "qsa_replace",
    "qsa_fsync",
    "qwen_replace",
    "qwen_fsync",
}


class RecoveryRequiredError(RuntimeError):
    """Raised when automatic rollback cannot be verified as complete."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular_file(path: Path, expected: str, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is not a regular non-symlink file: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def run(command: list[str], *, stdin_path: Path | None = None) -> None:
    if stdin_path is None:
        subprocess.run(command, check=True)
        return
    with stdin_path.open("rb", buffering=0) as reader:
        subprocess.run(command, stdin=reader, check=True)


def apply_patch(stage_root: Path, patch_path: Path) -> None:
    run(
        [
            "git",
            "apply",
            "--unsafe-paths",
            f"--directory={stage_root}",
            "--whitespace=nowarn",
            str(patch_path),
        ],
    )


def fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDWR)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def maybe_inject_failure(stage: str) -> None:
    requested = os.environ.get(INJECT_ENV)
    if requested and requested not in INJECT_STAGES:
        raise RuntimeError(
            f"invalid {INJECT_ENV}={requested!r}; expected one of {sorted(INJECT_STAGES)}"
        )
    if requested == stage:
        raise RuntimeError(f"injected test failure after {stage}")


def maybe_inject_rollback_failure(stage: str) -> None:
    requested = os.environ.get(ROLLBACK_INJECT_ENV)
    if requested and requested not in ROLLBACK_INJECT_STAGES:
        raise RuntimeError(
            f"invalid {ROLLBACK_INJECT_ENV}={requested!r}; "
            f"expected one of {sorted(ROLLBACK_INJECT_STAGES)}"
        )
    if requested == stage:
        raise RuntimeError(f"injected rollback failure at {stage}")


def require_pristine_target(sglang_dir: Path) -> tuple[Path, Path, Path]:
    inside = subprocess.run(
        ["git", "-C", str(sglang_dir), "rev-parse", "--is-inside-work-tree"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if inside != "true":
        raise RuntimeError(f"not a Git worktree: {sglang_dir}")
    commit = subprocess.run(
        ["git", "-C", str(sglang_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"SGLang HEAD {commit} != {EXPECTED_COMMIT}")

    qwen_file = sglang_dir / QWEN_REL
    qsa_file = sglang_dir / QSA_REL
    loader_file = sglang_dir / LOADER_REL
    if loader_file.exists() or loader_file.is_symlink():
        raise RuntimeError(f"sidecar loader already exists: {loader_file}")
    require_regular_file(qwen_file, EXPECTED_QWEN_BASE, "Qwen4-Exp base")
    require_regular_file(qsa_file, EXPECTED_QSA_BASE, "QSA base")
    return qwen_file, qsa_file, loader_file


def verify_release_inputs(release_dir: Path) -> dict[Path, Path]:
    verified = {}
    for relative, expected in EXPECTED_RELEASE_INPUTS.items():
        path = release_dir / relative
        require_regular_file(path, expected, f"release input {relative.as_posix()}")
        verified[relative] = path
    return verified


def compile_staged_files(paths: list[Path], cache_dir: Path) -> None:
    cache_dir.mkdir()
    for index, path in enumerate(paths):
        py_compile.compile(
            str(path),
            cfile=str(cache_dir / f"{index}.pyc"),
            doraise=True,
        )


def seal_release_inputs(
    stage_root: Path, release_inputs: dict[Path, Path]
) -> dict[Path, Path]:
    sealed = {}
    sealed_root = stage_root / "sealed-inputs"
    for relative, source in release_inputs.items():
        destination = sealed_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        require_regular_file(
            destination,
            EXPECTED_RELEASE_INPUTS[relative],
            f"sealed release input {relative.as_posix()}",
        )
        sealed[relative] = destination
    return sealed


def stage_outputs(
    stage_root: Path,
    qwen_file: Path,
    qsa_file: Path,
    release_inputs: dict[Path, Path],
) -> dict[str, Path]:
    release_inputs = seal_release_inputs(stage_root, release_inputs)
    staged_qwen = stage_root / QWEN_REL
    staged_qsa = stage_root / QSA_REL
    staged_qwen.parent.mkdir(parents=True)
    staged_qsa.parent.mkdir(parents=True)
    shutil.copy2(qwen_file, staged_qwen)
    shutil.copy2(qsa_file, staged_qsa)

    rollback_dir = stage_root / "rollback"
    rollback_dir.mkdir()
    rollback_qwen = rollback_dir / "qwen4_exp.py"
    rollback_qsa = rollback_dir / "qwen_sparse_attn_backend.py"
    shutil.copy2(qwen_file, rollback_qwen)
    shutil.copy2(qsa_file, rollback_qsa)

    apply_patch(
        stage_root,
        release_inputs[Path("patches/0001-qwen4-exp-w4-ple.patch")],
    )
    require_regular_file(staged_qwen, EXPECTED_QWEN_W4, "staged Qwen4-Exp W4")

    final_qwen = stage_root / "final-qwen4_exp.py"
    run(
        [
            sys.executable,
            str(release_inputs[Path("scripts/build_sidecar_overlay.py")]),
            "--base",
            str(staged_qwen),
            "--patch",
            str(release_inputs[Path("patches/0002-qwen4-exp-sidecar.patch")]),
            "--output",
            str(final_qwen),
        ]
    )
    apply_patch(
        stage_root,
        release_inputs[Path("patches/0003-qsa-sm120-xqa.patch")],
    )
    staged_loader = stage_root / "qwen4_ple_w4_sidecar.py"
    shutil.copy2(
        release_inputs[Path("src/qwen4_ple_w4_sidecar.py")], staged_loader
    )

    qwen_mode = stat.S_IMODE(qwen_file.stat().st_mode)
    qsa_mode = stat.S_IMODE(qsa_file.stat().st_mode)
    os.chmod(final_qwen, qwen_mode)
    os.chmod(staged_qsa, qsa_mode)
    os.chmod(staged_loader, 0o644)

    compile_staged_files(
        [final_qwen, staged_qsa, staged_loader], stage_root / "pycache"
    )
    require_regular_file(final_qwen, EXPECTED_QWEN_FINAL, "staged Qwen4-Exp final")
    require_regular_file(staged_qsa, EXPECTED_QSA_FINAL, "staged QSA final")
    require_regular_file(staged_loader, EXPECTED_LOADER, "staged sidecar loader")
    for path in (
        final_qwen,
        staged_qsa,
        staged_loader,
        rollback_qwen,
        rollback_qsa,
    ):
        fsync_file(path)
    maybe_inject_failure("staged")
    return {
        "qwen": final_qwen,
        "qsa": staged_qsa,
        "loader": staged_loader,
        "rollback_qwen": rollback_qwen,
        "rollback_qsa": rollback_qsa,
    }


def attempt_rollback(
    label: str, errors: list[str], action: Callable[[], None]
) -> None:
    try:
        action()
    except BaseException as exc:
        errors.append(f"{label}: {exc}")


def restore_from_backup(
    backup: Path, target: Path, injection_prefix: str
) -> None:
    replacement = backup.parent / f".{target.name}.{injection_prefix}.restore"
    shutil.copy2(backup, replacement)
    fsync_file(replacement)
    maybe_inject_rollback_failure(f"{injection_prefix}_replace")
    os.replace(replacement, target)
    maybe_inject_rollback_failure(f"{injection_prefix}_fsync")
    fsync_file(target)


def path_receipt(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if not path.exists():
        return "absent"
    if not path.is_file():
        return "not-a-regular-file"
    try:
        return f"sha256={sha256_file(path)}"
    except OSError as exc:
        return f"unreadable={exc}"


def write_recovery_instructions(
    stage_root: Path,
    errors: list[str],
    qwen_file: Path,
    qsa_file: Path,
    loader_file: Path,
) -> Path:
    instructions = stage_root / "RECOVERY.txt"
    rollback_qwen = stage_root / "rollback/qwen4_exp.py"
    rollback_qsa = stage_root / "rollback/qwen_sparse_attn_backend.py"
    content = "\n".join(
        [
            "Qwen4 PLE patch transaction recovery required",
            "",
            f"Pinned SGLang commit: {EXPECTED_COMMIT}",
            "Do not rerun the installer or start SGLang until recovery is complete.",
            "Stop processes that may hold these files open, then restore the two",
            "retained pristine files and remove the loader if it exists.",
            "",
            f"Qwen target: {qwen_file}",
            f"Qwen current: {path_receipt(qwen_file)}",
            f"Qwen recovery copy: {rollback_qwen}",
            f"Qwen expected: sha256={EXPECTED_QWEN_BASE}",
            "",
            f"QSA target: {qsa_file}",
            f"QSA current: {path_receipt(qsa_file)}",
            f"QSA recovery copy: {rollback_qsa}",
            f"QSA expected: sha256={EXPECTED_QSA_BASE}",
            "",
            f"Loader target (must be absent): {loader_file}",
            f"Loader current: {path_receipt(loader_file)}",
            "",
            "Recorded rollback failures:",
            *[f"- {error}" for error in errors],
            "",
            "After manual restoration, verify both expected hashes, confirm the",
            "loader is absent, and confirm `git status --porcelain --untracked-files=all`",
            "shows only this retained recovery directory before removing it.",
            "",
        ]
    )
    instructions.write_text(content, encoding="utf-8", newline="\n")
    fsync_file(instructions)
    fsync_directory(stage_root)
    return instructions


def install_transaction(
    stage_root: Path,
    outputs: dict[str, Path],
    qwen_file: Path,
    qsa_file: Path,
    loader_file: Path,
) -> None:
    require_regular_file(qwen_file, EXPECTED_QWEN_BASE, "Qwen4-Exp pre-install")
    require_regular_file(qsa_file, EXPECTED_QSA_BASE, "QSA pre-install")
    if loader_file.exists() or loader_file.is_symlink():
        raise RuntimeError(f"sidecar loader appeared before install: {loader_file}")

    installed: list[str] = []
    try:
        installed.append("qwen")
        os.replace(outputs["qwen"], qwen_file)
        fsync_file(qwen_file)
        maybe_inject_failure("qwen")

        installed.append("qsa")
        os.replace(outputs["qsa"], qsa_file)
        fsync_file(qsa_file)
        maybe_inject_failure("qsa")

        installed.append("loader")
        os.replace(outputs["loader"], loader_file)
        fsync_file(loader_file)
        fsync_directory(qwen_file.parent)
        fsync_directory(qsa_file.parent)
        maybe_inject_failure("loader")

        require_regular_file(qwen_file, EXPECTED_QWEN_FINAL, "Qwen4-Exp final")
        require_regular_file(qsa_file, EXPECTED_QSA_FINAL, "QSA final")
        require_regular_file(loader_file, EXPECTED_LOADER, "sidecar loader")
    except BaseException as exc:
        rollback_errors: list[str] = []

        if "loader" in installed:
            def remove_loader() -> None:
                maybe_inject_rollback_failure("loader_unlink")
                if loader_file.exists() or loader_file.is_symlink():
                    loader_file.unlink()

            attempt_rollback("loader removal", rollback_errors, remove_loader)
        if "qsa" in installed:
            attempt_rollback(
                "QSA restore",
                rollback_errors,
                lambda: restore_from_backup(
                    outputs["rollback_qsa"], qsa_file, "qsa"
                ),
            )
        if "qwen" in installed:
            attempt_rollback(
                "Qwen restore",
                rollback_errors,
                lambda: restore_from_backup(
                    outputs["rollback_qwen"], qwen_file, "qwen"
                ),
            )
        attempt_rollback(
            "Qwen directory fsync",
            rollback_errors,
            lambda: fsync_directory(qwen_file.parent),
        )
        attempt_rollback(
            "QSA directory fsync",
            rollback_errors,
            lambda: fsync_directory(qsa_file.parent),
        )

        verification_errors: list[str] = []
        attempt_rollback(
            "Qwen rollback verification",
            verification_errors,
            lambda: require_regular_file(
                qwen_file, EXPECTED_QWEN_BASE, "rolled-back Qwen4-Exp"
            ),
        )
        attempt_rollback(
            "QSA rollback verification",
            verification_errors,
            lambda: require_regular_file(
                qsa_file, EXPECTED_QSA_BASE, "rolled-back QSA"
            ),
        )

        def verify_loader_absent() -> None:
            if loader_file.exists() or loader_file.is_symlink():
                raise RuntimeError("rolled-back sidecar loader still exists")

        attempt_rollback(
            "loader rollback verification",
            verification_errors,
            verify_loader_absent,
        )

        recovery_errors = rollback_errors + verification_errors
        if recovery_errors:
            try:
                instructions = write_recovery_instructions(
                    stage_root,
                    recovery_errors,
                    qwen_file,
                    qsa_file,
                    loader_file,
                )
                instructions_message = f" recovery instructions: {instructions}"
            except BaseException as recovery_exc:
                recovery_errors.append(
                    f"writing recovery instructions: {recovery_exc}"
                )
                instructions_message = f" recovery material retained at: {stage_root}"
            raise RecoveryRequiredError(
                "patch transaction failed; automatic rollback was not fully verified;"
                + instructions_message
                + "; failures: "
                + "; ".join(recovery_errors)
            ) from exc
        raise RuntimeError("patch transaction failed; target restored") from exc

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sglang_checkout", type=Path)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve(strict=True).parent
    release_dir = script_dir.parent
    sglang_dir = args.sglang_checkout.expanduser().resolve(strict=True)
    qwen_file, qsa_file, loader_file = require_pristine_target(sglang_dir)
    release_inputs = verify_release_inputs(release_dir)

    stage_root = Path(
        tempfile.mkdtemp(prefix=".qwen4-ple-recovery-", dir=sglang_dir)
    )
    retain_recovery = False
    try:
        outputs = stage_outputs(
            stage_root, qwen_file, qsa_file, release_inputs
        )
        install_transaction(
            stage_root, outputs, qwen_file, qsa_file, loader_file
        )
    except RecoveryRequiredError:
        retain_recovery = True
        raise
    finally:
        if not retain_recovery:
            shutil.rmtree(stage_root)

    print(f"PATCH_SET_VERIFIED {EXPECTED_COMMIT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(78) from exc
