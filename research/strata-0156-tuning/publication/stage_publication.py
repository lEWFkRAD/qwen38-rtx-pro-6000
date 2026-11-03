#!/usr/bin/env python3
"""Plan-first, local-only sanitized publication staging. Never commits or pushes.

Run --self-test without reading receipts. After all GPU work and receipts finish,
run --plan, then --execute --repository PATH --expected-repo-commit COMMIT.
The output is a new isolated publication/.stages tree, never the repository.
"""
from __future__ import annotations

import argparse
import ast
import collections
import datetime as dt
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.parse


ROOT = Path(__file__).resolve().parents[1]
RELATIVE_DEST = Path("research/strata-0156-tuning")
TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".md", ".txt", ".log", ".csv", ".tsv", ".sh", ".ps1", ".patch", ".diff", ".sig"}
SKIP_DIRS = {"assets", "official-engine-0.1.56", "official-control-0.1.56", "__pycache__", ".venv", "venv", "node_modules", ".git", ".stages", ".pytest_cache"}
PRIVATE_NAMES = {"candidate-private.json", "api-key", "api-key.txt", ".env", ".env.local",
                 "sync-stock0156-status.ps1"}
METADATA = {
    "assets/strata-feed.json": "provenance/strata-feed.json",
    "assets/strata-feed.json.sig": "provenance/strata-feed.json.sig",
    "assets/SHA256SUMS.txt": "provenance/upstream-SHA256SUMS.txt",
    "official-engine-0.1.56/engine-component.json": "provenance/engine-component.json",
    "official-engine-0.1.56/BUILD.json": "provenance/engine-BUILD.json",
    "official-control-0.1.56/BUNDLE.json": "provenance/control-BUNDLE.json",
    "official-control-0.1.56/strata_control/update_keys.json": "provenance/upstream-public-update-keys.json",
}
ACTIVE_SCRIPTS = {"benchmark_strata.py", "initial_comparison.py", "mixed_latency.py", "api_canary.py", "api_throughput.py"}
SECRET_KEYS = {"api_key", "api_keys", "compatibility_api_keys", "authorization", "password", "access_token", "refresh_token", "client_secret", "bearer_token", "auth_token"}
HOME_RE = re.compile(r"/home/[A-Za-z0-9_.-]+")
WINDOWS_HOME_RE = re.compile(r"(?i)[A-Z]:\\+Users\\+[^\\\s\"'<>]+")
UUID_RE = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
OWNER_RE = re.compile(r"\bcodex-(?:lsa|strata)[A-Za-z0-9_-]*-\d{8}-[A-Za-z0-9._:-]+\b")
TAILNET_RE = re.compile(r"(?i)\b[a-z0-9.-]+\.tail[a-z0-9]+\.ts\.net\b")
IP_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~-]{24,}")
SECRET_LITERAL_RE = re.compile(r"(?i)\b(?:api[_-]?key|authorization|access[_-]?token|password|client[_-]?secret)\b[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9._~/-]{24,})")
PRIVATE_NETWORKS = tuple(ipaddress.ip_network(x) for x in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "169.254.0.0/16"))
NETWORK_ROOTS = {str(n.network_address) for n in PRIVATE_NETWORKS}


class Refusal(RuntimeError):
    pass


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise Refusal("required read-only state check failed: " + Path(argv[0]).name)
    return result.stdout.strip()


def sanitize_text(text, counts):
    for name, pattern, replacement in (
        ("operator_home", HOME_RE, "/home/USER"),
        ("windows_operator_home", WINDOWS_HOME_RE, lambda _: "C:" + chr(92) + "Users" + chr(92) + "USER"),
        ("uuid", UUID_RE, "<redacted-uuid>"),
        ("maintenance_owner", OWNER_RE, "<maintenance-owner>"),
        ("tailnet_hostname", TAILNET_RE, "fleet-host.invalid"),
        ("bearer_credential", BEARER_RE, "Bearer <redacted-secret>"),
        ("credential_literal", SECRET_LITERAL_RE,
         lambda m: m.group(0)[:-len(m.group(1))] + "REDACTED_SECRET"),
    ):
        text, n = pattern.subn(replacement, text)
        counts[name] += n

    def private_ip(match):
        value = match.group(0)
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            return value
        # Network definitions and loopback bindings are public reproduction code.
        if value not in NETWORK_ROOTS and any(address in n for n in PRIVATE_NETWORKS):
            counts["private_fleet_ip"] += 1
            return "<private-ip>"
        return value

    return IP_RE.sub(private_ip, text).replace("\r\n", "\n").replace("\r", "\n")


def sanitize_json(value, counts):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            public_key = sanitize_text(key, counts)
            if key.lower().replace("-", "_") in SECRET_KEYS and child not in (None, "", [], {}):
                if type(child) is bool:
                    pass  # e.g. health.api_key=true is a feature flag, not a key.
                else:
                    child = "<redacted-secret>"
                    counts["credential_value"] += 1
            if public_key in result:
                raise Refusal("redaction would collide JSON mapping keys")
            result[public_key] = sanitize_json(child, counts)
        return result
    if isinstance(value, list):
        return [sanitize_json(child, counts) for child in value]
    if isinstance(value, str):
        return sanitize_text(value, counts)
    return value


def normalize(raw, suffix):
    counts = collections.Counter()
    text = raw.decode("utf-8")
    if suffix == ".json":
        text = json.dumps(sanitize_json(json.loads(text), counts), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    elif suffix == ".jsonl":
        lines = [json.dumps(sanitize_json(json.loads(line), counts), ensure_ascii=False, allow_nan=False)
                 for line in text.splitlines() if line.strip()]
        text = "\n".join(lines) + ("\n" if lines else "")
    else:
        text = sanitize_text(text, counts)
        if text and not text.endswith("\n"):
            text += "\n"
    if BEARER_RE.search(text) or SECRET_LITERAL_RE.search(text):
        raise Refusal("a possible literal credential remains after sanitizing")
    return text.encode("utf-8"), dict(sorted((k, v) for k, v in counts.items() if v))


def inventory():
    selected, skipped = [], []
    for directory, subdirs, names in os.walk(ROOT, followlinks=False):
        parent = Path(directory)
        for name in list(subdirs):
            path = parent / name
            if name in SKIP_DIRS or name.startswith(".stages") or path.is_symlink():
                subdirs.remove(name)
                skipped.append({"path": path.relative_to(ROOT).as_posix(), "reason": "vendor/cache/staging or symlink directory"})
        for name in sorted(names):
            path = parent / name
            relative = path.relative_to(ROOT).as_posix()
            private_operations = ("wiki" in name.casefold() or "honcho" in name.casefold()
                                  or name.casefold().startswith("continuity."))
            if private_operations:
                skipped.append({"path": relative, "reason": "private fleet continuity/memory operations; research facts are in results/final-state"})
            elif path.is_symlink() or name in PRIVATE_NAMES or name.startswith(".") or name.endswith((".lock", ".part", ".pyc")):
                skipped.append({"path": relative, "reason": "private config, symlink, cache, lock or partial"})
            elif path.suffix.lower() in TEXT_SUFFIXES:
                selected.append((path, relative, False))
            elif name == "fresh-square.png" and relative.startswith("receipts/"):
                selected.append((path, relative, True))
            else:
                raise Refusal("unclassified file needs an explicit publication decision: " + relative)
    for source, destination in METADATA.items():
        path = ROOT / source
        if path.is_file() and not path.is_symlink():
            selected.append((path, destination, False))
    return sorted(selected, key=lambda item: item[1]), sorted(skipped, key=lambda item: item["path"])


def assert_quiescent():
    if run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]):
        raise Refusal("GPU compute work is still resident; publication staging must wait")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            words = [os.fsdecode(x) for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
        except (OSError, PermissionError):
            continue
        if any(Path(word).name in ACTIVE_SCRIPTS for word in words):
            raise Refusal("a benchmark/API driver is still active; publication staging must wait")
    if list((ROOT / "receipts").rglob("candidate-private.json")):
        raise Refusal("a private API candidate configuration still exists")
    for path in (ROOT / "receipts").rglob("*report.json"):
        value = json.loads(path.read_text())
        if isinstance(value, dict) and value.get("status") in {"running", "starting"} and not value.get("finished_at"):
            raise Refusal("a receipt report is unfinished: " + path.relative_to(ROOT).as_posix())


def repository_snapshot(path, expected):
    path = path.resolve(strict=True)
    head = run(["git", "-C", str(path), "rev-parse", "HEAD"])
    if not expected or not re.fullmatch(r"[0-9a-f]{7,40}", expected) or not head.startswith(expected):
        raise Refusal("repository HEAD differs from the explicitly reviewed starting commit")
    if run(["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"]):
        raise Refusal("repository has concurrent local changes; staging must wait for coordination")
    return {"commit": head, "branch": run(["git", "-C", str(path), "branch", "--show-current"]),
            "working_tree_clean": True, "remote_refresh_required_before_later_commit": True}


def public_urls(value):
    found = set()
    if isinstance(value, dict):
        for child in value.values():
            found.update(public_urls(child))
    elif isinstance(value, list):
        for child in value:
            found.update(public_urls(child))
    elif isinstance(value, str) and value.startswith("https://"):
        parsed = urllib.parse.urlsplit(value)
        if parsed.hostname in {"github.com", "api.github.com", "raw.githubusercontent.com"} and not parsed.username and not parsed.password:
            found.add(urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")))
    return found


def stage(repository, expected):
    assert_quiescent()
    repo = repository_snapshot(repository, expected)
    selected, skipped = inventory()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage_root = ROOT / "publication/.stages" / stamp
    destination = stage_root / RELATIVE_DEST
    destination.mkdir(parents=True, exist_ok=False)
    records, snapshots = [], {}
    urls = set()
    for path, relative, binary in selected:
        raw = path.read_bytes()
        source_sha = sha(raw)
        snapshots[path] = source_sha
        if binary:
            public, redactions = raw, {}
        elif relative.startswith("provenance/"):
            # These are explicitly selected public upstream metadata, not local
            # machine records. Raw signatures and raw-file hashes require exact
            # bytes, including upstream CRLF endings and final-newline policy.
            public, redactions = raw, {}
            if path.suffix.lower() == ".json":
                urls.update(public_urls(json.loads(raw)))
        else:
            public, redactions = normalize(raw, path.suffix.lower())
            if path.suffix.lower() in {".json", ".jsonl"}:
                for line in public.decode().splitlines() if path.suffix.lower() == ".jsonl" else [public.decode()]:
                    urls.update(public_urls(json.loads(line)))
        out = destination / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("xb") as stream:
            stream.write(public)
        records.append({"source_relative_path": path.relative_to(ROOT).as_posix(), "published_relative_path": relative,
                        "source_bytes": len(raw), "source_sha256": source_sha,
                        "published_bytes": len(public), "published_sha256": sha(public),
                        "normalization": ("binary unchanged" if binary else "public upstream metadata unchanged" if relative.startswith("provenance/")
                                          else "UTF-8 text, LF; JSON values preserved except listed redactions"),
                        "redactions": redactions})
    assert_quiescent()
    if repository_snapshot(repository, expected) != repo:
        raise Refusal("repository changed during isolated staging")
    for path, original in snapshots.items():
        if sha(path.read_bytes()) != original:
            raise Refusal("a source artifact changed while staging: " + path.relative_to(ROOT).as_posix())
    if {p for p, _, _ in inventory()[0]} != set(snapshots):
        raise Refusal("the source artifact set changed while staging")
    manifest = {"schema_version": 1, "prepared_at": now(), "repository_baseline": repo,
                "destination_relative_path": RELATIVE_DEST.as_posix(), "source_snapshot": records,
                "excluded": skipped, "no_repository_files_changed": True, "not_committed_or_pushed": True,
                "upstream_artifacts": {"download_urls": sorted(urls), "vendor_binaries_and_archives_omitted": True,
                    "signature_evidence": "provenance/strata-feed.json(.sig), engine-component.json, public update keys and receipts/official-bundle-verification.json plus independent-openssl-verification.json",
                    "normalization_note": "Public upstream metadata is retained byte-for-byte, including CRLF and detached signatures. Source/public SHA256 identities verify those exact copies; local research receipts use documented text normalization/redactions."},
                "redaction_policy": "Operator homes, fleet private IPs/hostnames, UUIDs, maintenance owners and literal credential values are replaced in public text. Boolean api_key feature flags are retained. Raw local copies stay local.",
                "reproduction_note": "Public source paths are placeholders. Set model/assets/config paths for your own machine. This package contains benchmark code and synthetic evidence; vendor weights/runtime bundles are external pinned dependencies."}
    (destination / "PUBLICATION-MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checksum = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            checksum.append(sha(path.read_bytes()) + "  " + path.relative_to(destination).as_posix())
    (destination / "SHA256SUMS.txt").write_text("\n".join(checksum) + "\n")
    print(json.dumps({"status": "staged_local_only", "files": len(records), "destination": str(destination),
                      "no_commit_or_push": True, "next": "Review manifest/redactions, refresh remote state, then copy reviewed subtree and commit separately."}, indent=2))


def self_test():
    home = "/".join(["", "home", "example-operator", "work"])
    owner = "-".join(["codex", "lsa", "20261103", "example-owner"])
    uuid = "-".join(["12345678", "abcd", "1234", "abcd", "123456789abc"])
    ip = ".".join(["100", "95", "12", "34"])
    secret = "-".join(["synthetic", "credential", "example"])
    raw = json.dumps({"home": home, "owner": owner, "uuid": uuid, "ip": ip,
                      "api_key": True, "authorization": secret}).encode()
    public, counts = normalize(raw, ".json")
    value = json.loads(public)
    assert value["api_key"] is True and value["authorization"] == "<redacted-secret>"
    assert home not in public.decode() and owner not in public.decode() and uuid not in public.decode() and ip not in public.decode()
    assert all(counts.get(k) for k in ("operator_home", "maintenance_owner", "uuid", "private_fleet_ip", "credential_value"))
    text, _ = normalize(b"source\r\nline\r\n", ".txt")
    assert text == b"source\nline\n"
    own_public, _ = normalize(Path(__file__).read_bytes(), ".py")
    ast.parse(own_public.decode())
    print(json.dumps({"status": "self_test_passed", "receipts_read": False, "gpu_work_started": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--expected-repo-commit")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif not args.execute:
        selected, skipped = inventory()
        print(json.dumps({"status": "plan_only", "files": [{"source": p.relative_to(ROOT).as_posix(), "public": rel, "bytes": p.stat().st_size} for p, rel, _ in selected],
                          "excluded": skipped, "requires_finished_gpu_work_and_receipts": True,
                          "repository_writes": False, "commit_or_push": False}, indent=2))
    else:
        if args.repository is None or args.expected_repo_commit is None:
            raise Refusal("execution needs an explicitly reviewed repository path and starting commit")
        stage(args.repository, args.expected_repo_commit)


if __name__ == "__main__":
    try:
        main()
    except (Refusal, OSError, UnicodeError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "refused", "error": str(error) if isinstance(error, Refusal) else type(error).__name__}), file=sys.stderr)
        raise SystemExit(2)
