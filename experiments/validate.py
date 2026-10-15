"""Offline archive integrity and numeric reconciliation; no network or GPU use."""
from pathlib import Path
import hashlib
import json
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

def check(condition, message):
    if not condition:
        raise ValueError(message)

def receipt(name):
    return json.loads((ROOT / "receipts" / (name + ".json")).read_text(encoding="utf-8"))["data"]

def main():
    entries = {}
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        check(name not in entries, "Duplicate manifest entry: " + name)
        path = (ROOT / name).resolve()
        check(path.is_relative_to(ROOT), "Manifest escapes archive")
        check(path.is_file(), "Missing file: " + name)
        check(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "Hash mismatch: " + name)
        entries[name] = digest
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
              if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts}
    check(set(entries) == actual, "Manifest coverage differs from archive files")
    for path in [REPO / "README.md", *ROOT.glob("*.md")]:
        content = path.read_text(encoding="utf-8")
        for link in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", content):
            if "://" in link or link.startswith("#"):
                continue
            target = (path.parent / link.split("#")[0]).resolve()
            check(target.is_relative_to(REPO) and target.exists(), f"Broken local link in {path.name}: {link}")
    index = json.loads((ROOT / "receipts" / "index.json").read_text(encoding="utf-8"))
    for item in index["receipts"]:
        data = json.loads((ROOT / "receipts" / item["file"]).read_text(encoding="utf-8"))
        check(data["source"]["sha256"] == item["source_sha256"], "Source identity mismatch")
        check(re.fullmatch("[0-9a-f]{64}", data["source"]["sha256"]) is not None, "Invalid source hash")
    kv = receipt("kv-streaming")["conditions"]
    check(all(x["quality_passed"] and x["quality_cases"] == 32 and x["max_decoding"] == 8
              for x in kv.values()), "KV qualification mismatch")
    check(round((25395 - 14773) / 1024, 2) == 10.37, "VRAM arithmetic mismatch")
    four = receipt("four-slot")["conditions"]
    saving = four["eight_slots"]["loaded_service_pss_gib"] - four["four_slots"]["loaded_service_pss_gib"]
    check(round(saving, 3) == 6.148, "RAM saving mismatch")
    mtp = receipt("batched-mtp")
    check(mtp["production_promoted"] is False and mtp["full_model_qualified"] is False,
          "Unqualified candidate status changed")
    for pair in mtp["matched_comparisons"].values():
        check(pair["passed"] and len(pair["slots"]) == 8 and all(s["identical"] for s in pair["slots"]),
              "MTP parity mismatch")
        check(pair["candidate_tokens_per_second"] < pair["baseline_tokens_per_second"],
              "MTP speed conclusion changed")
    pair = mtp["matched_comparisons"]["greedy-draft-min-p-0"]
    check(round(100 * (1 - pair["candidate_tokens_per_second"] / pair["baseline_tokens_per_second"]), 2) == 4.83,
          "MTP slowdown arithmetic mismatch")
    accounting = receipt("lfm-accounting")
    for field in ("attempts", "submitted", "exact"):
        check(sum(arm[field] for arm in accounting["arms"]) == accounting[field], "LFM arm totals mismatch")
    check((accounting["attempts"], accounting["submitted"], accounting["exact"]) == (16, 1, 0),
          "LFM result mismatch")
    check((accounting["turns"], accounting["tool_calls"], accounting["generated_tokens"]) == (209, 308, 541195),
          "LFM activity totals mismatch")
    smoke = receipt("lfm-smoke")
    check(smoke["summary"] == {"passed": 14, "total": 14, "errors": 0, "truncated": 0},
          "Smoke summary mismatch")
    check(round(statistics.median(x["timings"]["predicted_per_second"] for x in smoke["speed"]), 1) == 320.1,
          "LFM speed median mismatch")
    check(all(x["finish_reason"] == "length" for x in smoke["speed"]), "Speed cap scope changed")
    visual = receipt("d1-visual")
    check(visual["attempts"] == visual["passes"] == len(visual["results"]) == 10, "d1 total mismatch")
    check(all(x["passed"] for x in visual["results"]), "d1 case mismatch")
    check(round(statistics.median(x["inference_ms"] for x in visual["results"]), 2) == 27.69,
          "d1 inference median mismatch")
    check(round(statistics.median(x["elapsed_seconds"] for x in visual["results"]), 3) == 1.548,
          "d1 wall-time median mismatch")
    tf = receipt("tensorfold-four-stream")
    check(round(statistics.median(tf["selected_aggregate_tps"]), 2) == 383.08, "TensorFold median mismatch")
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in (".md", ".json"):
            continue
        content = path.read_text(encoding="utf-8")
        patterns = (r"(?i)[A-Z]:[\\/](?:Users|BlueStone)", r"/home/[^ /]+",
                    r"(?i)tail[0-9a-f]+\.ts\.net", r"\b100\.(?:\d{1,3}\.){2}\d{1,3}\b",
                    r"-----BEGIN .*PRIVATE KEY-----", r"\bgh[pousr]_[A-Za-z0-9]{20,}",
                    r"\bsk-[A-Za-z0-9]{20,}")
        check(not any(re.search(pattern, content) for pattern in patterns), "Private data pattern in " + path.name)
    print(f"PASS: {len(entries)} archive hashes, local links, {len(index['receipts'])} source bindings, numeric checks and privacy patterns")

if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, AssertionError) as exc:
        print("FAIL:", exc, file=sys.stderr)
        sys.exit(1)
