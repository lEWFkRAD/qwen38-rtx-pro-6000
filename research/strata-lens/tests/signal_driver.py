#!/usr/bin/env python3
"""Subprocess-only real-signal regression driver for the Phase-0 harness."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import phase0_harness as harness


FAKE_ENGINE = Path(__file__).resolve().with_name("fake_engine.py")


def main() -> int:
    output_root = Path(sys.argv[1]).resolve()
    run_id = sys.argv[2]
    cleanup_marker = output_root / "cleanup-entered"
    result_path = output_root / "signal-driver-result.json"
    prior = {signum: signal.getsignal(signum) for signum in harness.HANDLED_LAUNCH_SIGNALS}

    original_cleanup = harness._terminate_process_group

    def marked_cleanup(process, pgid, grace_seconds):
        cleanup_marker.write_text(f"{process.pid} {pgid}\n", encoding="ascii")
        return original_cleanup(process, pgid, grace_seconds)

    harness._terminate_process_group = marked_cleanup
    request = harness.RunRequest(
        real_token_ids=(11, 12, 13),
        output_root=output_root,
        timeout_seconds=30.0,
        kill_grace_seconds=0.5,
        command_prefix=(sys.executable, str(FAKE_ENGINE)),
        engine_identity={"test_override": True, "mode": "ignore_term_child"},
        extra_child_env={
            "FAKE_ENGINE_MODE": "ignore_term_child",
            "FAKE_CHILD_PID_FILE": str(
                output_root / f"{run_id}.partial" / "fake-child.pid"
            ),
        },
        test_only_allow_unpinned=True,
        run_id=run_id,
    )
    try:
        harness.run_once(request)
    except harness.ExecutionSignal as exc:
        restored = all(
            signal.getsignal(signum) == previous for signum, previous in prior.items()
        )
        result_path.write_text(
            json.dumps(
                {
                    "signal": exc.signum,
                    "signal_name": exc.signal_name,
                    "handlers_restored": restored,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 128 + exc.signum
    except BaseException as exc:
        result_path.write_text(
            json.dumps({"unexpected": type(exc).__name__, "message": str(exc)}) + "\n",
            encoding="utf-8",
        )
        return 120
    result_path.write_text('{"unexpected": "success"}\n', encoding="utf-8")
    return 121


if __name__ == "__main__":
    raise SystemExit(main())
