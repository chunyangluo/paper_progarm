"""
Poll a running `run_experiment.py` session and write a small status JSON for near-real-time UI/monitoring.

Writes: <session_dir>/session_status.json  (overwritten on each tick)

It combines:
- checkpoint.json (completed pairs)
- Windows process list (find active worker: --worker-model / --worker-seed)

Usage:
  python scripts/experiment/watch_session_status.py --session-dir output/experiments/MY_RUN --interval 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_run_experiment_processes() -> str:
    # WMIC is available on all supported Windows versions used here; avoids extra deps.
    try:
        out = subprocess.check_output(
            [
                "wmic",
                "process",
                "where",
                "name='python.exe'",
                "get",
                "CommandLine,ProcessId",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
    except Exception as exc:
        return f"wmic_error: {exc}"
    return out


def _parse_worker_from_wmic(wmic_text: str) -> dict | None:
    """
    Return {pid, model, seed} for a worker command line, else None.
    """
    for line in wmic_text.splitlines():
        line = line.strip()
        if "run_experiment.py" not in line or "--worker-model" not in line:
            continue
        m_model = re.search(r"--worker-model\s+([^\s]+)", line)
        m_seed = re.search(r"--worker-seed\s+(\d+)", line)
        m_pid = re.search(r"(\d+)\s*$", line)  # last number on line in many WMIC outputs
        if not m_model or not m_seed:
            continue
        pid = int(m_pid.group(1)) if m_pid else None
        return {"pid": pid, "model": m_model.group(1), "seed": int(m_seed.group(1))}
    return None


def _planned_pairs_from_models(models: list[str]) -> list[str]:
    seeds = [42, 123, 456]
    return [f"{m}__seed_{s}" for m in models for s in seeds]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True, help="Experiment session directory")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between updates")
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Optional comma-separated model list (if omitted, inferred from checkpoint keys + common plan)",
    )
    args = parser.parse_args()

    session_dir = os.path.abspath(os.path.join(PROJECT_ROOT, args.session_dir))
    ckpt_path = os.path.join(session_dir, "checkpoint.json")
    out_path = os.path.join(session_dir, "session_status.json")

    models_filter = [m.strip() for m in args.models.split(",") if m.strip()]

    while True:
        now = datetime.now().isoformat(timespec="seconds")
        status: dict = {
            "updated_at": now,
            "session_dir": session_dir,
            "checkpoint_path": ckpt_path,
            "orchestrator": {"running": False, "pid": None, "command": None},
            "worker": None,
            "progress": {
                "completed_pairs": [],
                "failed_pairs": {},
                "planned_pairs": [],
                "remaining_pairs": [],
                "completed": 0,
                "total": 0,
                "percent": 0.0,
            },
        }

        wmic_text = _list_run_experiment_processes()
        status["worker"] = _parse_worker_from_wmic(wmic_text) or {
            "state": "no_worker_process_detected",
        }

        # Find orchestrator process (non-worker)
        for line in wmic_text.splitlines():
            line = line.strip()
            if "run_experiment.py" not in line or "--worker-model" in line:
                continue
            m_pid = re.search(r"(\d+)\s*$", line)
            if m_pid:
                status["orchestrator"]["running"] = True
                status["orchestrator"]["pid"] = int(m_pid.group(1))
                status["orchestrator"]["command"] = line
                break

        if not os.path.exists(ckpt_path):
            status["error"] = "checkpoint.json not found yet"
        else:
            try:
                ckpt = _read_json(ckpt_path)
            except Exception as exc:
                status["error"] = f"failed to read checkpoint: {exc}"
            else:
                completed = sorted(ckpt.get("completed_pairs", []))
                failed = ckpt.get("failed_pairs", {}) or {}

                # Infer model universe for progress: prefer explicit --models, else all keys in all_results
                if models_filter:
                    models = models_filter
                else:
                    models = list(ckpt.get("all_results", {}).keys())

                # If the session was started with a subset, the checkpoint may not list future models;
                # allow user to pass --models. If still empty, keep minimal info.
                planned = _planned_pairs_from_models(models) if models else []
                completed_set = set(completed)

                remaining: list[str] = []
                in_progress = None
                w = status.get("worker")
                if isinstance(w, dict) and w.get("model") and w.get("seed") is not None:
                    in_progress = f"{w['model']}__seed_{w['seed']}"
                status["progress"]["in_progress_pair"] = in_progress

                if planned:
                    done_in_plan = [p for p in planned if p in completed_set]
                    not_done = [p for p in planned if p not in completed_set]
                    pending = list(not_done)
                    if in_progress and in_progress in not_done:
                        pending = [p for p in not_done if p != in_progress]

                    status["progress"]["completed_pairs_in_plan"] = done_in_plan
                    status["progress"]["not_started_pairs"] = pending
                    status["progress"]["remaining_pairs"] = not_done
                    # Prefer pending for a cleaner “what’s left after current”
                    status["progress"]["pending_pairs"] = pending

                    status["progress"]["completed"] = len(done_in_plan)
                    status["progress"]["total"] = len(planned)
                else:
                    status["progress"]["completed"] = len(completed)
                    # Unknown planned total: avoid misleading percent
                    status["progress"]["total"] = max(1, len(completed))

                status["progress"]["completed_pairs"] = completed
                status["progress"]["failed_pairs"] = failed
                status["progress"]["planned_pairs"] = planned
                denom = status["progress"]["total"] or 1
                status["progress"]["percent"] = round(100.0 * float(status["progress"]["completed"]) / float(denom), 2)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2, default=str)

        time.sleep(max(0.5, float(args.interval)))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        raise SystemExit(0)
