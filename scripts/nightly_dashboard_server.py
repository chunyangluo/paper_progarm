#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local dashboard control server for nightly network collection.

Provides:
- static file serving for dashboard page
- start/stop/status control APIs for collector process
"""

import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUN_NAME = "dataset_20260421_100k_nightly"
DEFAULT_INPUT = "data/versions/dataset_20260421_100k.csv"
DEFAULT_CHUNK_SIZE = 20000
DEFAULT_CHUNKS_PER_RUN = 2
DEFAULT_BATCH_SIZE = 100
STATE_DIR = os.path.join(PROJECT_ROOT, "data", "nightly_network_runs", DEFAULT_RUN_NAME)
PID_FILE = os.path.join(STATE_DIR, "collector_process.json")

_collector_process = None
_process_lock = threading.Lock()


def _ensure_state_dir():
    os.makedirs(STATE_DIR, exist_ok=True)


def _read_pid_meta():
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_pid_meta(meta):
    _ensure_state_dir()
    with open(PID_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _clear_pid_meta():
    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
        except Exception:
            pass


def _build_collect_cmd():
    return [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "nightly_network_feature_pipeline.py"),
        "--input",
        DEFAULT_INPUT,
        "--run-name",
        DEFAULT_RUN_NAME,
        "--chunk-size",
        str(DEFAULT_CHUNK_SIZE),
        "--chunks-per-run",
        str(DEFAULT_CHUNKS_PER_RUN),
        "--batch-size",
        str(DEFAULT_BATCH_SIZE),
        "--no-auto-merge",
    ]


def _is_pid_running(pid):
    if not pid:
        return False
    try:
        # Windows-friendly check with tasklist CSV output.
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH", "/FI", f"PID eq {int(pid)}"],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (result.stdout or "").strip()
        if not out or out.lower().startswith("info:"):
            return False
        return f"\"{int(pid)}\"" in out
    except Exception:
        return False


def _start_collector():
    global _collector_process
    with _process_lock:
        if _collector_process and _collector_process.poll() is None:
            return {"ok": True, "message": "collector already running", "pid": _collector_process.pid}
        if _collector_process and _collector_process.poll() is not None:
            _collector_process = None

        meta = _read_pid_meta()
        if meta and _is_pid_running(meta.get("pid")):
            return {"ok": True, "message": "collector already running (pid file)", "pid": meta.get("pid")}
        if meta and not _is_pid_running(meta.get("pid")):
            _clear_pid_meta()

        cmd = _build_collect_cmd()
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _collector_process = proc
        _write_pid_meta({"pid": proc.pid, "cmd": cmd, "started_at": datetime.now().isoformat()})
        return {"ok": True, "message": "collector started", "pid": proc.pid}


def _stop_collector():
    global _collector_process
    with _process_lock:
        pid = None
        if _collector_process and _collector_process.poll() is None:
            pid = _collector_process.pid
        else:
            meta = _read_pid_meta()
            if meta:
                pid = meta.get("pid")

        if not pid:
            _clear_pid_meta()
            return {"ok": True, "message": "collector not running"}

        # Kill process tree to avoid orphan children.
        subprocess.run(
            ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        _collector_process = None
        _clear_pid_meta()
        return {"ok": True, "message": "collector stopped", "pid": pid}


def _collector_status():
    pid = None
    running = False
    if _collector_process and _collector_process.poll() is None:
        pid = _collector_process.pid
        running = True
    else:
        meta = _read_pid_meta()
        if meta:
            pid = meta.get("pid")
            running = _is_pid_running(pid)
            if not running:
                _clear_pid_meta()
                pid = None
    return {"running": running, "pid": pid, "run_name": DEFAULT_RUN_NAME}


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=PROJECT_ROOT, **kwargs)

    def _send_json(self, payload, code=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/collector/status":
            return self._send_json(_collector_status())
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/collector/start":
            return self._send_json(_start_collector())
        if parsed.path == "/api/collector/stop":
            return self._send_json(_stop_collector())
        return self._send_json({"ok": False, "error": "unknown endpoint"}, 404)


def main():
    port = 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
    print(f"Dashboard control server started at http://127.0.0.1:{port}")
    print("Use /api/collector/start and /api/collector/stop to control collection.")
    server.serve_forever()


if __name__ == "__main__":
    main()
