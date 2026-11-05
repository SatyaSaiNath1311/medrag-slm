"""Kaggle-side launcher: boots the chat UI with the live backend and
exposes it through a cloudflared quick tunnel.

Usage (from repo root on Kaggle):
    python3 kaggle_chat/launch_ui.py

Requires ``cloudflared`` on PATH.  The UI subprocess logs to
``outputs/deployment/chat_ui_<YYYY-MM-DD>.log``.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SEP = "=" * 70
PORT = 7860
POLL_TIMEOUT = 900          # seconds — model load is slow on T4
POLL_INTERVAL = 2           # seconds between TCP probes
LOG_INTERVAL = 30           # seconds between progress prints
TAIL_LINES = 40             # lines to dump if the UI dies early


def _repo_root() -> Path:
    """Return the repository root (parent of kaggle_chat/)."""
    return Path(__file__).resolve().parent.parent


def _wait_for_port(proc: subprocess.Popen, log_path: Path) -> None:
    """Block until 127.0.0.1:PORT accepts a TCP connection."""
    start = time.monotonic()
    last_print = start
    while time.monotonic() - start < POLL_TIMEOUT:
        # If the child already died, bail out with a log dump.
        if proc.poll() is not None:
            print(f"\n[launch_ui] UI process exited with code {proc.returncode}")
            print(f"[launch_ui] last {TAIL_LINES} lines of {log_path}:\n")
            try:
                lines = log_path.read_text().splitlines()
                for line in lines[-TAIL_LINES:]:
                    print(line)
            except FileNotFoundError:
                print("  (log file not found)")
            sys.exit(1)

        # Try a TCP connect.
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                elapsed = time.monotonic() - start
                print(f"[launch_ui] port {PORT} ready after {elapsed:.0f}s")
                return
        except OSError:
            pass

        now = time.monotonic()
        if now - last_print >= LOG_INTERVAL:
            print(f"[launch_ui] waiting for port {PORT}… {now - start:.0f}s elapsed")
            last_print = now
        time.sleep(POLL_INTERVAL)

    print(f"[launch_ui] TIMEOUT: port {PORT} not open after {POLL_TIMEOUT}s")
    sys.exit(1)


def _parse_tunnel_url(log_path: Path, proc: subprocess.Popen) -> str:
    """Tail the cloudflared log file until the trycloudflare URL appears."""
    pattern = re.compile(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)")
    deadline = time.monotonic() + 120
    with open(log_path, "r") as fh:
        while time.monotonic() < deadline:
            line = fh.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
                continue
            text = line.strip()
            if text:
                print(f"  [cloudflared] {text}")
            m = pattern.search(text)
            if m:
                return m.group(1)
    # Timeout or process died — dump the tail of the log for diagnostics.
    print("[launch_ui] ERROR: could not parse tunnel URL from cloudflared output")
    try:
        lines = log_path.read_text().splitlines()
        for line in lines[-20:]:
            print(line)
    except FileNotFoundError:
        print("  (cloudflared log not found)")
    sys.exit(3)


def main() -> None:
    print(SEP)
    print("MedRAG-SLM  CHAT UI LAUNCHER")
    print(SEP)

    # ---- pre-flight: cloudflared must be on PATH ----
    if shutil.which("cloudflared") is None:
        print("[launch_ui] ERROR: cloudflared is not installed.")
        print("  Install it first:")
        print("    curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared")
        sys.exit(2)

    repo = _repo_root()
    log_dir = repo / "outputs" / "deployment"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"chat_ui_{datetime.now():%Y-%m-%d}.log"

    # ---- start the UI subprocess ----
    env = {**os.environ, "MEDRAG_BACKEND": "live", "GRADIO_SERVER_PORT": str(PORT)}
    print(f"[launch_ui] log  → {log_path}")
    print(f"[launch_ui] port → {PORT}")
    print(f"[launch_ui] starting UI (MEDRAG_BACKEND=live) …")

    log_fh = open(log_path, "a")
    ui_proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "kaggle_chat.chat_ui"],
        cwd=str(repo),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

    # ---- wait for Gradio to bind the port ----
    _wait_for_port(ui_proc, log_path)

    # ---- start cloudflared tunnel ----
    date_stamp = log_path.stem.replace("chat_ui_", "")      # YYYY-MM-DD
    cf_log_path = log_dir / f"cloudflared_{date_stamp}.log"
    cf_log_fh = open(cf_log_path, "ab")
    print(f"[launch_ui] starting cloudflared tunnel …")
    print(f"[launch_ui] cloudflared log → {cf_log_path}")
    cf_proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
        stdout=cf_log_fh,
        stderr=subprocess.STDOUT,
    )
    url = _parse_tunnel_url(cf_log_path, cf_proc)
    print(SEP)
    print(f"PUBLIC URL: {url}")
    print(SEP)

    # ---- block until Ctrl-C ----
    try:
        signal.pause()
    except (KeyboardInterrupt, AttributeError):
        # AttributeError: signal.pause() not available on Windows
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    # ---- clean shutdown ----
    print("\n[launch_ui] shutting down …")
    for label, proc in [("cloudflared", cf_proc), ("UI", ui_proc)]:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
                print(f"  {label}: terminated")
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                print(f"  {label}: killed")
    log_fh.close()
    cf_log_fh.close()
    print("[launch_ui] done")


if __name__ == "__main__":
    main()

