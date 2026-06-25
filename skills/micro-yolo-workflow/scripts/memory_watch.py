#!/usr/bin/env python3
"""Maintain a one-second local memory heartbeat for the workspace."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


MEMORY_FILES = (
    ".codex-memory/PROJECT_STATE.md",
    ".codex-memory/TODO.md",
    ".codex-memory/DECISIONS.md",
    ".codex-memory/LAST_SESSION.md",
    ".codex-memory/RESUME_PROMPT.md",
)


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_memory_files(workspace: Path) -> dict[str, dict[str, str]]:
    state: dict[str, dict[str, str]] = {}
    for rel_path in MEMORY_FILES:
        path = workspace / rel_path
        if not path.exists():
            state[rel_path] = {"exists": "false", "sha256": "", "bytes": "0"}
            continue
        text = path.read_text(encoding="utf-8")
        state[rel_path] = {
            "exists": "true",
            "sha256": sha256_text(text),
            "bytes": str(len(text.encode("utf-8"))),
        }
    return state


def run_git_status(workspace: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # noqa: BLE001 - live status must be best-effort
        return f"git status unavailable: {exc}"
    output = (result.stdout + result.stderr).strip()
    return output or f"git status exited {result.returncode} with no output"


def build_status(
    *,
    workspace: Path,
    interval: float,
    git_status: str,
    memory_state: dict[str, dict[str, str]],
    changed_at: str,
) -> str:
    lines = [
        "# Live Codex Memory",
        "",
        f"- Updated UTC: `{utc_timestamp()}`",
        f"- Workspace: `{workspace}`",
        f"- Heartbeat interval seconds: `{interval:g}`",
        f"- Last memory-content change UTC: `{changed_at}`",
        "- Persistence mode: local `.codex-memory/live/` files; no Git commit or push.",
        "",
        "## Best-Effort Git Status",
        "",
        "```text",
        git_status,
        "```",
        "",
        "## Memory Files",
        "",
    ]
    for rel_path, meta in memory_state.items():
        exists = meta["exists"]
        digest = meta["sha256"][:12] if meta["sha256"] else "missing"
        byte_count = meta["bytes"]
        lines.append(f"- `{rel_path}` exists={exists} bytes={byte_count} sha256={digest}")
    return "\n".join(lines).rstrip() + "\n"


def build_memory_bundle(workspace: Path, changed_at: str) -> str:
    sections = [
        "# Latest Live Memory Bundle",
        "",
        f"- Updated UTC: `{utc_timestamp()}`",
        f"- Last memory-content change UTC: `{changed_at}`",
        f"- Workspace: `{workspace}`",
        "- Source: `memory_watch.py` local-only watcher",
    ]
    for rel_path in MEMORY_FILES:
        path = workspace / rel_path
        sections.extend(["", f"## {rel_path}", ""])
        if path.exists():
            sections.append(path.read_text(encoding="utf-8"))
        else:
            sections.append(f"Missing: `{rel_path}`")
    return "\n".join(sections).rstrip() + "\n"


def write_pid_file(path: Path) -> None:
    atomic_write(path, f"{os.getpid()}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=".", help="Project workspace containing .codex-memory")
    parser.add_argument("--interval", type=float, default=1.0, help="Heartbeat interval in seconds")
    parser.add_argument("--git-interval", type=float, default=10.0, help="Minimum seconds between git status checks")
    parser.add_argument("--live-dir", default=".codex-memory/live", help="Workspace-relative live memory directory")
    parser.add_argument("--pid-file", default="memory_watch.pid", help="Live-dir-relative PID file name")
    parser.add_argument("--once", action="store_true", help="Write one heartbeat and exit")
    args = parser.parse_args()

    if args.interval <= 0:
        raise SystemExit("--interval must be greater than 0")
    if args.git_interval <= 0:
        raise SystemExit("--git-interval must be greater than 0")

    workspace = Path(args.workspace).expanduser().resolve()
    memory_dir = workspace / ".codex-memory"
    if not memory_dir.is_dir():
        raise SystemExit(f"Missing memory directory: {memory_dir}")

    live_dir = workspace / args.live_dir
    live_dir.mkdir(parents=True, exist_ok=True)
    write_pid_file(live_dir / args.pid_file)

    heartbeat_path = live_dir / "HEARTBEAT.json"
    status_path = live_dir / "STATUS.md"
    bundle_path = live_dir / "latest_memory.md"

    previous_memory_state: dict[str, dict[str, str]] | None = None
    changed_at = utc_timestamp()
    git_status = "not checked yet"
    last_git_check = 0.0

    while True:
        started = time.monotonic()
        now = time.time()
        if now - last_git_check >= args.git_interval:
            git_status = run_git_status(workspace)
            last_git_check = now

        memory_state = read_memory_files(workspace)
        if memory_state != previous_memory_state:
            changed_at = utc_timestamp()
            previous_memory_state = memory_state
            atomic_write(bundle_path, build_memory_bundle(workspace, changed_at))

        heartbeat = {
            "updated_utc": utc_timestamp(),
            "workspace": str(workspace),
            "pid": os.getpid(),
            "interval_seconds": args.interval,
            "last_memory_content_change_utc": changed_at,
            "git_status": git_status,
            "memory_files": memory_state,
            "mode": "local-only; no git commit or push",
        }
        atomic_write(heartbeat_path, json.dumps(heartbeat, indent=2, sort_keys=True) + "\n")
        atomic_write(
            status_path,
            build_status(
                workspace=workspace,
                interval=args.interval,
                git_status=git_status,
                memory_state=memory_state,
                changed_at=changed_at,
            ),
        )

        if args.once:
            return 0

        elapsed = time.monotonic() - started
        time.sleep(max(0.0, args.interval - elapsed))


if __name__ == "__main__":
    raise SystemExit(main())
