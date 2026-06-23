#!/usr/bin/env python3
"""Create a non-Git snapshot of the project memory files."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path


MEMORY_FILES = (
    ".codex-memory/PROJECT_STATE.md",
    ".codex-memory/TODO.md",
    ".codex-memory/DECISIONS.md",
    ".codex-memory/LAST_SESSION.md",
)


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
    except Exception as exc:  # noqa: BLE001 - best-effort diagnostic only
        return f"git status unavailable: {exc}"
    output = (result.stdout + result.stderr).strip()
    return output or f"git status exited {result.returncode} with no output"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=".", help="Project workspace containing .codex-memory")
    parser.add_argument("--note", default="", help="Short status note to include in the snapshot")
    parser.add_argument("--no-git", action="store_true", help="Skip best-effort git status capture")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    memory_dir = workspace / ".codex-memory"
    if not memory_dir.is_dir():
        raise SystemExit(f"Missing memory directory: {memory_dir}")

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = memory_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{timestamp}.md"

    sections: list[str] = [
        "# Memory Snapshot",
        "",
        f"- Timestamp: `{timestamp}`",
        f"- Workspace: `{workspace}`",
    ]
    if args.note:
        sections.append(f"- Note: {args.note}")
    if not args.no_git:
        sections.extend(["", "## Best-Effort Git Status", "", "```text", run_git_status(workspace), "```"])

    for rel_path in MEMORY_FILES:
        path = workspace / rel_path
        sections.extend(["", f"## {rel_path}", ""])
        if path.exists():
            sections.append(path.read_text(encoding="utf-8"))
        else:
            sections.append(f"Missing: `{rel_path}`")

    snapshot_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    latest_path = memory_dir / "LATEST_SNAPSHOT.md"
    latest_path.write_text(snapshot_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(snapshot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
