# Codex Workspace Rules

Before starting work, read:

- `.codex-memory/PROJECT_STATE.md`
- `.codex-memory/TODO.md`
- `.codex-memory/DECISIONS.md`
- `.codex-memory/LAST_SESSION.md`

Treat those files as the project's persistent memory.

Before editing files, run `git status`.

Automatic status saving should use local `.codex-memory/` files, not Git push.

After every meaningful change:

- Update `.codex-memory/LAST_SESSION.md`.
- Update `.codex-memory/TODO.md`, `.codex-memory/DECISIONS.md`, and `.codex-memory/PROJECT_STATE.md` if needed.
- Refresh the local memory snapshot or ensure the live memory watcher is running.
- Create local Git checkpoint commits only when intentionally useful or explicitly requested.
- Never push to Git automatically; pushing requires an explicit user request.

Never keep important progress only in chat.

Never store secrets, tokens, private keys, passwords, or credentials in memory files, logs, or commits.

## Proprietary YOLO Development Rules

- Treat `docs/PROPRIETARY_YOLO_DEVELOPMENT_PLAYBOOK.md` as the active workflow for YOLO architecture changes, benchmark review, and Rockchip/RKNN edge compatibility.
- Keep Git commits authored as `Akhilbeeram <AkhilBeeram25@users.noreply.github.com>`; verify `git config user.name` and `git config user.email` before committing or pushing.
- Do not add Codex-branded or AI-generated banners/comments to source files. Code should follow the surrounding project and Ultralytics-style conventions.
- For YOLO architecture work, make the smallest useful change, add focused tests, run the relevant smoke or benchmark checks, and record the measured result before promoting the change.
- Preserve Ultralytics API compatibility in `ULTRALYTICS_MICRO/` unless a breaking change is explicitly requested.
- When the user explicitly says to push, run the relevant checks, commit intended changes if needed under Akhilbeeram's identity, verify the latest author metadata, and then push. Never push without that explicit request.
