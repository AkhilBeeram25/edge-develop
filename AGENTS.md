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
