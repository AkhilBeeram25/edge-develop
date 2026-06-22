# Codex Workspace Rules

Before starting work, read:

- `.codex-memory/PROJECT_STATE.md`
- `.codex-memory/TODO.md`
- `.codex-memory/DECISIONS.md`
- `.codex-memory/LAST_SESSION.md`

Treat those files as the project's persistent memory.

Before editing files, run `git status`.

After every meaningful change:

- Update `.codex-memory/LAST_SESSION.md`.
- Update `.codex-memory/TODO.md`, `.codex-memory/DECISIONS.md`, and `.codex-memory/PROJECT_STATE.md` if needed.
- Create a small Git checkpoint commit.

Never keep important progress only in chat.

Never store secrets, tokens, private keys, passwords, or credentials in memory files, logs, or commits.

