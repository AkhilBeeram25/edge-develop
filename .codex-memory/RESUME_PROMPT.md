Read AGENTS.md and all files under .codex-memory.

Resume from:
- .codex-memory/LAST_SESSION.md
- .codex-memory/TODO.md
- .codex-memory/DECISIONS.md
- .codex-memory/PROJECT_STATE.md

First run:
- git status
- git log --oneline -5

Continue from the next pending task.

After every meaningful step:
1. update .codex-memory/LAST_SESSION.md
2. update TODO.md, DECISIONS.md, and PROJECT_STATE.md if needed
3. refresh the local memory snapshot or confirm the live memory watcher is running
4. create local Git commits only when intentionally useful or explicitly requested

Never push to Git automatically. Pushing requires an explicit user request.

Never keep important context only in chat.
Never store secrets, tokens, passwords, private keys, or credentials.
