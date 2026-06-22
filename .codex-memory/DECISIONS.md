# Decisions

## Durable Checkpoints

- Use Git commits as durable checkpoints so work can be resumed after unexpected shutdown.

## Persistent Memory

- Use `.codex-memory/` as project memory for Codex state, tasks, decisions, and last-session recovery notes.

## Secrets

- Do not store secrets, tokens, private keys, passwords, or credentials in memory files, logs, or commits.

