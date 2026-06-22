# Project State

## Workspace

- Path: `/home/open/ak`
- Purpose: persistent Codex memory and resumable development on an ARM64 SoC device that may shut down unexpectedly.

## Important Commands

- Check workspace state: `git status`
- View recent checkpoints: `git log --oneline -5`
- Read last session: `cat .codex-memory/LAST_SESSION.md`
- Manual checkpoint: `savecodex`
- Autosave checkpoint script: `/home/open/.local/bin/codex-checkpoint /home/open/ak`

## Current Setup Status

- Git repository initialized for durable checkpoints.
- `.codex-memory/` created for persistent project memory.
- Autosave checkpoint script installed at `/home/open/.local/bin/codex-checkpoint`.
- Cron autosave configured to run once per minute for `/home/open/ak`.
- Manual shell helper `savecodex` configured in `/home/open/.bashrc` when that file exists.
- First durable checkpoint commit is ready to create.
