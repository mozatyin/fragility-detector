# Project Rules

## Sensitive Information Protection

1. **Pre-commit/push scan required**: Before every `git commit` or `git push`, automatically scan ALL staged files for hardcoded API keys, passwords, tokens, secrets, or other sensitive information (patterns: `sk-or-`, `sk-`, `ghp_`, `password=`, `secret=`, `token=`, `Bearer `, etc.).

2. **Block on detection**: If any sensitive information is found, STOP immediately. Do NOT commit or push. Report the exact file and line number, then refactor the code to read from environment variables instead, storing actual values in `.env`.

3. **Protect .env**: Ensure `.env` is listed in `.gitignore` and will never be committed to the repository.

4. **Commit only when clean**: Only execute `git commit` or `git push` after confirming all files are free of plaintext secrets.

5. **Proactive interception**: Even if the user says "push" without asking for a check, proactively scan first. Always check before any git operation that sends code to a remote or creates a commit.

## Architecture

Fragility Detector detects how a user expresses or hides vulnerability. Forked from emotion-detector.

4 fragility patterns:
- **Open** (开放型): directly expresses vulnerability
- **Defensive** (防御型): deflects vulnerability
- **Masked** (伪装型): hides vulnerability behind humor/casualness
- **Denial** (否认型): denies vulnerability exists

Core pipeline: text -> behavioral features + LLM anchors -> pattern classification -> star label

## Testing
- Run tests: `.venv/bin/pytest`
- All code changes must have tests
