# Development instructions

This is a game-agent laboratory, not a solved Zelda bot. Read README.md and docs/STATUS.md before changing scope.

- Python 3.12+, uv, FastAPI/SQLAlchemy; React/Vite/TypeScript; native C++20.
- Run `uv run pytest -q` and `cd web && npm run build`. The C++ lease test requires g++ or clang++ and skips when absent. Never describe a skipped build as successful.
- Preserve the explicit simulator label. Never silently fall back from Codex/OpenRouter/SoH to demo responses, synthetic metrics or fabricated gameplay.
- Unknown usage/cost is null. Cached/reasoning tokens are subsets. Record failures and paid output truncation; never retry paid requests without the operator choosing to do so.
- Keep state and prompts bounded; video capture is local to the browser. No screenshots to models by default.
- Keep Codex auth inside the official CLI profile. Do not read/exfiltrate OAuth tokens, commit credentials, or convert subscription tokens to API keys. Test real providers only with explicit operator authorization and a configured budget.
- A gameplay model returns a typed decision; it does not get developer shell access. This does not restrict the development agent from editing code, building the project, or using worktrees normally.
- Native input must expire, release on stop, reject stale/cross-scene packets, and remain limited to controller input. No teleports, HP writes, or hidden solution flags in observations.
- Preserve SoH as a separate checkout pinned by the integration script. Do not hard-reset user source/assets or distribute ROMs, saves or copyrighted game data.
- Capture actual validation in docs/STATUS.md. No claim of autonomous completion, learned aiming or reliable navigation before reproducing it in the game.
- UI direction: functional laboratory cockpit, olive/graphite, compact telemetry, readable tables, operational Portuguese copy. No decorative fake graphs, scores, gradients or fantasy screenshots.
- No GitHub Actions required. Prefer local checks; generate dependency lockfiles after an actual successful resolve and do not invent them.
