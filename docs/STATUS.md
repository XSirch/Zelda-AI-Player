# Implementation status — Autonomy V3

## Implemented

- Minimal realtime panel: connection, operational thought, raw N64 inputs, tokens/cost/quota, start/stop.
- Continuous motor loop independent from LLM latency.
- Raw-action PPO actor-critic: analog stick + physical button bits.
- Goal-conditioned camera-relative steering prior inside the PPO stick distribution, with learned residual control and a mild navigation button-quiet prior.
- Collision-aware local goal guidance: observed probes bend a blocked direct heading toward a walkable side direction, hard blockage weakens the prior, prolonged dwell fades stale-target attraction, and cognition can replan on `guidance_blocked`.
- One-shot `intent_target_reached` replanning so completed waypoints do not remain stale.
- Four-frame structured-state stack for temporal behaviour and combat timing.
- Online PPO learner with GAE and separate actor/learner weights.
- RND intrinsic curiosity.
- Observable reward shaping without a scripted Zelda quest path, including escalating coarse-area dwell penalties, frontier exploration, stale-waypoint `intent_progress` fadeout, and delta-based resource rewards (rupees/ammo/health/magic) that naturally suppress full-capacity pickups.
- Native one-shot chest-open telemetry from `FLAG_SCENE_TREASURE`, surfaced as `chest_opened`.
- Persistent atomic PyTorch training checkpoint plus immutable completion champions under `.local/ml/champions/`.
- High-level `AgentIntent` cognition contract with no skill/action selector.
- Codex and OpenRouter intent providers.
- Codex token usage accounting and independent quota polling.
- Persisted run benchmark summary with frozen terminal elapsed time, input/output token totals, known/unknown cost semantics, per-provider/model usage breakdown, and the same benchmark fields embedded in completion champion metadata.
- Bridge authority/watchdog remains the hard safety boundary for controller ownership.
- Existing native scene autosave and structured observation telemetry remain available.
- Automatic champion capture after training `game_completed`, plus frozen deterministic evaluation mode.

## Removed

- monolithic skill-planner runtime;
- semantic skill catalog/executor;
- scripted opening checkpoint planner;
- heuristic per-enemy combat policy;
- Python NavMesh/A* action executor;
- manual input diagnostics from the primary application;
- dashboard tabs for Skills/Navigation/Combat/Memory/Benchmarks.

## Validation gates

Before calling this autonomous completion-capable, reproduce on real SoH:

1. click **INICIAR** and confirm raw inputs continue while a deliberately slow cognition call is pending;
2. confirm ML checkpoint update count grows during play and survives restart;
3. confirm the policy discovers useful button effects from reward rather than hard-coded mappings;
4. verify structured Luna waypoints produce active motor guidance, distance falls, `intent_target_reached` advances the waypoint, and navigation improves without human commands;
5. verify combat behaviour improves across repeated enemy encounters;
6. verify **PARAR** produces immediate neutral input even during provider inference or PPO training;
7. verify token totals and Codex remaining quota update independently of motor control;
8. after a real completion, verify a champion is created only after final PPO work settles;
9. evaluate that champion from a new save and confirm `run_updates == 0`, the champion SHA stays unchanged, and repeat performance is measurable.

## Current limitation

A fresh PPO policy begins largely random. Autonomy V3 provides the learning architecture; it is not yet evidence that an untrained policy can complete OoT. The next milestone is real-game training and reward/feature tuning based on observed learning curves rather than adding semantic controller macros.
