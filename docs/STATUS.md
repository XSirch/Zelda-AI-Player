# Implementation status — Autonomy V3

## Implemented

- Minimal realtime panel: connection, operational thought, raw N64 inputs, tokens/cost/quota, start/stop.
- Continuous motor loop independent from LLM latency.
- Raw-action PPO actor-critic: residual analog stick + physical button bits.
- Goal-conditioned camera-relative guidance is now mixed deterministically with the sampled residual stick, so structured navigation authority cannot be numerically overwhelmed by a mature Beta distribution.
- Indoor/fixed-camera steering now preserves a world-space heading and reprojects it at the 20 Hz motor cadence from SoH's current camera input yaw. Abrupt camera cuts briefly neutralize stale navigation stick, then resume the same waypoint; camera changes do not invalidate the learned route, and dialogue/interaction overrides keep authority. A view eye/lookAt fallback exists for payloads without input yaw. This has regression coverage in code but still requires live SoH validation across real indoor camera-mode switches before claiming reliable indoor navigation.
- Stick/button entropy are trained separately; button entropy anneals to zero over the first 50k trained samples and expected button count is explicitly regularized to stop indefinite button-mashing.
- Collision-aware local goal guidance: observed probes bend a blocked direct heading toward a walkable side direction, hard blockage weakens the prior, prolonged dwell fades stale-target attraction, and cognition can replan on `guidance_blocked`.
- Persistent route memory v1: directed topological edges are learned only from paths Link actually traverses and replay through the real observed edge-entry gateway rather than coarse cell centroids. Active learned edges are health-checked during replay; stalled/expired edges accumulate failure cost, enter temporary cooldown and trigger alternate route/exit/frontier recovery. Frontier failures are now persistent negative memory too: repeatedly bad projected cells lose priority across runs, successful visits heal them, and high room-local failure pressure can force an observed exit under a trackable objective.
- Learned interaction affordances: doors/context actions are probed one physical button at a time, successful mappings are persisted in the route graph and replayed later, and mappings are evicted after repeated failure rather than hard-coding semantic button meanings. Active linear dialogue preempts movement and learns its advance button causally; closing a dialogue now releases the button immediately and applies a bounded post-dialogue button guard while stick movement disengages Link from the same interaction zone.
- Sticky objective lock v1: trackable strategic goals are replaced only when their structured completion predicate becomes true; local waypoint completion/stuck/scene transitions no longer cause strategic churn.
- Four-frame structured-state stack for temporal behaviour and combat timing.
- Online PPO learner with GAE and separate actor/learner weights.
- RND intrinsic curiosity.
- Observable reward shaping without a scripted Zelda quest path, including coarse-area dwell pressure rescaled so it remains an anti-loop signal without dominating PPO returns, frontier exploration, stale-waypoint `intent_progress` fadeout, and delta-based resource rewards (rupees/ammo/health/magic) that naturally suppress full-capacity pickups.
- Native one-shot chest-open telemetry from `FLAG_SCENE_TREASURE`, surfaced as `chest_opened`.
- Persistent atomic PyTorch V3 training checkpoint (`.local/ml/raw-controller-ppo-rnd-v3.pt`) plus route graph and immutable completion champions under `.local/ml/champions/`; the incompatible V2 checkpoint is left untouched and each new champion freezes both policy and route-memory snapshot used by evaluation.
- High-level `AgentIntent` cognition contract with `ObjectiveCompletion`; the LLM chooses the next strategic objective, while local systems own transient movement/interactions.
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
5. deliberately traverse a non-straight detour around collision, confirm route nodes/edges grow, then revisit the same target and verify `rota aprendida` follows the observed multi-waypoint path instead of returning to the direct wall attractor;
6. verify combat behaviour improves across repeated enemy encounters;
7. verify **PARAR** produces immediate neutral input even during provider inference or PPO training;
8. verify token totals and Codex remaining quota update independently of motor control;
9. after a real completion, verify both policy and route-memory champion files are created only after final work settles;
10. evaluate that champion from a new save and confirm `run_updates == 0`, policy/route SHA values stay unchanged, and repeat performance is measurable.

## Current limitation

A fresh PPO policy begins largely random. Autonomy V3 provides the learning architecture; it is not yet evidence that an untrained policy can complete OoT. The next milestone is real-game training and reward/feature tuning based on observed learning curves rather than adding semantic controller macros.
