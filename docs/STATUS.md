# Implementation status — V3 and V4 motor foundation

## Validated locally on 2026-10-04

This remains a game-agent laboratory. The master plan's complete P0–P7 system, G1 batch, skill coverage and campaign gates are not delivered by this foundation patch. No cognition calls were authorized or made. Learning improvement has not been measured in physical training; these gameplay episodes use a frozen existing V3 policy and read-only route/room memories.

The principal reproduced movement defect was the horizontal sign of camera-relative guidance. Pinned SoH computes `Math_Atan2S(relY, -relX)` and adds `Camera_GetInputDirYaw`; the previous Python projection used the opposite horizontal sign in ordinary worlds. The correction inverts that native transform and accounts for native mirrored-world inversion. Original tests asserted the erroneous sign and were corrected alongside an independent native-transform contract test.

Actual SoH validation used the existing Arquivo 2 with ordinary controller inputs, adapter `rt-input-v3.2`, wire protocol 3, upstream `d30fc192f2eb01ceea45bd1e12de61636cafbf86`. Startup selection was automated and consumed receipts were recorded. No teleports, save-state restores, HP writes, route scripts or human game input were used. Native autosave remains enabled; a local copy of the save directory was preserved before process restart.

- Before the sign correction, one frozen episode remained in scene 52, room 0 for 120.032 seconds and failed.
- After correction, portal crossings were observed between scene 52 (Link's house) and scene 85 (Kokiri Forest), including normal save reloads from `(1, 0, 95)`.
- The stricter final harness requires the destination to be playable with the transition cutscene finished. A normal reload then left the house in 7.427 seconds; return to the house took 1.843 and 1.862 seconds in subsequent episodes. Earlier measurements ended during transition and are kept separately in the evidence.
- Policy, route graph and room-map SHA-256 values stayed unchanged in every episode. G1 remains unqualified: the required 100 varied scenarios were not executed. Loft/ladder starts, varied camera/age conditions, combat, puzzles and campaign completion are unproven.

The sanitized evidence index is [validation/v4_motor_2026-10-04.json](validation/v4_motor_2026-10-04.json). Complete reports and the bounded motor traces are local under `.local/qualification/`; they are not distributed with ROMs or saves.

## Foundation added

- A local task supervisor records preparation/execution/verification, verifies portals by physical scene/room change and keeps the strategic completion contract intact. Missing consumed inputs, room/campaign stagnation, death and a modal interaction without observed effects have separate bounded failure conditions. A fast watchdog pauses the run, releases controller authority and writes a whitelisted local incident ring. The modal timeout applies even while a non-interactive cutscene owns control. Automatic full save recovery and campaign replanning are still pending.
- Current directed collision-mesh links can propose a Dijkstra approach path. Waypoints are held for a bounded attempt and revalidated against the current mesh; these proposed corridors do not become learned traversal evidence or earn learned-route rewards. Long-range geometric planning and traversal qualification remain pending.
- PPO fragments close at interventions, cutscenes and actor-version boundaries; their bootstrap belongs to the last PPO-controlled endpoint. Overridden buttons, unconsumed native commands and stale actor-version batches are excluded. Latent residual and executed-stick semantics are preserved. These are correctness fixes, not proof of faster learning.
- The runtime contract was versioned for the changed motor semantics. Previously captured champions fail the compatibility check without being rewritten, moved or silently evaluated under different guidance.
- Camera-change neutralization is bounded, so repeated indoor cuts cannot suppress movement indefinitely. Observed and remembered exit aliases share cooldown identity. Death, process restart, native save loading and age/mirror changes cannot manufacture successful portal transitions.
- Context-action label changes and horizontal inertia alone cannot learn a physical button association. Native interaction learning requires a consumed probe plus a specific observed effect. Dialogue closure releases the button and keeps the re-entry guard even when no causal mapping can be credited.
- Startup observation and a native input fence allow bounded physical probing of the explicitly selected existing save. Unsafe and unknown menu modes fail closed; no semantic button mapping or direct save-loader API is exposed to the motor.
- The minimal panel reports blocked/modal/local-task states. It no longer presents a blocked local executor as active training.
- Browser restart QA exposed an expired-session reconnect loop. The panel now refreshes its local session credential on reconnection and binds callbacks to the specific socket. An actual backend restart recovered with one authenticated connection and eight observed realtime frames, without manual page reload or cognition input.

Validation: `uv run pytest -q` and `web/npm run build`; the final counts are recorded in the evidence index. The pytest native cases skip without g++/clang++; separately, the real C++20 scheduler harness passed 19 cases under MSVC with assertions and `/W4 /WX`. The pinned SoH Release build completed and the resulting executable supplied actual adapter 3.2 telemetry. Browser QA used agent-browser against an isolated local server, verified loading and no browser errors, and did not activate a cognition run.

The V3 inventory below documents the pre-existing architecture. Its remaining gameplay gates are still open.

## Implemented

- Minimal realtime panel: connection, operational thought, raw N64 inputs, tokens/cost/quota, start/stop.
- Continuous motor loop independent from LLM latency.
- Raw-action PPO actor-critic: residual analog stick + physical button bits.
- Goal-conditioned camera-relative guidance is now mixed deterministically with the sampled residual stick, so structured navigation authority cannot be numerically overwhelmed by a mature Beta distribution.
- Indoor/fixed-camera steering now preserves a world-space heading and reprojects it at the 20 Hz motor cadence from SoH's current camera input yaw. Abrupt camera cuts briefly neutralize stale navigation stick, then resume the same waypoint; camera changes do not invalidate the learned route, and dialogue/interaction overrides keep authority. A view eye/lookAt fallback exists for payloads without input yaw. This has regression coverage in code but still requires live SoH validation across real indoor camera-mode switches before claiming reliable indoor navigation.
- Stick/button entropy are trained separately; button entropy anneals to zero over the first 50k trained samples and expected button count is explicitly regularized to stop indefinite button-mashing.
- Collision-aware local goal guidance: observed probes bend a blocked direct heading toward a walkable side direction, hard blockage weakens the prior, prolonged dwell fades stale-target attraction, and cognition can replan on `guidance_blocked`.
- Persistent route memory v1: directed topological edges are learned only from paths Link actually traverses and replay through the real observed edge-entry gateway rather than coarse cell centroids. Active learned edges are health-checked during replay; stalled/expired edges accumulate failure cost, enter temporary cooldown and trigger alternate route/exit/frontier recovery. Frontier failures are now persistent negative memory too: repeatedly bad projected cells lose priority across runs, successful visits heal them, and high room-local failure pressure can force an observed exit under a trackable objective.
- Room-escape fallback: after ~20 s without meaningful local expansion/progress, an observed native scene exit is preferred; when the native exit scan does not expose one, `ACTORCAT_DOOR` observations from `room_actors` are used as generic room-exit targets. Door approach remains collision-aware until close range, contextual interaction discovery still owns button choice, and escape steering gets full analog authority so a bad PPO residual cannot pull Link back into the room. This addresses the reproduced failure mode where thousands of frontier attempts remained inside one interior for hours; live SoH validation is still required before calling indoor escape reliable.
- Room Map Memory v1: `.local/ml/room-map-v1.json` persists empirical room-local knowledge independently from the directed route graph. It records entry points, actual successful room transitions/departure positions, observed scene exits/doors, traversed/NavMesh/probe walkable cells, blocked probe endpoints and vertical traversal affordances, isolated by scene/room, age and mirrored world. On revisits, remembered successful transition points are preferred as escape targets when the current scan exposes no exit; route memory still decides whether a path was actually traversed. Evaluation uses a checksum-verified read-only room-map snapshot frozen with the champion.
- Exit-guidance regression: ordinary directly observed `scene_exit` guidance retains its prior strength (0.92 when directly reachable); only an explicit recovery/remembered room escape sets `forced_escape` and takes 100% analog authority. This preserves the existing controller contract while fixing the multi-hour interior loop.
- Known-room escape controller v1: for a trackable objective in an otherwise locally uninteresting room, an observed/proven boundary is pursued immediately instead of waiting 20 s. Actual remembered transition points outrank merely scanned exits. Forced escape targets are health-checked: ~3.5 s without geometric progress puts that target in ~20 s cooldown so a bad exit cannot remain an hour-long attractor. The panel exposes active escape key, age, failures, successes and cooldown count.
- Vertical traversal frontier v1: targetless exploration now prefers an unvisited collision-observed ladder/stair/ledge/climb-wall affordance before horizontal frontier wandering. Structured traversal owns 100% of analog steering, suppresses PPO button noise and is excluded from PPO rollouts; if the game exposes a contextual traversal action, one-button-at-a-time empirical discovery learns the physical button from observed ladder/ledge state or movement change. Forced escape may insert one of these traversal steps before an unreachable scene exit, addressing elevated interior starts such as lofts/upper floors without hard-coding a Zelda room.
- PPO no-progress cutoff: after 180 s of the same coarse local dwell without meaningful expansion/progress, further transitions are excluded from PPO rollouts until progress resets. The controller keeps playing, but hours of duplicate failed-room experience can no longer dominate the checkpoint; full-authority room-escape actions are also excluded because the latent residual did not cause the executed stick.
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
