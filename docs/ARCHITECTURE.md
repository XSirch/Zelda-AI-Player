# Arquitetura — Autonomy V3

## Objetivo

Ao clicar **INICIAR**, Link deve continuar recebendo inputs enquanto percepção, cognição e treinamento acontecem em paralelo. Não existe catálogo de skills na arquitetura ativa.

## Fluxo

```text
Ship of Harkinian
      │ structured state / events
      ▼
Authenticated UDP Bridge
      │
      ├──────────────► Cognition loop (Codex/OpenRouter)
      │                    │ AgentIntent only
      │                    ▼
      │              current high-level intent
      │
      ▼
Goal-conditioned ML Actor (10 Hz action sampling / 20 Hz lease renewal)
      │ raw stick_x / stick_y / physical button bits
      ▼
SoH input consumer

Experience ──► reward + RND curiosity ──► PPO learner thread
                                      │
                                      └── atomic weight publish ──► ML Actor
```

## ML actor

`autonomy/ml_policy.py` implements a hybrid actor-critic:

- two continuous stick dimensions through Beta distributions;
- nine independent Bernoulli physical-button outputs;
- critic/value head;
- PPO updates with GAE;
- RND predictor/target networks for intrinsic novelty reward.

The policy receives four structured frames, left-padded at episode start. It is not told semantic button meanings. V3 samples a latent Beta **residual stick**. When cognition supplies an observed target/actor/direction, `goal_guidance()` projects that objective into camera-relative stick space and the executed N64 stick becomes a deterministic mix of guidance + residual. PPO log-probs are computed on the latent residual action. This keeps on-policy credit assignment valid while preventing a mature Beta head from overpowering structured navigation authority.

The 9 Bernoulli button outputs no longer share one large entropy term with the stick. Stick/button entropy coefficients are separate; button entropy anneals linearly to zero over the first 50k trained samples, while expected active-button count receives an explicit regularization cost. This converts exploration from a permanent objective into a temporary phase.

Route memory is a second, empirical navigation layer. `LearnedRouteGraph` quantizes only positions Link actually occupies, stores only directed transitions Link actually traverses, and persists them in `.local/ml/route-graph-v1.json`. A target query searches the reachable observed graph for a path whose endpoint makes useful geometric progress; it may therefore replay either a complete known route or a partial known prefix toward an unseen target. The route is cached between motor ticks and each waypoint is still collision-checked locally. Partial endpoints are marked exhausted once reached and are not replayed until new graph structure is observed. Graph contexts are isolated by scene/room, child/adult state and mirrored-world flag.

When cognition deliberately chooses `explore` without a structured target, route memory also selects a **local observed frontier** rather than handing full stick authority to stochastic PPO. It considers only current navigation probes with observed floor and no wall, prefers unseen/low-visit cells, but commits to one projected world-space frontier for several seconds instead of greedily reselecting a relative direction each motor tick. The commitment ends on arrival, ~2.5 seconds without geometric progress or a ~7 second maximum age; failed frontiers receive a short retry cooldown. Rearward probes are excluded whenever any forward/side probe is walkable. After ~20 seconds without local expansion, an observed `scene_exit` outranks another interior frontier; a known route to that exit is reused when available, otherwise the exit surface itself becomes the temporary guidance target. No reverse edge, hidden collision map or Zelda route is inferred.

Contextual interactions form a second empirical memory inside the same route-graph artifact. A key combines the observed `context_action` with the current context actor category/id. For an unknown door/interaction, the controller probes one physical button at a time (excluding START), suppresses PPO training for those overridden actions, and records a mapping only after an immediate observable effect such as scene/room transition, dialogue or durable progress. Known mappings are replayed, and three failed replays remove the mapping so it can be relearned. This is causal affordance discovery, not a semantic hard-code such as `Open -> A`.

### Waypoint completion

Movement intents with structured targets are monitored against observed player position. Reaching the waypoint emits a deduplicated `intent_target_reached` strategic trigger; cognition then selects the next observed waypoint instead of leaving a stale point active. Interaction intents are not considered complete merely by proximity.

## Concurrency

`autonomy/controller.py` renews short native input leases every 50 ms and samples a new ML action every two ticks. The learner trains a private network copy in `asyncio.to_thread`; only completed updates publish weights to the actor. A slow LLM call therefore does not create a movement pause.

Stopping revokes bridge authority first. An already-running gradient update may finish and atomically checkpoint, but cannot keep sending controller input.

## Cognition

`AgentIntent` contains objective, short spectator summary, mode, optional observed actor/coordinate target, optional direction/choice and a bounded horizon. It contains no skill name and no raw controller action.

The cognition prompt uses a compact strategic subset of structured game state, dialogue, observed actors, ML progress, learned world edges and recent events. Unknown transitions remain unknown until observed.

Cognition is sparse and event-driven. There is one initial call; subsequent calls are requested only by strategic evidence such as a scene/room transition, durable progress, semantic dialogue choice/resolution, death/game-over, boss defeat, or sustained motor stagnation. Target/context-action churn does not trigger cognition, and `AgentIntent.horizon_ms` is not a periodic refresh timer. Stuck replanning starts only after 90 seconds without useful progress and is rate-limited to at most one stuck-triggered request per 180 seconds.

## Reward

`autonomy/reward.py` uses observable evidence: surprise-gated RND novelty, new spatial cells/actors, scene transitions, durable game progress, explicit observed objective milestones, dialogue/context changes, target-distance progress, enemy health deltas, damage/death, stagnation and button-activity penalties. Generic native events do not earn reward.

There is no forced Kokiri/Saria/Mido route. Milestones such as Kokiri Sword or story flags can carry larger bounded training bonuses when observed, independent of order.

Long-horizon anti-loop shaping is separate from per-step stagnation. The tracker remembers coarse 500×160×500 scene/room macro-regions. After 60 seconds without a previously unseen macro-region or meaningful game progress, `local_dwell` becomes negative and ramps only to roughly -0.02/action over the next 180 seconds. The previous -0.35/action scale produced up to ~-3.5 reward/second at 10 Hz and dominated the return without identifying a better action. The dwell timer itself is unchanged, so it still drives diagnostics and sparse cognition's `local_area_stuck` trigger after 90 seconds.

Reward v7 adds a directional, game-generic exploration signal: `frontier_progress` only pays when Link sets a new maximum radius from the current local exploration anchor, and a first visit to a new coarse macro-region pays `new_macro_region`. Returning/circling at an already achieved radius cannot farm this reward. Straight-line target-distance shaping fades for stale targets and is disabled while collision detours or learned routes are active, because a correct path around an obstacle may temporarily increase Euclidean distance.

Reward v7 also treats resource collection as observable state change rather than pickup intent: positive rupee/ammo/health/magic deltas earn small bounded rewards; unchanged values earn zero, which naturally suppresses pickups attempted at full capacity. New persistent inventory items remain on the stronger objective-milestone path. Native `FLAG_SCENE_TREASURE` transitions are surfaced as one-shot `chest_opened` events and rewarded separately from their contents.

## Persistence

- SQLAlchemy stores runs, calls, events, memories and empirically observed world edges.
- V3 residual-policy weights/optimizer/RND state persist in `.local/ml/raw-controller-ppo-rnd-v3.pt`; the incompatible V2 checkpoint is preserved but not loaded.
- Directed route memory persists separately in `.local/ml/route-graph-v1.json`; it changes navigation behaviour without changing the PPO network shape.
- On a training `game_completed`, the runtime first waits for queued/in-flight PPO work and the final partial rollout to settle, then snapshots the resulting checkpoint under `.local/ml/champions/completion-XXXX.pt` and route memory under `completion-XXXX.routes.json`, each with its own SHA-256.
- Completion snapshots are immutable. `best-completion.pt` / `best-completion.routes.json` are mutable convenience aliases for the shortest observed completed run; the default evaluation candidate is the latest completion because it usually contains the newest training.
- Evaluation loads both frozen artifacts, disables all PPO/RND optimization and route-memory writes, and uses deterministic actor actions (Beta mean for stick; Bernoulli probability threshold for buttons).
- Checkpoint writes/copies use temporary-file replacement.
- Existing SQLite tables from older versions may remain in an old database, but Autonomy V3 code no longer reads/writes skill trajectories or heuristic combat profiles.

## Train vs evaluation

`RunConfig.run_mode` is `train` by default. In training mode the actor is stochastic and online PPO/RND continues updating. In `evaluation` mode a selected champion is required; the actor is deterministic and its weights are frozen. Cognition remains sparse/event-driven in both modes because it supplies high-level intent rather than motor learning.

A frozen evaluation does not itself prove generalization unless it is started from an appropriate fresh/held-out game save. Save lifecycle remains under SoH/user control.

## Providers and usage

Codex/ChatGPT uses the official isolated Codex app-server profile. Token usage comes from `thread/tokenUsage/updated`. Quota is polled separately via `account/rateLimits/read`, so quota refresh neither invokes a model nor stops the motor loop. OpenRouter keeps its API usage/cost accounting.

## UI

The React panel intentionally exposes only:

- SoH/bridge/cognition connection;
- operational thought/intention;
- raw stick and currently pressed physical buttons;
- run tokens/cache/reasoning;
- API cost when the provider reports USD;
- Codex quota windows actually returned by the app-server;
- start/stop.

Advanced run records remain accessible through backend API endpoints, not the primary gameplay screen.
