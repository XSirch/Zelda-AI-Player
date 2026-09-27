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
ML Actor (10 Hz action sampling / 20 Hz lease renewal)
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

The policy receives four structured frames, left-padded at episode start. It is not told semantic button meanings.

## Concurrency

`autonomy/controller.py` renews short native input leases every 50 ms and samples a new ML action every two ticks. The learner trains a private network copy in `asyncio.to_thread`; only completed updates publish weights to the actor. A slow LLM call therefore does not create a movement pause.

Stopping revokes bridge authority first. An already-running gradient update may finish and atomically checkpoint, but cannot keep sending controller input.

## Cognition

`AgentIntent` contains objective, short spectator summary, mode, optional observed actor/coordinate target, optional direction/choice and a bounded horizon. It contains no skill name and no raw controller action.

The cognition prompt can use structured game state, dialogue, observed actors, current ML telemetry, learned world edges and recent events. Unknown transitions remain unknown until observed.

## Reward

`autonomy/reward.py` uses observable evidence: surprise-gated RND novelty, new spatial cells/actors, scene transitions, durable game progress, explicit observed objective milestones, dialogue/context changes, target-distance progress, enemy health deltas, damage/death, stagnation and button-activity penalties. Generic native events do not earn reward.

There is no forced Kokiri/Saria/Mido route. Milestones such as Kokiri Sword or story flags can carry larger bounded training bonuses when observed, independent of order.

## Persistence

- SQLAlchemy stores runs, calls, events, memories and empirically observed world edges.
- ML weights/optimizer/RND state persist in `.local/ml/raw-controller-ppo-rnd-v2.pt`.
- Checkpoint writes use temporary-file replacement.
- Existing SQLite tables from older versions may remain in an old database, but Autonomy V3 code no longer reads/writes skill trajectories or heuristic combat profiles.

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
