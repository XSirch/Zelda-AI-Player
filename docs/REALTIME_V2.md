# Realtime input foundation (v2.7)

## Delivered scope

This change implements the input/observation foundation plus the local Navigation V2 collision mesh; it is still not a complete global navigation or combat system. Existing databases, saves and provider credentials are not deleted or migrated destructively. The model continues to return typed decisions; local skills own the controller.

- Independent resource budgets: `max_calls`, `max_tokens`, `max_cost_usd` and `max_runtime_s` accept **0 = unlimited for that field only**. Usage/cost accounting continues. Unknown usage/cost stays unknown. A finite token/cost cap still requires reconcilable accounting.
- `max_output_tokens` remains a per-response bound (256–16384 for the OpenRouter adapter), not an unlimited run budget. Disabling harness budgets does not remove provider quotas or billing.
- Stop/take-control revokes input before provider cleanup. Model validation occurs outside the control lock. Late inference/validation cannot resume a stopped run. Late billable usage is reconciled when available.
- A global monotonic run deadline also covers dialogue, cutscene, replay and model waits. A zero duration disables that deadline, not the native input watchdog.
- One input owner epoch and task-local generation fence. A stale skill's `finally` cannot release its successor. Neutral/cancel is distinct from human handoff.
- V2 uses latest-wins continuous setpoints and separately identified discrete sequences. Sequences advance at consuming `PadMgr_RequestPadData` reads (mode 1), not render frames. Duplicate packets neither replay a sequence nor extend its lease.
- Accepted, consumed and completed receipts are separate. The panel separates Python→consume latency from the sub-millisecond native accept→consume queue. Consumed input is NOT proof of gameplay effect; dodge skills use OoT's exact camera input yaw, wait until Player_ProcessControlStick reports the requested Link-relative direction, then verify HOPPING plus the engine-classified hop direction. Feedback loss ends renewal; the monotonic native watchdog releases control.
- Fast snapshots carry dynamic observations. Full snapshots at approximately 200 ms carry slow metadata. Fast samples require an exact full snapshot and matching scene/context. Gaps trigger resynchronization instead of mixing maps.
- Native actor lifetime IDs prevent selecting another enemy of the same type. Lock target and targeting candidate are distinct. A bounded event journal resends unacknowledged events and reports unrecoverable gaps.
- Basic generic combat acquires a real lock, approaches and alternates attacks with guarded recovery locally. It does not infer enemy animation openings, guarantee boss victories or use hidden solution flags.
- Movement uses new feedback rather than a fixed 100/140 ms sleep in the updated controllers. Stuck windows do not become shorter merely because sampling is faster. RT retreat uses observed short floor probes; missing floor stops that recovery instead of blindly reversing over a ledge.
- Full snapshots also carry a compact 9×9 moving NavMesh generated from SoH floor/wall collision. Reciprocal links require walk-safe height, multiple floor-continuity samples along each edge, no corner cutting and a body-width corridor. Python A* chooses local waypoints; fast 70-unit probes can veto a stale waypoint immediately. The raw mesh is not sent to the model prompt. The same collision sampling now exposes currently loaded `scene_exits` when a floor polygon has non-zero `SceneExitIndex`; `traverse_exit` walks onto that exact observed surface so interiors such as Link's House can transition without a door actor.
- Bridge v2.7 advertises `probe_yaw_v2`: navigation-probe labels now match OoT `PlayerStickDirection` exactly (positive relative yaw = left). The Python controller retains a legacy label mapping when connected to v2.5 so mixed-version startup fails safely instead of checking the wrong side.
- Local skill implementations moved from the monolithic runtime into `skills/`. Compatibility exports remain. Natural-language summary/goal strings no longer change the selected executable skill.
- The learning contract is versioned separately. Failed/intervened traces are not promoted as autonomous routes. Unsolicited replay before the planner is disabled; prior experience remains stored.

## Deploy

1. Stop the backend and game. Back up the application's data directory normally; do not remove its database/auth profile.
2. Use the current `main` (`git switch main; git pull --ff-only origin main`) and run `uv sync` and `uv run pytest -q`.
3. Run `uv run python scripts/integrate_soh.py <your-pinned-Shipwright-checkout>`.
   The installer recognizes the verified original or the previous hook, validates the reconstructed upstream file, backs up files outside the CMake source glob, and installs every required header. It does not reset the checkout or touch game assets.
4. Reconfigure and rebuild the pinned Shipwright C++ project. **An existing soh.exe does not change when adapter source changes.**
5. Run `npm install` and `npm run build` in `web`, then restart `uv run zelda-ai serve` and the newly built game.
6. The panel must report `rt-input-v2.7`, protocol 2, `local_navmesh`, `scene_exit_surfaces` and consumed receipts. An old game remains visibly legacy (protocol 1); it does not silently gain low-latency capability.

A new backend is compatible with the old V1 bridge for rollback/testing, but V1 acknowledgment means acceptance only. The new V2 native bridge requires this backend. To fully roll back, restore both the previous backend and game binary/adapter from the preserved backup.

## Unlimited example

```json
{
  "max_calls": 0,
  "max_tokens": 0,
  "max_cost_usd": 0,
  "max_runtime_s": 0,
  "max_output_tokens": 2048
}
```

Use zero only for the caps that should be disabled. There is no automatic retry of failed paid model calls.

## Validation and remaining work

See STATUS.md for checks actually executed. The standalone C++ scheduler and synthetic UDP tests do not certify SoH gameplay. Acceptance-to-consumption latency is measured inside the native process; it is not video latency or time until Link completes an action.

Still pending: the actual Windows/SoH integration build, recorded input/animation/navigation tests, measured latency and NavMesh sampling cost under load, full original project suite and production Vite build; global/cross-room route planning plus explicit special links and actor-obstacle avoidance; per-enemy/boss action adapters; goal-bound certified trajectory replay; full isolation of SQLite/UI work into another process. Full snapshots/events can still invoke synchronous persistence, so do not claim a hard real-time scheduling guarantee.

Validate pause/unpause, scene transitions, human handoff with held buttons, repeated A/B edges, lost/reordered packets, two identical enemies crossing, provider timeouts, and a finite budget beside an unlimited budget before long autonomous runs.


## Learned opponent policies

In Adaptive mode the runtime maintains a separate combat policy per enemy class inside the model+effort namespace. The policy is a small interpretable state/action table, not a universal hard-coded fight script. State buckets combine range, immediate threat, target lock and disabled/frozen observations. Safe available actions are scored with accumulated reward plus bounded exploration, so repeated encounters can converge on different tactics for different enemies.

The LLM sees compact summaries of visible learned enemy profiles at decision time. It chooses high-level intent such as whether to fight, equip or disengage; once `fight_enemy` begins, the opponent-specific local policy executes at the bridge feedback rate without per-hit model calls.
