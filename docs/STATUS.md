# Implementation status — V3, V4 motor foundation and local curriculum

## Expanded Laya walking dataset and native evaluation on 2026-10-04

The additional reference batch `g1-fb2d9b81b58b` completed **120 attempted tasks, 114 successful**, across six continuous native sessions. Only **1,788 consumed actions from successful reference tasks** entered the dataset. Four whole instances supply 1,186 training actions; one supplies 296 validation actions and another 306 test actions. No provider calls were made. The dataset loader verified causal records and checksums; all 268 collection JSON/JSONL files passed strict UTF-8 decoding. These are reference collection results, not Laya gameplay success.

A new numeric-conditioned candidate trained from the same pinned generic base for 1,000 updates with batch size 16, using four training sessions rather than the earlier single session. Argmax test MAE was **16.98**, versus **38.64** for the training-only constant baseline. The immutable expectation-decoder derivative kept the exact same weights and made no additional updates. A separate fresh reload measured actual decoded test MAE **15.57**, with 50-inference p95 **9.35 ms**. This dataset differs from the earlier one; their raw MAEs are not a paired before/after comparison.

In `g1-516e3939b56f`, the candidate completed **18/30 native short walking tasks** across three sessions of ten consecutive tasks, with the new evaluation seed `4102500`. There were 1,240 worker replies, **zero expirations and zero context rejections**. IPC response p95 was **29.14 ms**, maximum **43.56 ms**. Sampled policy-output age had p95 **67.91 ms** and maximum **91.41 ms**; it does not measure native consumption or full physical reaction. Reference steering blend and online updates were zero, strategic objectives stayed unchanged, and integrity checks passed. The harness normally closed its owned game processes and released input; no evaluation game remained running afterwards.

The accepted **100 ms decision-response target is met in this batch**, while control accuracy remains insufficient. The 18/30 result is not G2, a reliable navigation claim, or evidence of improvement over earlier candidates evaluated on a different sequence. Additional demonstrations alone have not established a solution to the remaining control failures. Promotion stays disabled; sword acquisition, learned climbing/buttons/combat and campaign completion remain unproven. The sanitized index retains this batch alongside the earlier successes and failures: [validation/laya_runtime_2026-10-04.json](validation/laya_runtime_2026-10-04.json). Runtime code for this batch was committed in `562bb2c`; subsequent edits only record evidence.

## Continuous Laya walking evaluation on 2026-10-04

The accepted target is **100 ms per local decision**. The frozen numeric-conditioned candidate now meets that response target in two small real SoH batches. It is still an experimental walking motor, with **8/15 successful short walking tasks in each batch**. Neither batch establishes reliable navigation, G2 qualification, sword acquisition or campaign completion. Sanitized evidence: [validation/laya_runtime_2026-10-04.json](validation/laya_runtime_2026-10-04.json).

The earlier 120-update pilot was reproduced as worse than the training-only constant baseline. Increasing head-only training to 1,000 updates still lost to that baseline (development MAE 36.04 versus 32.83). Updating the last two encoder layers improved development MAE to 28.61, but response age under concurrent game rendering was unreliable. The original 228-action dataset has been repeatedly inspected during diagnosis and is now development data.

Two changes address observed failures. First, the asynchronous local policy may refresh its nonblocking output at the existing 20 Hz motor cadence; otherwise replies arriving between 10 Hz policy samples expired before reaching the arbiter. The regression was reproduced before the fix. Ordinary PPO residual and camera-projection behavior is preserved. Second, a trainable numeric projection feeds current bounded telemetry into the actual pretrained Laya decision head. The frozen encoder caches only the constant schema and physical options, never live game values. This is a changed input representation, not a claim that textual numeric reasoning was learned. Numeric candidates currently use the last frame only; original text candidates remain compatible.

Reference collection used three continuous native sessions, with ordinary Arquivo 2 startup in isolated copied homes: **60 attempted tasks, 54 successful, 931 positive consumed actions**. Failed tasks remain in the evidence without positive labels. Whole native instances, including neighboring tasks across families, stay in one split: 306 training actions, 304 validation actions and 321 test actions. Only one session supplies each split, which limits generalization evidence.

| Candidate | Training updates / batch | Offline test stick MAE | Physical short walking |
| --- | --- | --- | --- |
| Numeric telemetry, argmax | 1,000 / 2 | 13.65 | 8/15 |
| Numeric telemetry, argmax | 1,000 / 16 | 12.63 | Not separately evaluated |
| Same batch-16 weights, expectation decoder | No additional updates | 10.77, fresh reload | 8/15 |
| Constant stick chosen from training labels | No model updates | 37.39 | Not a physical candidate |

The expectation decoder was chosen using validation predictions. Its immutable derivative keeps exactly the batch-16 weights and the parent's original argmax training metrics; the separately recorded fresh benchmark measures its actual decoder. Argmax classification accuracy and continuous-stick error have separate scopes. After these diagnostics the new validation/test splits have also been inspected; future promotion needs new independent holdouts.

In `g1-69de0f082946`, IPC response p95 was **29.26 ms**, maximum 34.84 ms (430 replies). In `g1-d19fcdce6b4a`, p95 was **24.93 ms**, maximum 38.48 ms (504 replies). Neither batch had expired responses or context rejections. Observed request-submission-to-arbiter ages had p95 66.56/65.93 ms and maxima 70.05/89.48 ms. The policy accepts replies only within 100 ms and allows at most one 50 ms motor tick for actuation. These measurements end at the arbiter: native input-consumption and complete physical reaction latency are **not** measured by these counters.

Evaluations chain five model-controlled tasks per native session, with zero reference steering blend, zero online updates and unchanged strategic objectives. Real consumed input receipts and actual positions establish physical movement; offline loss is not treated as gameplay success. The local worker never owns the bridge or invokes a provider. Outputs fail closed on stale/mismatched replies, changed game/task context, unsupported modal or ladder/ledge states, invalid values and stop. Evaluation checks candidate, original motor/map and pinned-source hashes. The stopped direct-play game was normally closed before these runs; all evaluation games use separate working copies. This does not assert that closing the original game could not save its state normally.

The four earlier native pilots made 18 attempts: nine reached candidate control and nine failed preparation. No candidate task succeeded in those pilots. Preparation failures are recorded separately from control failures. The cadence regression and late responses explain observed failures in those batches, without attributing every later movement failure to latency.

No screenshots or paid/provider requests were used. Original PPO and persistent maps remain separate. The primary INICIAR flow has not been switched to Laya. Learned buttons, climbing, combat, inventory, sword acquisition, broad navigation reliability and the campaign remain unqualified.

Validation: **318 pytest cases passed, 20 skipped**, with the existing Starlette/httpx warning. Skips require unavailable g++/clang++; this patch makes no C++ changes and claims no new native build. Web production build and focused Ruff checks passed. Final encoding verification is recorded in the evidence index.

## Generic Laya base specialization pilot on 2026-10-04

The operator clarified that the starting point is the generic Laya base, specialized exclusively for Zelda. The local generic weights were verified against the public `convaiinnovations/laya` repository at revision `7b928d828b7b0e022f929d9bd2e44165aa270148`: SHA-256 `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`. Only those matching weights were copied; the pinned revision supplied the configuration/tokenizer files. No trading checkpoint or trading dataset was loaded.

The isolated Python 3.13 environment `.local/laya-env` successfully installed Laya source `573e5b62696ba441230cd6be71d593331b5d23af`, transformers 4.57.6 and torch 2.11.0+cu128. `requirements-laya.lock.txt` was generated from the successful installation, and `scripts/setup_laya.ps1` was executed successfully against that lock. The application environment and its existing PPO/maps remain separate.

Dataset: 228 consumed physical reference actions from twelve successful tasks in the earlier 24-attempt surface demonstration batch. Six whole episodes/112 actions train, three/70 validate, and three/46 test. All source failures remain in the original batch; neither failed tasks nor failed preparations are positive training labels. Features are bounded to four causal walking frames and two quantized executed-stick axes. The profile contains no buttons, ladder/ledge locomotion, combat, aiming, dialogue, inventory operation or sword-acquisition demonstration.

The real local pilot completed **120 supervised updates** to **26,512,131 trainable decision-layer parameters**, keeping the pretrained encoder frozen. GPU: NVIDIA GeForce RTX 4070 Laptop, 8 GB. Base files passed their pinned digest checks after training. Candidate heads are immutable and remain local under `.local/laya/candidate-20261004`; promotion is disabled. Audited summary: [validation/laya_base_2026-10-04.json](validation/laya_base_2026-10-04.json).

| Offline reserved metric | Before, validation | After, validation | After, test |
| --- | --- | --- | --- |
| Individual analog-axis accuracy | 5.0% | 13.6% | 19.6% |
| Both axes correct together | 0% | 0% | 0% |
| Mean absolute executed-stick error, native units | 78.36 | 53.23 | 47.26 |

A constant stick chosen only from training-label frequencies (`[-20, -20]`) had test-axis accuracy 21.7% and mean absolute error **32.83**, better than this candidate. Loss reduction or improvement over the untuned base therefore does **not** establish useful learning. The data volume, numeric input representation, head-only specialization and training settings require separate investigation; none is asserted as the sole cause.

A fresh reload reproduced the candidate's test predictions and measured **50 end-to-end local inferences**: median **90.0 ms**, p95 **118.4 ms**, maximum **142.7 ms**. These times include tokenization, host/device transfer, the two-axis forward pass and output readback. The operator subsequently accepted **100 ms per decision as the initial target**, replacing the proposed 50 ms criterion for this pilot. The median fits that target; p95 exceeds it. These measurements do not include native input-consumption latency or establish full gameplay reaction time. Decision quality remains the main limitation of this candidate. There was no physical Laya gameplay evaluation and no new claim of navigation or campaign ability. Laya tactical planning with a separate fast motor remains an untested architectural option, not a delivered capability.

No cognition/provider calls were made. The pilot commands cannot send native controller input and are not selected by the primary UI. During the preceding direct-play attempt, the Kokiri Sword was **not obtained**; its continuous helper was explicitly stopped and the game inputs released before this training work. That unsuccessful attempt was not used as a successful demonstration.

Validation: **309 pytest cases passed, 20 skipped**, with the existing Starlette/httpx deprecation warning; the skips require unavailable g++/clang++. Web production build and focused Ruff checks passed. UTF-8 verification is recorded in the sanitized index. This patch has no native C++ changes; it does not claim a new native build.

## Physical local-learning comparison on 2026-10-04

The isolated batch `g1-da24d7be4d70` completed **94 physical SoH trials**: 24 reference demonstration attempts, 30 before-training evaluations, 30 after-training evaluations and 10 outbound portal retention tests. The audited index is [validation/surface_learning_2026-10-04.json](validation/surface_learning_2026-10-04.json). Full native observations, actual input receipts, copied working saves and immutable candidate checkpoints remain local under `.local/qualification/g1-da24d7be4d70/`.

| Family | Before training | After training | G2 |
| --- | --- | --- | --- |
| Observed short surface ascent | 0/10 | 2/10 | Unqualified |
| Observed short surface descent | 0/10 | 9/10 | Unqualified |
| Observed cell after a withheld-movement stall | 0/10 | 10/10 | Unqualified |
| Outbound house portal retention, existing V3 policy | — | 10/10 | Retention pilot only |

The before/after comparison uses the same network initialization and the same family/setup schedule, with fresh native processes and ordinary Arquivo 2 loads. The evaluation seed differs from the demonstration schedule. Exact poses are not savestates and can vary with actual startup/movement timing. Both controllers use current observed targets and neutral verification guards; the candidate owns raw analog steering with **zero reference steering blending**. This measures a small local control domain rather than learned route discovery.

- Twelve of the 24 demonstration attempts succeeded, four in each family. Only their **228 consumed reference actions** entered the separate imitation dataset. Failed attempts remain in the records. The candidate trained for 120 local gradient steps; MSE fell from 0.236369 to 0.007460. Physical evaluation improved from **0/30 to 21/30**; loss reduction alone was not treated as gameplay success.
- All nine after-training failures are preserved. Seven ascent attempts could not expose/select a supported short upward surface during bounded preparation; one ascent controller attempt stalled; one descent preparation could not select its surface. Ascent is therefore **2/10 overall**, not a qualified capability. The three families have only ten reserved attempts each, below the 100-per-family gate; no candidate was promoted.
- Scope: child Link, normal world, house interior, native `stairs_or_slope` observations with short height changes (observed successful ascent/descent here is approximately 14 units). These are not evidence of loft/ladder traversal, full staircases, aiming, combat, general collision recovery or campaign completion. The recovery perturbation is specifically consumed neutral input followed by another local task, not every kind of stuck condition.
- `LocalTask` gives a currently observed walking surface/cell exclusive analog authority, a bounded attempt, fresh stopped verification frames, actual height checks and typed failure/interruption. Unrelated PPO buttons and contextual probes cannot run under a walking task. Dialogue/pause/context changes preempt it; ladder/ledge attachment rejects walking projection. Surface failure puts the observed target in cooldown and adds negative memory without inventing edges; actual success heals that memory.
- Strategic intent/completion stays unchanged. All physical-task overrides remain excluded from PPO, whose latent residual/action probabilities retain their V3 semantics. Imitation labels are executed N64 analog actions in a separate network and never enter the PPO buffer as on-policy samples. The candidate freezes its action contract and fixed normalizer metadata together with weights; evaluation does not optimize or overwrite it.
- There were **zero cognition/provider calls**, zero V3 PPO updates, and unchanged SHA-256 values for original native executable/assets/configuration/saves, original policy/maps, frozen snapshots and both before/after candidate files. Physical keyboard/gamepad mappings were disabled only in QA copies. No teleports, game-memory writes, walkthrough routes or operator game input were used.
- Reported trial durations totalled 2071.596 s, with median 22.040 s and maximum 29.496 s. A 120 s async deadline bounds post-launch startup, preparation, motor execution and settlement; shutdown/checksum/reporting occur separately. Each physical analog task has an eight-second attempt budget and a 2.5-second geometric stall limit.

The corrected development pilot `g1-9ae41d451c84` had 5/6 after-training successes versus 0/6 before, plus 2/2 outbound retention. An earlier pilot `g1-97717e656889` preceded the preparation/geometry corrections. Three development batches (`g1-e47c0b0191a7`, `g1-2b18e9bd4f27`, `g1-44666e720230`) were deliberately stopped by the source-hash fence during corrections. None of these attempts is substituted into the 94-trial measurement. Source-hash stops do not mean that original game saves/assets were rewritten.

Reproduce with `train-local-surfaces` in README. `scripts/export_surfaces.py` audits the local records, native consumption, verification, unchanged objectives and artifact hashes before publishing a small whitelist index. The existing campaign PPO checkpoint and completion champions remain separate; this candidate is **not promoted**.

A separate current-source portal regression, `g1-684c7a5df049` with seed `4102032`, passed **10/10**, including five house exits and five actual forest-to-house revisits. It covered eight observed pose cells, three initial camera bins and all four consumed setup heading bins. The audited index is [validation/g1_retention_2026-10-04.json](validation/g1_retention_2026-10-04.json). Its command correctly returned code 2 and `finished_unqualified`, because ten episodes cannot satisfy the 100-episode G1 gate. This supplements the ten outbound checks above and does not replace the historical 99/100 qualification with a smaller batch.

Checks: `uv run pytest -q` **292 passed, 20 skipped**, with the existing Starlette/httpx deprecation warning; `web/npm run build` passed; the separate C++20 MSVC harness passed **19 cases**. The pytest C++ skips require unavailable g++/clang++ and are not counted as passes. Focused Ruff checks for the new task/imitation/curriculum/exporter/tests passed. No frontend or native adapter source changed in this step.

## Physical G1 batch qualified on 2026-10-04

The automatic batch `g1-3a53157b3632` completed 100 fresh physical SoH episodes with **99 verified successes**. It meets the proposed initial G1 threshold of at least 99/100 within 120 seconds per complete scenario. The sanitized, audited index is [validation/g1_2026-10-04.json](validation/g1_2026-10-04.json); full observations, preparation traces, native receipts and working save copies remain local under `.local/qualification/g1-3a53157b3632/`.

Scope: existing Arquivo 2, child Link, normal world, initial house and physical portal revisits from Kokiri Forest. The batch used 50 separate native process instances and ordinary controller-based save loads, with a frozen existing V3 policy and read-only route/room memories. There were zero cognition/provider calls and zero training updates. This proves the measured local motor behavior; new ML learning, loft/ladder starts, other ages/worlds, general navigation, combat and campaign completion remain unproven.

- All 100 attempts are retained, including episode 74, which timed out after 119.659 s on a forest revisit. Link started on the outside platform at height 100, later descended and finished at height -80 without crossing a portal. The trace contains a blocked scene-exit approach followed by other observed-door attempts; no exact root cause is claimed. No failed attempt was replaced or removed from the rate.
- Successful episodes covered ten measured position cells, two initial room contexts and three initial camera bins. Physical setup movements covered all four world-heading bins; motor traces covered all eight 45-degree camera bins. These are observed values, not credit for a requested direction or camera setting.
- The measured scenario durations totalled 1440.176 s; median 19.232 s, nearest-rank p95 26.633 s, maximum 119.659 s. The 120 s budget includes reset/process launch, native startup, legitimate collision-guided preparation, motor execution and destination verification. Reporting/checksum overhead is separate.
- Success requires a fresh, living, playable destination in the same native process, with no new save-load event, an actual scene/room crossing and consumed input receipts. Three fresh stopped destination frames verify settlement. The strategic objective stays unchanged during each attempt.
- Original executable, assets, configuration, saves, policy and memories passed SHA-256 checks. Each pair uses a new local native-home copy; physical keyboard/gamepad mappings are disabled only there. All owned SoH processes were closed after the batch. No teleports, game-memory position/camera writes, save-state API, hidden routes or human game input were used.
- Three earlier development pilots (4, 10 and 10 episodes) are excluded from the qualification batch. The main batch used a separate seed, `1042027`. It is an instrumented, previously trained policy evaluation, not a zero-shot benchmark.

Reproduce with the `qualify-g1` command documented in README. `scripts/export_g1.py` audits the full local observations/receipts and exports only whitelisted public evidence. Smaller pilot batches fail the G1 gate by design. The harness never constructs a cognition provider or silently substitutes simulation.

Checks for this change: `uv run pytest -q` **271 passed, 20 skipped**, with the existing Starlette/httpx deprecation warning; `web/npm run build` passed; the separate C++20 MSVC harness passed **19 cases** with assertions and `/W4 /WX`. The 20 pytest skips require unavailable g++/clang++; they are not reported as passing. The stale frontend source assertion was updated to match the already implemented per-socket reconnect callback. Focused Ruff checks passed. Written task text and all batch JSON evidence were checked with strict UTF-8 decoding.

## Foundation validated locally on 2026-10-04

This remains a game-agent laboratory. The master plan's complete P0–P7 system, G1 batch, skill coverage and campaign gates are not delivered by this foundation patch. No cognition calls were authorized or made. Learning improvement has not been measured in physical training; these gameplay episodes use a frozen existing V3 policy and read-only route/room memories.

The principal reproduced movement defect was the horizontal sign of camera-relative guidance. Pinned SoH computes `Math_Atan2S(relY, -relX)` and adds `Camera_GetInputDirYaw`; the previous Python projection used the opposite horizontal sign in ordinary worlds. The correction inverts that native transform and accounts for native mirrored-world inversion. Original tests asserted the erroneous sign and were corrected alongside an independent native-transform contract test.

Actual SoH validation used the existing Arquivo 2 with ordinary controller inputs, adapter `rt-input-v3.2`, wire protocol 3, upstream `d30fc192f2eb01ceea45bd1e12de61636cafbf86`. Startup selection was automated and consumed receipts were recorded. No teleports, save-state restores, HP writes, route scripts or human game input were used. Native autosave remains enabled; a local copy of the save directory was preserved before process restart.

- Before the sign correction, one frozen episode remained in scene 52, room 0 for 120.032 seconds and failed.
- After correction, portal crossings were observed between scene 52 (Link's house) and scene 85 (Kokiri Forest), including normal save reloads from `(1, 0, 95)`.
- The stricter final harness requires the destination to be playable with the transition cutscene finished. A normal reload then left the house in 7.427 seconds; return to the house took 1.843 and 1.862 seconds in subsequent episodes. Earlier measurements ended during transition and are kept separately in the evidence.
- Policy, route graph and room-map SHA-256 values stayed unchanged in every episode. At this foundation snapshot, G1 was still unqualified because the required 100 varied scenarios had not been executed; the subsequent batch above qualifies only its stated initial profile. Loft/ladder starts, other age conditions, combat, puzzles and campaign completion remain unproven.

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
