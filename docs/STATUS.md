# Implementation status — realtime-input foundation

Base: `XSirch/Zelda-AI-Player@13d44ce95f0df81b1a01669cb11ab0b94293449f`.
SoH target remains `HarbourMasters/Shipwright@d30fc192f2eb01ceea45bd1e12de61636cafbf86`.
The previous status record is preserved as `STATUS_V1.md`; its historical claims are not new validation of this change.

## Implemented

Independent zero/unlimited budgets and UI labels; nonblocking stop/model validation with stale-result fencing; global run deadline; input-owner generations; versioned fast/full snapshots; consumed receipts; native discrete sequences with wall-clock watchdog and deduplication; stable actor lifetime IDs and a recoverable bounded event journal; local feedback-based steering and generic combat; conservative probed RT retreat and time-based stuck windows; skill-module extraction; versioned learning and removal of unsolicited replay; native installer migration/backup manifest; live transport/input telemetry.

Deployment and exact contracts: [REALTIME_V2.md](REALTIME_V2.md).

## Validation actually executed in the implementation environment

- `python -m pytest -q`: **113 passed** in the available local test tree. This includes newly added budget/lifecycle/protocol/installer/RT tests and hydrated original locomotion, bridge-native and gameplay-v1 tests. It is **not the complete original repository suite**.
- C++20 standalone scheduler scenarios and the original InputLease test compiled and ran with the available compiler. They do **not** compile ZeldaAiBridge.cpp against the full SoH SDK.
- Python compilation (`python -m compileall -q src`) passed.
- TypeScript/TSX syntax transpilation passed using the available TypeScript installation. This is **not** a React dependency/type check or production Vite build.
- No paid provider request, ROM/save upload, gameplay memory modification or GitHub Actions run was performed.

Network checkout/dependency retrieval was unavailable in this environment (GitHub DNS and an npm install timeout). Files used locally were read through the connector; original contents were hash-checked where hydrated. No dependency lockfile was fabricated.

## Not certified / next acceptance checks

- Full original pytest suite and `cd web && npm run build` with resolved dependencies.
- Windows SoH reconfigure/build and in-game input/animation/targeting tests.
- Actual p50/p95/p99 latency under load. One-to-two input ticks is a design goal, not a measured result.
- Robust combat against all enemies/bosses, global navmesh navigation, trustworthy goal-bound route replay, and process-level isolation of the fast motor from persistence/UI.

The local controller still has heuristic skills and bounded fallbacks. This is not an autonomous completion claim. Keep previous builds/data available while validating the new native hook.
