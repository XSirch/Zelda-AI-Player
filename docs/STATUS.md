# Implementation status — realtime-input foundation v2.1

Current branch of record: `main`.  
SoH target: `HarbourMasters/Shipwright@d30fc192f2eb01ceea45bd1e12de61636cafbf86`.

The historical pre-v2 status is preserved as `STATUS_V1.md`.

## Implemented

Independent zero/unlimited budgets and UI labels; nonblocking stop/model validation with stale-result fencing; global run deadline; input-owner generations; versioned fast/full snapshots; consumed receipts; native discrete sequences with wall-clock watchdog and deduplication; stable actor lifetime IDs and a recoverable bounded event journal; local feedback-based steering and generic combat; conservative probed RT retreat and time-based stuck windows; skill-module extraction; versioned learning and removal of unsolicited replay; native installer migration/backup manifest; live transport/input telemetry.

Deployment/contracts: [REALTIME_V2.md](REALTIME_V2.md).

## Windows validation report after the v2 merge

A Windows/Python 3.14 run reported:

- **156 passed**
- **17 skipped**
- **1 failed**

The single failure was a test-environment text decoding issue: `Path.read_text()` selected Windows `cp1252` while reading UTF-8 TSX. The current revision makes repository source/fixture text access explicit UTF-8. Rerun the suite before recording a final all-green Windows count.

That run also emitted a Starlette/httpx deprecation warning and Pydantic serializer warnings from a contract fixture. They did not cause the failure.

## Other checks previously executed

- Standalone C++20 scheduler scenarios and the original InputLease test compiled and ran in the implementation environment. They do **not** compile `ZeldaAiBridge.cpp` against the full SoH SDK.
- Python compilation passed.
- TypeScript/TSX syntax transpilation passed there; this was not a React dependency/type check or production Vite build.
- No paid provider request, ROM/save upload or manual GitHub Actions run was performed.

## Not certified / next acceptance checks

- Rerun the complete local pytest suite after the UTF-8 fix.
- `cd web && npm run build` with resolved dependencies.
- Windows SoH reconfigure/build and in-game input/animation/targeting tests.
- Actual p50/p95/p99 latency under load.
- Robust combat across enemies/bosses, global navmesh navigation, goal-bound route replay and process-level isolation of the fast motor.

The local controller still contains heuristic skills and bounded fallbacks. This is not an autonomous completion claim.


## Windows SoH build observation — Visual Studio 2026

The pinned Shipwright successfully configured with generator `Visual Studio 18 2026`, toolset `v143`, Windows SDK 10.0.26100.0, and generated `soh.o2r`. The first full Release build reached `ZeldaAiBridge.cpp` and exposed an MSVC narrowing error where Shipwright's controller button field is wider than the protocol's 16-bit N64 button mask.

The current revision fixes this at the boundary with an explicit low-16-bit conversion (`ToN64PadState`) and adds a native scheduler test covering a wider raw button value. A full SoH rebuild is still required to certify the fix.
