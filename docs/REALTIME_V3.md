# Realtime input foundation (Native Bridge V3)

This document describes the low-level SoH bridge used by Autonomy V3, currently **Adapter v3.10**. The realtime wire protocol is `3`; protocol `2` adapters are intentionally incompatible so stale native builds fail clearly.

## Responsibilities

- authenticated localhost UDP transport;
- fast structured observations plus periodic full state;
- native input consumer scheduling;
- short leases and monotonic watchdog;
- owner epochs and stale/cross-scene rejection;
- exact input receipts when supported;
- event journal/gap detection;
- player, actor, dialogue, inventory/progress, terrain and combat-relevant raw telemetry;
- native scene autosave.

The bridge does **not** decide movement, navigation, combat or quest steps. Autonomy V3's local ML policy consumes the structured state and emits raw controller actions.

## Control cadence

The Python ML actor renews a short setpoint lease at ~20 Hz. If Python stops renewing, native code returns control to neutral. State feedback remains independent from provider inference.

## Structured geometry

Navigation probes, traversal observations, scene-exit surfaces and the compact collision NavMesh can remain in the wire state as observations. Autonomy V3 may learn from these features; it does not call a Python A* skill executor.

Adapter 3.10 first queries the existing 70-unit local lattice. If its current player cell has no walking links, it reobserves one 35-unit lattice with the same 81-cell limit. The shorter segments retain floor continuity, diagonal restrictions, body clearance, backface and lower-body wall checks. This is a current collision observation; no previous mesh edges are merged or stored as traversed routes. The independent native exit scan retains its 280-unit extent. A refined mesh may still contain an isolated root and must not authorize collision-blind movement. Experimental local tasks may consume these observations under their own bounded physical contracts; this does not qualify general navigation.

## Deployment

Re-run the integration script after native bridge changes:

```powershell
uv run python scripts/integrate_soh.py C:\Projetos\Shipwright-AI
```

Then rebuild Shipwright/SoH and launch it through `zelda-ai launch-soh` so token and port are supplied to the native adapter.
