# Realtime input foundation (Native Bridge V3)

This document describes **Native Bridge v3.1 / Adapter v3.1**, the low-level SoH bridge used by Autonomy V3. The realtime wire protocol is `3`; protocol `2` adapters are intentionally incompatible so stale native builds fail clearly.

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

## Deployment

Re-run the integration script after native bridge changes:

```powershell
uv run python scripts/integrate_soh.py C:\Projetos\Shipwright-AI
```

Then rebuild Shipwright/SoH and launch it through `zelda-ai launch-soh` so token and port are supplied to the native adapter.
