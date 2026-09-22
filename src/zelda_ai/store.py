"""Small transactional store; SQLite locally, SQLAlchemy URL for other deployments."""
from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections import Counter
from typing import Any

from sqlalchemy import JSON, Boolean, Column, Float, Integer, MetaData, String, Table, create_engine, select

metadata = MetaData()
runs = Table("runs", metadata,
    Column("id", String, primary_key=True), Column("created_at", Float, nullable=False),
    Column("status", String, nullable=False), Column("config", JSON, nullable=False),
    Column("source", String, nullable=False), Column("fingerprint", String, nullable=False),
    Column("assisted", Boolean, default=False), Column("mixed", Boolean, default=False),
    Column("reason", String, default=""))
segments = Table("segments", metadata,
    Column("id", String, primary_key=True), Column("run_id", String, index=True),
    Column("started_at", Float), Column("ended_at", Float), Column("config", JSON), Column("namespace", String))
calls = Table("calls", metadata,
    Column("id", String, primary_key=True), Column("run_id", String, index=True),
    Column("segment_id", String, index=True), Column("created_at", Float),
    Column("status", String), Column("latency_ms", Float), Column("usage", JSON),
    Column("decision", JSON), Column("observation", JSON), Column("error", String))
events = Table("events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True), Column("run_id", String, index=True),
    Column("created_at", Float), Column("kind", String), Column("data", JSON))
memories = Table("memories", metadata,
    Column("id", String, primary_key=True), Column("namespace", String, index=True),
    Column("scene", Integer), Column("note", String), Column("created_at", Float))
trajectories = Table("trajectories", metadata,
    Column("id", String, primary_key=True), Column("namespace", String, index=True),
    Column("signature", String, index=True), Column("from_scene", Integer, index=True),
    Column("from_room", Integer), Column("start_position", JSON), Column("start_yaw", Integer),
    Column("to_scene", Integer), Column("to_room", Integer), Column("actions", JSON, nullable=False),
    Column("successes", Integer, default=1), Column("failures", Integer, default=0),
    Column("created_at", Float), Column("updated_at", Float))


def uid() -> str:
    return uuid.uuid4().hex


class Store:
    def __init__(self, url: str):
        self.engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
        metadata.create_all(self.engine)
        with self.engine.begin() as conn:
            if url.startswith("sqlite"):
                conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            # Never resume paid calls or controls implicitly after a process restart.
            conn.execute(runs.update().where(runs.c.status.in_(["running", "paused"])).values(
                status="interrupted", reason="runtime_restarted"))
            conn.execute(segments.update().where(segments.c.ended_at.is_(None)).values(ended_at=time.time()))
            conn.execute(calls.update().where(calls.c.status == "pending").values(
                status="interrupted", error="Runtime restarted; provider usage may be incomplete."))

    def close(self):
        self.engine.dispose()

    def new_run(self, config: dict, source: str, fingerprint: str) -> str:
        run_id = uid()
        with self.engine.begin() as conn:
            conn.execute(runs.insert().values(id=run_id, created_at=time.time(), status="running",
                config=config, source=source, fingerprint=fingerprint, assisted=False, mixed=False, reason=""))
        return run_id

    def update_run(self, run_id: str, **values):
        with self.engine.begin() as conn:
            conn.execute(runs.update().where(runs.c.id == run_id).values(**values))

    def segment(self, run_id: str, config: dict, namespace: str) -> str:
        segment_id = uid()
        with self.engine.begin() as conn:
            conn.execute(segments.update().where(segments.c.run_id == run_id, segments.c.ended_at.is_(None))
                .values(ended_at=time.time()))
            conn.execute(segments.insert().values(id=segment_id, run_id=run_id,
                started_at=time.time(), config=config, namespace=namespace))
        return segment_id

    def end_segments(self, run_id: str):
        with self.engine.begin() as conn:
            conn.execute(segments.update().where(segments.c.run_id == run_id, segments.c.ended_at.is_(None))
                .values(ended_at=time.time()))

    def begin_call(self, run_id: str, segment_id: str, observation: dict) -> str:
        call_id = uid()
        with self.engine.begin() as conn:
            conn.execute(calls.insert().values(id=call_id, run_id=run_id, segment_id=segment_id,
                created_at=time.time(), status="pending", observation=observation, usage={}, decision=None))
        return call_id

    def finish_call(self, call_id: str, **values):
        with self.engine.begin() as conn:
            conn.execute(calls.update().where(calls.c.id == call_id).values(**values))

    def event(self, run_id: str, kind: str, data: dict | None = None):
        with self.engine.begin() as conn:
            conn.execute(events.insert().values(run_id=run_id, created_at=time.time(), kind=kind, data=data or {}))

    def remember(self, namespace: str, scene: int, note: str):
        note = note.strip()[:400]
        if not note:
            return
        with self.engine.begin() as conn:
            exists = conn.execute(select(memories.c.id).where(
                memories.c.namespace == namespace, memories.c.scene == scene, memories.c.note == note)).first()
            if not exists:
                conn.execute(memories.insert().values(id=uid(), namespace=namespace, scene=scene,
                    note=note, created_at=time.time()))

    def recall(self, namespace: str, scene: int | None = None, limit: int = 8) -> list[dict]:
        query = select(memories).where(memories.c.namespace == namespace)
        if scene is not None:
            query = query.where(memories.c.scene == scene)
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(query.order_by(memories.c.created_at.desc())
                .limit(limit)).mappings()]

    def learn_trajectory(self, namespace: str, origin: dict, destination: dict,
                         actions: list[dict]) -> str | None:
        safe_actions = []
        for action in actions[:96]:
            skill = action.get("skill")
            args = action.get("args")
            if skill not in {"move", "turn", "interact", "wait", "camera_center",
                              "roll", "backflip", "sidestep"} or not isinstance(args, dict):
                return None
            safe_actions.append({"skill": skill, "args": {
                "direction": args.get("direction"), "duration_ms": args.get("duration_ms"),
                "strength": args.get("strength"), "slot": args.get("slot"),
                "choice_index": args.get("choice_index")}})
        if not safe_actions or not any(a["skill"] in {"move", "turn"} for a in safe_actions):
            return None
        signature_payload = {"from": [origin["scene"], origin["room"]],
            "to": [destination["scene"], destination["room"]], "actions": safe_actions}
        signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        now = time.time()
        with self.engine.begin() as conn:
            row = conn.execute(select(trajectories).where(
                trajectories.c.namespace == namespace,
                trajectories.c.signature == signature)).mappings().first()
            if row:
                conn.execute(trajectories.update().where(trajectories.c.id == row["id"]).values(
                    successes=(row["successes"] or 0) + 1, updated_at=now))
                return row["id"]
            route_id = uid()
            conn.execute(trajectories.insert().values(id=route_id, namespace=namespace,
                signature=signature, from_scene=origin["scene"], from_room=origin["room"],
                start_position=list(origin.get("position") or []),
                start_yaw=origin.get("yaw"), to_scene=destination["scene"],
                to_room=destination["room"], actions=safe_actions, successes=1, failures=0,
                created_at=now, updated_at=now))
            return route_id

    def best_trajectory(self, namespace: str, scene: int, room: int,
                        position: tuple[float, float, float] | None = None,
                        yaw: int | None = None) -> dict | None:
        with self.engine.connect() as conn:
            rows = [dict(r) for r in conn.execute(select(trajectories).where(
                trajectories.c.namespace == namespace,
                trajectories.c.from_scene == scene,
                trajectories.c.from_room == room).order_by(trajectories.c.updated_at.desc())
                .limit(20)).mappings()]
        candidates = []
        for row in rows:
            successes, failures = row.get("successes") or 0, row.get("failures") or 0
            if successes < 1 or failures >= max(2, successes * 2):
                continue
            distance = 0.0
            start = row.get("start_position") or []
            if position is not None and len(start) == 3:
                distance = math.dist(position, start)
                if distance > 180:
                    continue
            yaw_distance = 0
            if yaw is not None and row.get("start_yaw") is not None:
                yaw_distance = abs(((yaw - row["start_yaw"] + 32768) % 65536) - 32768)
                if yaw_distance > 16384:
                    continue
            row["_score"] = successes * 3 - failures * 4 - distance / 120 - yaw_distance / 8192
            candidates.append(row)
        if not candidates:
            return None
        best = max(candidates, key=lambda r: (r["_score"], r.get("updated_at") or 0))
        best.pop("_score", None)
        return best

    def trajectory_outcome(self, route_id: str, success: bool):
        with self.engine.begin() as conn:
            row = conn.execute(select(trajectories.c.successes, trajectories.c.failures)
                .where(trajectories.c.id == route_id)).mappings().first()
            if not row:
                return
            values = {"updated_at": time.time()}
            key = "successes" if success else "failures"
            values[key] = (row[key] or 0) + 1
            conn.execute(trajectories.update().where(trajectories.c.id == route_id).values(**values))

    def list_trajectories(self, namespace: str, limit: int = 30) -> list[dict]:
        if not namespace:
            return []
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(select(trajectories).where(
                trajectories.c.namespace == namespace).order_by(trajectories.c.updated_at.desc())
                .limit(limit)).mappings()]

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self.engine.connect() as conn:
            result = [dict(row) for row in conn.execute(select(runs).order_by(runs.c.created_at.desc())
                .limit(limit)).mappings()]
        for run in result:
            run["metrics"] = self.metrics(run["id"])
        return result

    def detail(self, run_id: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(runs).where(runs.c.id == run_id)).mappings().first()
            if row is None:
                return None
            result = dict(row)
            for name, table in [("segments", segments), ("calls", calls), ("events", events)]:
                order = table.c.started_at if name == "segments" else table.c.created_at
                result[name] = [dict(r) for r in conn.execute(select(table).where(table.c.run_id == run_id)
                    .order_by(order.desc()).limit(200)).mappings()]
        result["metrics"] = self.metrics(run_id)
        return result

    def metrics(self, run_id: str) -> dict[str, Any]:
        with self.engine.connect() as conn:
            data = list(conn.execute(select(calls.c.usage, calls.c.latency_ms, calls.c.status,
                calls.c.segment_id).where(calls.c.run_id == run_id)).mappings())
            segment_providers = dict(conn.execute(select(segments.c.id, segments.c.config)
                .where(segments.c.run_id == run_id)).all())
            event_data = list(conn.execute(select(events.c.kind, events.c.data)
                .where(events.c.run_id == run_id)).mappings())
        kinds = Counter(r["kind"] for r in event_data)
        totals = {key: 0 for key in ["input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens"]}
        unknown_usage = unknown_cost = 0
        cost = 0.0
        codex_calls = 0
        latencies = []
        for row in data:
            usage = row["usage"] or {}
            unknown_usage += int(usage.get("input_tokens") is None or usage.get("output_tokens") is None)
            for key in totals:
                totals[key] += usage.get(key) or 0
            provider = segment_providers[row["segment_id"]]["provider"]
            codex_calls += int(provider == "codex")
            if provider == "openrouter" and usage.get("cost_usd") is None:
                unknown_cost += 1
            cost += usage.get("cost_usd") or 0
            if row["latency_ms"] is not None:
                latencies.append(row["latency_ms"])
        bosses = {json.dumps(r["data"], sort_keys=True) for r in event_data if r["kind"] == "boss_defeated"}
        return {**totals, "total_tokens": totals["input_tokens"] + totals["output_tokens"],
            "calls": len(data), "unknown_usage_calls": unknown_usage, "unknown_cost_calls": unknown_cost,
            "known_cost_usd": round(cost, 8), "cost_usd": None if codex_calls or unknown_cost else round(cost, 8),
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "deaths": kinds["player_died"], "boss_events": len(bosses),
            "interventions": kinds["human_hint"] + kinds["take_control"],
            "skill_failures": kinds["skill_failed"], "vision_calls": 0,
            "trajectories_learned": kinds["trajectory_learned"],
            "trajectory_replays": kinds["trajectory_replay_started"],
            "trajectory_replay_successes": kinds["trajectory_replay_succeeded"]}
