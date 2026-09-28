"""Small transactional store; SQLite locally, SQLAlchemy URL for other deployments."""
from __future__ import annotations

import json
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
world_edges = Table("world_edges", metadata,
    Column("id", String, primary_key=True), Column("namespace", String, index=True),
    Column("from_scene", Integer, index=True), Column("from_scene_name", String),
    Column("from_room", Integer), Column("from_position", JSON),
    Column("to_scene", Integer, index=True), Column("to_scene_name", String),
    Column("to_room", Integer), Column("to_position", JSON), Column("entrance_index", Integer),
    Column("traversals", Integer, default=1), Column("created_at", Float), Column("updated_at", Float))
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

    def learn_world_edge(self, namespace: str, origin: dict, destination: dict) -> str | None:
        if not namespace or min(origin.get("scene", -1), origin.get("room", -1),
                                destination.get("scene", -1), destination.get("room", -1)) < 0:
            return None
        now = time.time()
        with self.engine.begin() as conn:
            row = conn.execute(select(world_edges).where(
                world_edges.c.namespace == namespace,
                world_edges.c.from_scene == origin["scene"],
                world_edges.c.from_room == origin["room"],
                world_edges.c.to_scene == destination["scene"],
                world_edges.c.to_room == destination["room"],
                world_edges.c.entrance_index == destination.get("entrance_index", -1),
            )).mappings().first()
            values = {
                "from_scene_name": origin.get("scene_name", ""),
                "from_position": list(origin.get("position") or []),
                "to_scene_name": destination.get("scene_name", ""),
                "to_position": list(destination.get("position") or []),
                "updated_at": now,
            }
            if row:
                conn.execute(world_edges.update().where(world_edges.c.id == row["id"]).values(
                    traversals=(row["traversals"] or 0) + 1, **values))
                return row["id"]
            edge_id = uid()
            conn.execute(world_edges.insert().values(id=edge_id, namespace=namespace,
                from_scene=origin["scene"], from_scene_name=origin.get("scene_name", ""),
                from_room=origin["room"], from_position=list(origin.get("position") or []),
                to_scene=destination["scene"], to_scene_name=destination.get("scene_name", ""),
                to_room=destination["room"], to_position=list(destination.get("position") or []),
                entrance_index=destination.get("entrance_index", -1), traversals=1,
                created_at=now, updated_at=now))
            return edge_id

    def world_neighbors(self, namespace: str, scene: int, room: int, limit: int = 12) -> list[dict]:
        if not namespace:
            return []
        with self.engine.connect() as conn:
            rows = conn.execute(select(world_edges).where(
                world_edges.c.namespace == namespace,
                world_edges.c.from_scene == scene,
                world_edges.c.from_room == room).order_by(
                    world_edges.c.traversals.desc(), world_edges.c.updated_at.desc()).limit(limit)).mappings()
            return [dict(row) for row in rows]

    def list_world_edges(self, namespace: str, limit: int = 100) -> list[dict]:
        if not namespace:
            return []
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(select(world_edges).where(
                world_edges.c.namespace == namespace).order_by(
                    world_edges.c.updated_at.desc()).limit(limit)).mappings()]

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
            segment_configs = dict(conn.execute(select(segments.c.id, segments.c.config)
                .where(segments.c.run_id == run_id)).all())
            segment_times = list(conn.execute(select(segments.c.started_at, segments.c.ended_at)
                .where(segments.c.run_id == run_id)).mappings())
            event_data = list(conn.execute(select(events.c.kind, events.c.data, events.c.created_at)
                .where(events.c.run_id == run_id)).mappings())
            run_row = conn.execute(select(runs.c.created_at, runs.c.status)
                .where(runs.c.id == run_id)).mappings().first()

        kinds = Counter(r["kind"] for r in event_data)
        totals = {key: 0 for key in ["input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens"]}
        unknown_usage = unknown_cost = 0
        cost = 0.0
        codex_calls = 0
        latencies = []
        breakdowns: dict[tuple[str, str, str | None], dict[str, Any]] = {}

        for row in data:
            usage = row["usage"] or {}
            call_usage_unknown = usage.get("input_tokens") is None or usage.get("output_tokens") is None
            unknown_usage += int(call_usage_unknown)
            for token_key in totals:
                totals[token_key] += usage.get(token_key) or 0

            config = segment_configs.get(row["segment_id"]) or {}
            provider = str(config.get("provider") or "unknown")
            model = str(usage.get("actual_model") or config.get("model") or "unknown")
            effort = config.get("effort")
            codex_calls += int(provider == "codex")
            call_cost_unknown = provider == "openrouter" and usage.get("cost_usd") is None
            unknown_cost += int(call_cost_unknown)
            call_cost = usage.get("cost_usd") or 0
            cost += call_cost
            if row["latency_ms"] is not None:
                latencies.append(row["latency_ms"])

            breakdown_key = (provider, model, effort)
            bucket = breakdowns.setdefault(breakdown_key, {
                "provider": provider,
                "model": model,
                "effort": effort,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "reasoning_output_tokens": 0,
                "total_tokens": 0,
                "unknown_usage_calls": 0,
                "unknown_cost_calls": 0,
                "known_cost_usd": 0.0,
                "_cost_unknown": provider == "codex",
            })
            bucket["calls"] += 1
            bucket["unknown_usage_calls"] += int(call_usage_unknown)
            bucket["unknown_cost_calls"] += int(call_cost_unknown)
            bucket["_cost_unknown"] = bucket["_cost_unknown"] or call_cost_unknown
            bucket["known_cost_usd"] += call_cost
            for token_key in totals:
                bucket[token_key] += usage.get(token_key) or 0
            bucket["total_tokens"] = bucket["input_tokens"] + bucket["output_tokens"]

        usage_by_model = []
        for bucket in breakdowns.values():
            cost_unknown = bool(bucket.pop("_cost_unknown"))
            bucket["known_cost_usd"] = round(bucket["known_cost_usd"], 8)
            bucket["cost_usd"] = None if cost_unknown else bucket["known_cost_usd"]
            usage_by_model.append(bucket)
        usage_by_model.sort(key=lambda row: (
            row["provider"], row["model"], row["effort"] or ""
        ))

        now = time.time()
        run_status = run_row["status"] if run_row else None
        started_at = float(run_row["created_at"]) if run_row else None
        elapsed_s = 0.0
        ended_at = None
        if segment_times:
            all_ended = all(row["ended_at"] is not None for row in segment_times)
            for row in segment_times:
                if row["started_at"] is None:
                    continue
                segment_end = row["ended_at"] if row["ended_at"] is not None else now
                elapsed_s += max(0.0, float(segment_end) - float(row["started_at"]))
            ended_values = [float(row["ended_at"]) for row in segment_times if row["ended_at"] is not None]
            if all_ended and ended_values:
                ended_at = max(ended_values)
        elif started_at is not None:
            terminal_events = [
                float(row["created_at"]) for row in event_data
                if row["kind"] in {"run_completed", "run_stopped", "run_interrupted"}
                and row["created_at"] is not None
            ]
            ended_at = max(terminal_events) if terminal_events else None
            effective_end = ended_at if ended_at is not None else now
            elapsed_s = max(0.0, effective_end - started_at)

        bosses = {json.dumps(r["data"], sort_keys=True) for r in event_data if r["kind"] == "boss_defeated"}
        return {
            **totals,
            "total_tokens": totals["input_tokens"] + totals["output_tokens"],
            "calls": len(data),
            "unknown_usage_calls": unknown_usage,
            "unknown_cost_calls": unknown_cost,
            "known_cost_usd": round(cost, 8),
            "cost_usd": None if codex_calls or unknown_cost else round(cost, 8),
            "usage_by_model": usage_by_model,
            "run_status": run_status,
            "started_at": started_at,
            "ended_at": ended_at,
            "elapsed_s": round(elapsed_s, 3),
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "deaths": kinds["player_died"],
            "boss_events": len(bosses),
            "game_completions": kinds["game_completed"],
            "interventions": kinds["human_hint"] + kinds["take_control"],
            "vision_calls": 0,
        }
