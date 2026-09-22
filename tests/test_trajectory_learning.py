from zelda_ai.bridge import Bridge
from zelda_ai.models import RunConfig
from zelda_ai.runtime import Runtime


def test_store_learns_deduplicates_and_scores_trajectory(store):
    origin = {"scene": 1, "room": 0, "position": (0.0, 0.0, 0.0), "yaw": 0}
    destination = {"scene": 1, "room": 1}
    actions = [
        {"skill": "turn", "args": {"direction": "right", "duration_ms": 500, "strength": .7, "slot": None}},
        {"skill": "move", "args": {"direction": "forward", "duration_ms": 700, "strength": .8, "slot": None}},
        {"skill": "interact", "args": {"direction": None, "duration_ms": 100, "strength": .5, "slot": None}},
    ]
    route_id = store.learn_trajectory("adaptive:test", origin, destination, actions)
    assert route_id
    assert store.learn_trajectory("adaptive:test", origin, destination, actions) == route_id
    route = store.best_trajectory("adaptive:test", 1, 0, (5.0, 0.0, 0.0), 0)
    assert route["id"] == route_id
    assert route["successes"] == 2
    store.trajectory_outcome(route_id, False)
    rows = store.list_trajectories("adaptive:test")
    assert rows[0]["failures"] == 1


def test_transition_promotes_only_autonomous_adaptive_navigation(store, state):
    bridge = Bridge("x" * 32, True)
    runtime = Runtime(bridge, store, {})
    runtime.config = RunConfig(provider="demo", model="deterministic-demo", memory_mode="adaptive")
    runtime.run_id = store.new_run(runtime.config.model_dump(), "simulator", "hash")
    runtime.namespace = runtime.new_namespace(runtime.config)
    runtime.state = "running"
    runtime._reset_trajectory_trace(state)
    runtime.trajectory_trace.append({"skill": "move", "args": {
        "direction": "forward", "duration_ms": 500, "strength": .7, "slot": None}})
    next_state = state.model_copy(update={"room": 1, "seq": state.seq + 1})
    runtime.on_state(next_state, state)
    route = store.best_trajectory(runtime.namespace, state.scene, state.room, state.player.position)
    assert route is not None
    assert (route["to_scene"], route["to_room"]) == (next_state.scene, next_state.room)


def test_human_hint_taints_current_trajectory(store, state):
    bridge = Bridge("x" * 32, True)
    bridge.state = state
    runtime = Runtime(bridge, store, {})
    runtime.config = RunConfig(provider="demo", model="deterministic-demo", memory_mode="adaptive")
    runtime.run_id = store.new_run(runtime.config.model_dump(), "simulator", "hash")
    runtime.namespace = runtime.new_namespace(runtime.config)
    runtime.state = "running"
    runtime._reset_trajectory_trace(state)
    runtime.trajectory_trace.append({"skill": "move", "args": {
        "direction": "forward", "duration_ms": 500, "strength": .7, "slot": None}})
    runtime.hint("go right")
    assert runtime.trajectory_tainted
    next_state = state.model_copy(update={"room": 1, "seq": state.seq + 1})
    runtime.on_state(next_state, state)
    assert store.best_trajectory(runtime.namespace, state.scene, state.room, state.player.position) is None


def test_trajectory_rejects_far_spawn_or_opposite_heading(store):
    origin = {"scene": 2, "room": 0, "position": (0.0, 0.0, 0.0), "yaw": 0}
    destination = {"scene": 2, "room": 1}
    actions = [{"skill": "move", "args": {
        "direction": "forward", "duration_ms": 500, "strength": .7, "slot": None}}]
    store.learn_trajectory("adaptive:match", origin, destination, actions)
    assert store.best_trajectory("adaptive:match", 2, 0, (500.0, 0.0, 0.0), 0) is None
    assert store.best_trajectory("adaptive:match", 2, 0, (0.0, 0.0, 0.0), 32767) is None
