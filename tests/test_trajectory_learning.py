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


def test_transition_builds_observed_world_graph(store, state):
    bridge = Bridge("x" * 32, True)
    runtime = Runtime(bridge, store, {})
    runtime.config = RunConfig(provider="demo", model="deterministic-demo", memory_mode="adaptive")
    runtime.run_id = store.new_run(runtime.config.model_dump(), "simulator", "hash")
    runtime.namespace = runtime.new_namespace(runtime.config)
    runtime.state = "running"

    origin = type(state).model_validate({**state.model_dump(), "scene_name": "Link's House",
        "player": {**state.player.model_dump(), "position": [10, 0, 20]}})
    destination = type(state).model_validate({**state.model_dump(), "seq": state.seq + 1,
        "scene_epoch": state.scene_epoch + 1, "scene": 84, "scene_name": "Kokiri Forest",
        "room": 0, "entrance_index": 187, "player": {**state.player.model_dump(),
            "position": [-30, 0, 45]}})
    runtime.on_state(destination, origin)

    edges = store.world_neighbors(runtime.namespace, origin.scene, origin.room)
    assert len(edges) == 1
    assert edges[0]["from_scene_name"] == "Link's House"
    assert edges[0]["to_scene_name"] == "Kokiri Forest"
    assert edges[0]["entrance_index"] == 187
    assert edges[0]["from_position"] == [10.0, 0.0, 20.0]
    assert edges[0]["to_position"] == [-30.0, 0.0, 45.0]


def test_traverse_exit_learns_destination_only_after_transition(store, state):
    bridge = Bridge("x" * 32, True)
    runtime = Runtime(bridge, store, {})
    runtime.config = RunConfig(provider="demo", model="deterministic-demo", memory_mode="adaptive")
    runtime.run_id = store.new_run(runtime.config.model_dump(), "simulator", "hash")
    runtime.namespace = runtime.new_namespace(runtime.config)
    runtime.state = "running"

    exit_position = [70.0, 0.0, 116.0]
    requested_position = [82.0, 0.0, 110.0]
    origin = type(state).model_validate({
        **state.model_dump(),
        "scene": 52,
        "scene_name": "Origin Interior",
        "room": 0,
        "capabilities": [*state.capabilities, "scene_exit_surfaces"],
        "scene_exits": [{
            "exit_index": 1,
            "entrance_index": 0x211,
            "position": exit_position,
            "samples": 6,
        }],
    })
    runtime._reset_trajectory_trace(origin)
    runtime.last_decision = {
        "goal": "Leave this interior",
        "summary": "Traverse the observed exit surface.",
        "skill": "traverse_exit",
        "args": {
            "direction": None, "duration_ms": 8000, "strength": 0.7, "slot": None,
            "choice_index": None, "song": None, "target_actor_id": None,
            "target_actor_uid": None, "target_actor_params": None,
            "target_position": requested_position, "stop_distance": None, "item_id": None,
        },
        "memory_note": None,
    }
    runtime.trajectory_trace.append({
        "skill": "traverse_exit",
        "args": runtime.last_decision["args"],
    })

    # No transition yet => there is no learned destination.
    assert store.world_neighbors(runtime.namespace, origin.scene, origin.room) == []

    # Latest fast/full evidence shows Link actually standing on the selected
    # exit surface immediately before the scene change.
    bridge.previous_state = origin.model_copy(update={
        "seq": origin.seq + 1,
        "player": origin.player.model_copy(update={
            "position": tuple(exit_position),
            "floor_exit_index": 1,
        }),
    })

    destination = origin.model_copy(update={
        "seq": origin.seq + 1,
        "scene_epoch": origin.scene_epoch + 1,
        "scene": 84,
        "scene_name": "Observed Destination",
        "room": 0,
        "entrance_index": 187,
        "scene_exits": [],
    })
    runtime.on_state(destination, origin)

    edges = store.world_neighbors(runtime.namespace, origin.scene, origin.room)
    assert len(edges) == 1
    # The learned edge is anchored to the bridge-observed exit surface, not
    # the model's approximate requested coordinate.
    assert edges[0]["from_position"] == exit_position
    assert edges[0]["from_position"] != requested_position
    assert edges[0]["to_scene_name"] == "Observed Destination"

    route = store.best_trajectory(runtime.namespace, origin.scene, origin.room,
                                  tuple(origin.player.position), origin.player.yaw)
    assert route is not None
    assert route["actions"][0]["skill"] == "traverse_exit"


def test_non_exit_transition_keeps_player_origin_even_near_scene_exit(store, state):
    bridge = Bridge("x" * 32, True)
    runtime = Runtime(bridge, store, {})
    runtime.config = RunConfig(provider="demo", model="deterministic-demo", memory_mode="adaptive")
    runtime.run_id = store.new_run(runtime.config.model_dump(), "simulator", "hash")
    runtime.namespace = runtime.new_namespace(runtime.config)
    runtime.state = "running"

    player_position = [10.0, 0.0, 20.0]
    unrelated_exit = [25.0, 0.0, 20.0]
    origin = type(state).model_validate({
        **state.model_dump(),
        "scene": 52, "scene_name": "Interior", "room": 0,
        "player": {**state.player.model_dump(), "position": player_position},
        "scene_exits": [{
            "exit_index": 1, "entrance_index": 0x211,
            "position": unrelated_exit, "samples": 4,
            "direct_reachable": True,
        }],
    })
    runtime.last_decision = {
        "goal": "Open observed door", "summary": "Use door",
        "skill": "interact_with_actor",
        "args": {
            "direction": None, "duration_ms": 5000, "strength": 0.7, "slot": None,
            "choice_index": None, "song": None, "target_actor_id": None,
            "target_actor_uid": None, "target_actor_params": None,
            "target_position": None, "stop_distance": None, "item_id": None,
        },
        "memory_note": None,
    }
    bridge.previous_state = origin.model_copy(update={
        "seq": origin.seq + 1,
        "player": origin.player.model_copy(update={
            "position": tuple(player_position),
            "floor_exit_index": 0,
        }),
    })
    destination = origin.model_copy(update={
        "seq": origin.seq + 2, "scene_epoch": origin.scene_epoch + 1,
        "scene": 84, "scene_name": "Observed Destination", "room": 0,
        "scene_exits": [],
    })
    runtime.on_state(destination, origin)

    edges = store.world_neighbors(runtime.namespace, origin.scene, origin.room)
    assert len(edges) == 1
    assert edges[0]["from_position"] == player_position
    assert edges[0]["from_position"] != unrelated_exit


def test_traverse_exit_does_not_anchor_planned_surface_if_different_exit_triggered(store, state):
    bridge = Bridge("x" * 32, True)
    runtime = Runtime(bridge, store, {})
    runtime.config = RunConfig(provider="demo", model="deterministic-demo", memory_mode="adaptive")
    runtime.run_id = store.new_run(runtime.config.model_dump(), "simulator", "hash")
    runtime.namespace = runtime.new_namespace(runtime.config)
    runtime.state = "running"

    surface_a = [70.0, 0.0, 116.0]
    surface_b = [35.0, 0.0, 40.0]
    origin = type(state).model_validate({
        **state.model_dump(),
        "scene": 52, "scene_name": "Interior", "room": 0,
        "scene_exits": [
            {"exit_index": 1, "entrance_index": 0x211,
             "position": surface_a, "samples": 5, "direct_reachable": True},
            {"exit_index": 2, "entrance_index": 0x212,
             "position": surface_b, "samples": 5, "direct_reachable": True},
        ],
    })
    runtime.last_decision = {
        "goal": "Use chosen exit", "summary": "Traverse A",
        "skill": "traverse_exit",
        "args": {
            "direction": None, "duration_ms": 8000, "strength": 0.7, "slot": None,
            "choice_index": None, "song": None, "target_actor_id": None,
            "target_actor_uid": None, "target_actor_params": None,
            "target_position": surface_a, "stop_distance": None, "item_id": None,
        },
        "memory_note": None,
    }
    bridge.previous_state = origin.model_copy(update={
        "seq": origin.seq + 1,
        "player": origin.player.model_copy(update={
            "position": tuple(surface_b),
            "floor_exit_index": 2,
        }),
    })
    destination = origin.model_copy(update={
        "seq": origin.seq + 2, "scene_epoch": origin.scene_epoch + 1,
        "scene": 84, "scene_name": "Observed Destination", "room": 0,
        "scene_exits": [],
    })
    runtime.on_state(destination, origin)
    edge = store.world_neighbors(runtime.namespace, origin.scene, origin.room)[0]
    assert edge["from_position"] == surface_b
    assert edge["from_position"] != surface_a
