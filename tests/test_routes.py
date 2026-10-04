from zelda_ai.autonomy.routes import LearnedRouteGraph
from zelda_ai.models import ActorObservation, NavigationProbe, SceneExitObservation


def _at(state, position, seq):
    row = state.model_copy(deep=True)
    row.seq = seq
    row.player.position = position
    return row


def test_route_graph_learns_persists_and_reuses_observed_detour(tmp_path, state):
    path = tmp_path / "route-graph-v1.json"
    graph = LearnedRouteGraph(path)

    # The learned route goes east first, then bends north around an imagined
    # obstacle. No direct start -> target shortcut is ever observed.
    positions = [
        (0.0, 0.0, 0.0),
        (80.0, 0.0, 0.0),
        (160.0, 0.0, 0.0),
        (160.0, 0.0, 80.0),
        (80.0, 0.0, 160.0),
        (0.0, 0.0, 160.0),
        (0.0, 0.0, 240.0),
    ]
    for index, position in enumerate(positions, start=1):
        graph.observe(_at(state, position, 10 + index), now_s=float(index))

    assert graph.stats()["edges"] == len(positions) - 1
    graph.save(force=True)
    assert path.is_file()

    loaded = LearnedRouteGraph(path, writable=False)
    start = _at(state, positions[0], 100)
    hint = loaded.next_waypoint(start, positions[-1])

    assert hint is not None
    assert hint["path_nodes"] == len(positions)
    # The first learned edge is replayed explicitly; nearby nodes are no longer
    # skipped far enough to cut corners through collision.
    assert 60.0 < hint["waypoint"][0] < 100.0
    assert abs(hint["waypoint"][2]) < 30.0
    assert loaded.stats()["routes_reused"] == 1

    # Re-reading the same target is the same route episode, not a new reuse.
    assert loaded.next_waypoint(start, positions[-1]) is not None
    assert loaded.stats()["routes_reused"] == 1


def test_route_replay_uses_actual_observed_edge_entry_not_cell_average(
    tmp_path, state
):
    graph = LearnedRouteGraph(tmp_path / "routes.json")
    start = (0.0, 0.0, 0.0)
    crossed = (81.0, 0.0, 10.0)
    graph.observe(_at(state, start, 1000), now_s=1.0)
    graph.observe(_at(state, crossed, 1001), now_s=2.0)

    source_id = next(iter(graph.edges))
    target_id = next(iter(graph.edges[source_id]))
    edge = graph.edges[source_id][target_id]
    assert tuple(edge["entry_position"]) == crossed

    # Simulate the coarse target node representative drifting toward another
    # part of the same 80u cell. Replay must still aim at the actually traversed
    # crossing point rather than this potentially unsafe centroid.
    graph.nodes[target_id]["position"] = [81.0, 0.0, 70.0]
    graph.reset_trace()

    hint = graph.next_waypoint(
        _at(state, start, 1002),
        (160.0, 0.0, 0.0),
        now_s=0.0,
    )

    assert hint is not None
    assert hint["edge_target"] == target_id
    assert hint["waypoint"] == crossed
    assert hint["edge_entry_position"] == crossed


def test_stalled_learned_edge_is_cooled_down_and_alternate_route_is_used(
    tmp_path, state
):
    graph = LearnedRouteGraph(tmp_path / "routes.json")
    start = (0.0, 0.0, 0.0)
    preferred = (81.0, 0.0, 0.0)
    alternate = (81.0, 0.0, 81.0)

    graph.observe(_at(state, start, 1100), now_s=1.0)
    graph.observe(_at(state, preferred, 1101), now_s=2.0)
    graph.reset_trace()
    graph.observe(_at(state, start, 1102), now_s=3.0)
    graph.observe(_at(state, alternate, 1103), now_s=4.0)
    graph.reset_trace()

    target = (240.0, 0.0, 0.0)
    stationary = _at(state, start, 1104)
    first = graph.next_waypoint(stationary, target, now_s=0.0)
    assert first is not None
    preferred_edge = first["edge_key"]

    held = graph.next_waypoint(stationary, target, now_s=1.0)
    assert held is not None
    assert held["edge_key"] == preferred_edge

    recovered = graph.next_waypoint(stationary, target, now_s=3.0)
    assert recovered is not None
    assert recovered["edge_key"] != preferred_edge
    assert graph.stats()["route_edge_abandoned"] == 1
    assert graph.stats()["route_edges_cooling_down"] == 1
    assert graph.route_recovery_needed(now_s=3.0) is True

    source, target_id = preferred_edge.split("->", 1)
    assert graph.edges[source][target_id]["failures"] == 1


def test_route_graph_reuses_partial_path_toward_unvisited_target(tmp_path, state):
    graph = LearnedRouteGraph(tmp_path / "routes.json")
    positions = [
        (0.0, 0.0, 0.0),
        (80.0, 0.0, 0.0),
        (160.0, 0.0, 0.0),
    ]
    for index, position in enumerate(positions, start=1):
        graph.observe(_at(state, position, 30 + index), now_s=float(index))

    start = _at(state, positions[0], 300)
    hint = graph.next_waypoint(start, (500.0, 0.0, 0.0))

    assert hint is not None
    assert hint["partial"] is True
    assert hint["target_gap"] > 240.0
    assert 60.0 < hint["waypoint"][0] < 100.0


def test_partial_route_is_not_replayed_until_graph_extends(tmp_path, state):
    graph = LearnedRouteGraph(tmp_path / "routes.json")
    positions = [
        (0.0, 0.0, 0.0),
        (80.0, 0.0, 0.0),
        (160.0, 0.0, 0.0),
    ]
    for index, position in enumerate(positions, start=1):
        graph.observe(_at(state, position, 500 + index), now_s=float(index))

    target = (500.0, 0.0, 0.0)
    start = _at(state, positions[0], 510)
    first = graph.next_waypoint(start, target)
    assert first is not None and first["partial"] is True

    # Once the bot reaches the end of the known partial branch, that whole
    # prefix is exhausted for this target instead of becoming a magnet.
    endpoint = _at(state, positions[-1], 511)
    assert graph.next_waypoint(endpoint, target) is None
    assert graph.stats()["exhausted_partial_nodes"] >= len(positions)
    assert graph.next_waypoint(start, target) is None

    # A genuinely new observed continuation changes graph revision and makes
    # the branch eligible again because it now reaches farther.
    graph.observe(_at(state, (240.0, 0.0, 0.0), 512), now_s=10.0)
    extended = graph.next_waypoint(start, target)
    assert extended is not None
    assert extended["partial"] is True
    assert extended["target_gap"] < first["target_gap"]


def test_route_graph_does_not_invent_reverse_or_unobserved_edges(tmp_path, state):
    graph = LearnedRouteGraph(tmp_path / "routes.json")
    positions = [
        (0.0, 0.0, 0.0),
        (80.0, 0.0, 0.0),
        (160.0, 0.0, 0.0),
        (240.0, 0.0, 0.0),
    ]
    for index, position in enumerate(positions, start=1):
        graph.observe(_at(state, position, 20 + index), now_s=float(index))

    forward = graph.next_waypoint(_at(state, positions[0], 200), positions[-1])
    reverse = graph.next_waypoint(_at(state, positions[-1], 201), positions[0])

    assert forward is not None
    assert reverse is None


def test_route_graph_isolates_age_and_mirrored_world(tmp_path, state):
    graph = LearnedRouteGraph(tmp_path / "routes.json")
    child_normal = _at(state, (0.0, 0.0, 0.0), 400)
    next_child = _at(state, (80.0, 0.0, 0.0), 401)
    graph.observe(child_normal, now_s=1.0)
    graph.observe(next_child, now_s=2.0)

    assert graph.next_waypoint(
        _at(state, (0.0, 0.0, 0.0), 402),
        (160.0, 0.0, 0.0),
    ) is not None

    before_edges = graph.stats()["edges"]

    adult = _at(state, (0.0, 0.0, 0.0), 403)
    adult.player.age = "adult"
    assert graph.next_waypoint(adult, (160.0, 0.0, 0.0)) is None
    graph.observe(adult, now_s=3.0)
    assert graph.stats()["edges"] == before_edges

    mirrored = _at(state, (0.0, 0.0, 0.0), 404)
    mirrored.mirrored_world = True
    assert graph.next_waypoint(mirrored, (160.0, 0.0, 0.0)) is None
    graph.observe(mirrored, now_s=4.0)
    assert graph.stats()["edges"] == before_edges


def test_targetless_exploration_prefers_unseen_open_probe(tmp_path, state):
    graph = LearnedRouteGraph(tmp_path / "routes.json")
    origin = _at(state, (0.0, 0.0, 0.0), 600)
    visited_forward = _at(state, (0.0, 0.0, 140.0), 601)
    graph.observe(origin, now_s=1.0)
    graph.observe(visited_forward, now_s=2.0)
    graph.reset_trace()

    origin.navigation_probes = [
        NavigationProbe(
            direction="forward",
            distance=140.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=False,
        ),
        NavigationProbe(
            direction="right",
            distance=140.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=False,
        ),
    ]

    hint = graph.exploration_waypoint(origin)

    assert hint is not None
    assert hint["frontier"] is True
    assert hint["direction"] == "right"
    assert hint["waypoint"][0] < -120.0 or hint["waypoint"][0] > 120.0


def test_frontier_commitment_does_not_flip_each_motor_tick(tmp_path, state):
    graph = LearnedRouteGraph(tmp_path / "routes.json")
    game = _at(state, (0.0, 0.0, 0.0), 650)
    game.player.yaw = 0
    game.navigation_probes = [
        NavigationProbe(
            direction="forward",
            distance=140.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=False,
        ),
        NavigationProbe(
            direction="left",
            distance=140.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=False,
        ),
    ]

    first = graph.exploration_waypoint(game, now_s=0.0)
    assert first is not None
    assert first["direction"] == "forward"

    # Even if the current probes now make left the only fresh candidate, hold
    # the world-space frontier briefly instead of greedily flipping directions.
    game.navigation_probes = [
        NavigationProbe(
            direction="left",
            distance=140.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=False,
        ),
    ]
    held = graph.exploration_waypoint(game, now_s=1.0)
    assert held is not None
    assert held["direction"] == "forward"
    assert held["waypoint"] == first["waypoint"]
    assert held["stable"] is True

    # With no progress, the commitment expires and the failed frontier is
    # temporarily blacklisted; only then may the selector choose left.
    replacement = graph.exploration_waypoint(game, now_s=3.0)
    assert replacement is not None
    assert replacement["direction"] == "left"
    assert graph.stats()["frontier_abandoned"] == 1


def test_failed_frontier_penalty_persists_and_changes_future_choice(
    tmp_path, state
):
    path = tmp_path / "routes.json"
    graph = LearnedRouteGraph(path)
    game = _at(state, (0.0, 0.0, 0.0), 655)
    game.player.yaw = 0
    game.navigation_probes = [
        NavigationProbe(
            direction="forward",
            distance=140.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=False,
        ),
        NavigationProbe(
            direction="right",
            distance=140.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=False,
        ),
    ]

    first = graph.exploration_waypoint(game, now_s=0.0)
    assert first is not None
    assert first["direction"] == "forward"
    failed_id = first["waypoint_id"]

    # No movement for > stall threshold marks this projected cell as a real
    # negative-memory frontier, not merely a 12-second transient cooldown.
    replacement = graph.exploration_waypoint(game, now_s=3.0)
    assert replacement is not None
    assert replacement["direction"] == "right"
    assert graph.frontier_failures[failed_id] == 1

    graph.save(force=True)
    loaded = LearnedRouteGraph(path)
    loaded_hint = loaded.exploration_waypoint(game, now_s=100.0)
    assert loaded_hint is not None
    assert loaded_hint["direction"] == "right"
    assert loaded.frontier_failures[failed_id] == 1
    assert loaded.stats()["frontier_failed_cells"] >= 1


def test_room_failure_pressure_is_context_local(tmp_path, state):
    graph = LearnedRouteGraph(tmp_path / "routes.json")
    game = _at(state, (0.0, 0.0, 0.0), 656)
    age = "adult" if game.player.age == "adult" else "child"
    current_prefix = (
        f"{int(bool(game.mirrored_world))}:{age}:"
        f"{game.scene}:{game.room}:"
    )
    graph.frontier_failures[current_prefix + "1:0:0"] = 3
    graph.frontier_failures[current_prefix + "2:0:0"] = 2
    graph.frontier_failures[
        f"{int(bool(game.mirrored_world))}:{age}:"
        f"{game.scene}:{game.room + 1}:1:0:0"
    ] = 20

    assert graph.room_failure_pressure(game) == 5


def test_frontier_does_not_choose_rearward_when_front_or_side_is_open(
    tmp_path, state
):
    graph = LearnedRouteGraph(tmp_path / "routes.json")
    origin = _at(state, (0.0, 0.0, 0.0), 660)
    visited_forward = _at(state, (0.0, 0.0, 140.0), 661)
    graph.observe(origin, now_s=1.0)
    graph.observe(visited_forward, now_s=2.0)
    graph.reset_trace()

    origin.navigation_probes = [
        NavigationProbe(
            direction="forward",
            distance=140.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=False,
        ),
        NavigationProbe(
            direction="back_left",
            distance=140.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=False,
        ),
    ]

    hint = graph.exploration_waypoint(origin, now_s=10.0)
    assert hint is not None
    # back_left is unseen, while forward was visited, but reverse exploration
    # still loses when any non-rear walkable option exists.
    assert hint["direction"] == "forward"


def test_exit_waypoint_prefers_direct_reachable_observed_exit(tmp_path, state):
    graph = LearnedRouteGraph(tmp_path / "routes.json")
    game = _at(state, (0.0, 0.0, 0.0), 700)
    game.scene_exits = [
        SceneExitObservation(
            exit_index=1,
            entrance_index=10,
            position=(40.0, 0.0, 0.0),
            samples=8,
            direct_reachable=False,
        ),
        SceneExitObservation(
            exit_index=2,
            entrance_index=20,
            position=(120.0, 0.0, 0.0),
            samples=4,
            direct_reachable=True,
        ),
    ]

    hint = graph.exit_waypoint(game)

    assert hint is not None
    assert hint["exit"] is True
    assert hint["exit_index"] == 2
    assert hint["direct_reachable"] is True
    assert hint["waypoint"] == (120.0, 0.0, 0.0)


def test_unreachable_exit_requires_route_or_door_evidence(tmp_path, state):
    graph = LearnedRouteGraph(tmp_path / "routes.json")
    game = _at(state, (0.0, 0.0, 0.0), 710)
    game.scene_exits = [
        SceneExitObservation(
            exit_index=1,
            entrance_index=10,
            position=(300.0, 0.0, 0.0),
            samples=6,
            direct_reachable=False,
        )
    ]

    assert graph.exit_waypoint(game) is None
    recovery_hint = graph.exit_waypoint(game, allow_unreachable=True)
    assert recovery_hint is not None
    assert recovery_hint["exit"] is True
    assert recovery_hint["direct_reachable"] is False
    assert recovery_hint["waypoint"] == (300.0, 0.0, 0.0)

    game.room_actors = [
        ActorObservation(
            actor_uid="door-near-exit",
            actor_id=9,
            name="Door",
            category=10,
            category_name="door",
            params=0,
            position=(280.0, 0.0, 0.0),
            distance=280.0,
        )
    ]
    hint = graph.exit_waypoint(game)

    assert hint is not None
    assert hint["exit"] is True
    assert hint["direct_reachable"] is False


def test_escape_waypoint_falls_back_to_observed_door_without_scene_exit(
    tmp_path, state
):
    graph = LearnedRouteGraph(tmp_path / "routes.json")
    game = _at(state, (0.0, 0.0, 0.0), 715)
    game.scene_exits = []
    game.room_actors = [
        ActorObservation(
            actor_uid="house-door",
            actor_id=9,
            name="Door",
            category=10,
            category_name="door",
            params=3,
            position=(180.0, 0.0, 20.0),
            distance=181.1,
        )
    ]

    hint = graph.escape_waypoint(game)

    assert hint is not None
    assert hint["exit"] is True
    assert hint["door"] is True
    assert hint["waypoint"] == (180.0, 0.0, 20.0)
    assert hint["exit_position"] == (180.0, 0.0, 20.0)
    assert hint["waypoint_id"].startswith("door:")
    assert graph.has_observed_escape(game) is True


def test_escape_waypoint_still_prefers_native_scene_exit_over_door(
    tmp_path, state
):
    graph = LearnedRouteGraph(tmp_path / "routes.json")
    game = _at(state, (0.0, 0.0, 0.0), 716)
    game.scene_exits = [
        SceneExitObservation(
            exit_index=4,
            entrance_index=40,
            position=(80.0, 0.0, 0.0),
            samples=5,
            direct_reachable=True,
        )
    ]
    game.room_actors = [
        ActorObservation(
            actor_uid="other-door",
            actor_id=9,
            name="Door",
            category=10,
            category_name="door",
            params=0,
            position=(30.0, 0.0, 0.0),
            distance=30.0,
        )
    ]

    hint = graph.escape_waypoint(game)

    assert hint is not None
    assert hint["exit"] is True
    assert hint.get("door") is not True
    assert hint["exit_index"] == 4
    assert hint["waypoint"] == (80.0, 0.0, 0.0)


def test_interaction_button_memory_persists_and_recovers_from_failures(tmp_path):
    path = tmp_path / "routes.json"
    graph = LearnedRouteGraph(path)
    key = "1:open:door:9"

    assert graph.interaction_button(key) is None
    graph.record_interaction_success(key, "A", now_s=1.0)
    assert graph.interaction_button(key) == "A"
    graph.save(force=True)

    loaded = LearnedRouteGraph(path)
    assert loaded.interaction_button(key) == "A"
    loaded.record_interaction_failure(key, "A", now_s=2.0)
    loaded.record_interaction_failure(key, "A", now_s=3.0)
    assert loaded.interaction_button(key) == "A"
    loaded.record_interaction_failure(key, "A", now_s=4.0)
    assert loaded.interaction_button(key) is None


def test_read_only_route_graph_never_learns_or_writes(tmp_path, state):
    path = tmp_path / "routes.json"
    writable = LearnedRouteGraph(path)
    writable.observe(_at(state, (0.0, 0.0, 0.0), 1), now_s=1.0)
    writable.observe(_at(state, (80.0, 0.0, 0.0), 2), now_s=2.0)
    writable.save(force=True)
    before = path.read_bytes()

    frozen = LearnedRouteGraph(path, writable=False)
    assert frozen.observe(_at(state, (160.0, 0.0, 0.0), 3), now_s=3.0) is False
    frozen.save(force=True)

    assert path.read_bytes() == before
    assert frozen.stats()["nodes"] == 2
