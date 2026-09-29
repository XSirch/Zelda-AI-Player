from zelda_ai.autonomy.routes import LearnedRouteGraph
from zelda_ai.models import NavigationProbe, SceneExitObservation


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
