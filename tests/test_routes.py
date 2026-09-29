from zelda_ai.autonomy.routes import LearnedRouteGraph


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
    # 80u is within the lookahead threshold, so the first actionable waypoint
    # is the second observed eastward node, not the straight-line north target.
    assert hint["waypoint"][0] > 120.0
    assert abs(hint["waypoint"][2]) < 30.0
    assert loaded.stats()["routes_reused"] == 1

    # Re-reading the same target is the same route episode, not a new reuse.
    assert loaded.next_waypoint(start, positions[-1]) is not None
    assert loaded.stats()["routes_reused"] == 1


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
