from zelda_ai.autonomy.room_map import RoomMapMemory
from zelda_ai.autonomy.routes import LearnedRouteGraph
from zelda_ai.models import (
    ActorObservation,
    NavigationMeshSnapshot,
    NavigationProbe,
    SceneExitObservation,
    TraversalAffordanceObservation,
)


def test_room_map_persists_observed_room_geometry(tmp_path, state):
    path = tmp_path / "room-map-v1.json"
    memory = RoomMapMemory(path)
    game = state.model_copy(deep=True)
    game.player.position = (20.0, 0.0, 30.0)
    game.room_actors = [
        ActorObservation(
            actor_uid="door-1",
            actor_id=9,
            name="Door",
            category=10,
            category_name="door",
            params=0,
            position=(160.0, 0.0, 0.0),
            distance=143.2,
        )
    ]
    game.scene_exits = [
        SceneExitObservation(
            exit_index=1,
            entrance_index=20,
            position=(180.0, 0.0, 0.0),
            samples=6,
            direct_reachable=True,
        )
    ]
    game.navmesh = NavigationMeshSnapshot(
        origin=(20.0, 0.0, 30.0),
        step=40.0,
        half_extent=1,
        cells=[
            (0, 0, 0.0, 15),
            (1, 0, 0.0, 8),
        ],
    )
    game.navigation_probes = [
        NavigationProbe(
            direction="forward",
            distance=70.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=False,
        ),
        NavigationProbe(
            direction="right",
            distance=70.0,
            floor_found=False,
            wall_hit=True,
            wall_distance=35.0,
            wall_flags=2,
        ),
    ]
    game.traversal_affordances = [
        TraversalAffordanceObservation(
            kind="ladder_up",
            direction="up",
            approach_position=(40.0, 0.0, 30.0),
            target_position=(40.0, 100.0, 30.0),
            distance=20.0,
            height_delta=100.0,
            wall_flags=1,
        )
    ]

    assert memory.observe(game, now_s=1.0) is True
    memory.save(force=True)

    loaded = RoomMapMemory(path, writable=False)
    stats = loaded.stats(game)
    assert stats["rooms"] == 1
    assert stats["entries"] == 1
    assert stats["exits"] == 1
    assert stats["doors"] == 1
    assert stats["walkable_cells"] >= 2
    assert stats["blocked_cells"] >= 1
    assert stats["affordances"] == 1
    assert stats["current_room"]["visits"] == 1
    assert loaded.load_error == ""


def test_room_map_remembers_actual_room_transition_as_escape(tmp_path, state):
    memory = RoomMapMemory(tmp_path / "room-map-v1.json")
    routes = LearnedRouteGraph(tmp_path / "routes.json")

    origin = state.model_copy(deep=True)
    origin.player.position = (0.0, 0.0, 0.0)
    memory.observe(origin, now_s=1.0)

    leaving = origin.model_copy(deep=True)
    leaving.seq += 1
    leaving.player.position = (120.0, 0.0, 0.0)
    memory.observe(leaving, now_s=2.0)

    destination = leaving.model_copy(deep=True)
    destination.seq += 1
    destination.room = 1
    destination.player.position = (10.0, 0.0, 0.0)
    memory.observe(destination, now_s=3.0)

    revisit = origin.model_copy(deep=True)
    revisit.seq += 10
    revisit.player.position = (0.0, 0.0, 0.0)

    candidates = memory.known_escape_candidates(revisit)
    assert candidates
    assert candidates[0]["kind"] == "transition"
    assert candidates[0]["position"] == (120.0, 0.0, 0.0)

    hint = memory.remembered_escape_waypoint(revisit, routes)
    assert hint is not None
    assert hint["exit"] is True
    assert hint["remembered"] is True
    assert hint["memory_kind"] == "transition"
    assert hint["forced_escape"] is True
    assert hint["waypoint"] == (120.0, 0.0, 0.0)


def test_room_map_keeps_contexts_separate_and_read_only_is_frozen(tmp_path, state):
    path = tmp_path / "room-map-v1.json"
    writable = RoomMapMemory(path)
    writable.observe(state, now_s=1.0)

    other = state.model_copy(deep=True)
    other.room = 2
    other.player.position = (500.0, 0.0, 0.0)
    writable.observe(other, now_s=2.0)
    writable.save(force=True)
    before = path.read_bytes()

    frozen = RoomMapMemory(path, writable=False)
    changed = other.model_copy(deep=True)
    changed.player.position = (700.0, 0.0, 0.0)
    assert frozen.observe(changed, now_s=3.0) is False
    frozen.save(force=True)

    assert path.read_bytes() == before
    assert frozen.stats()["rooms"] == 2
