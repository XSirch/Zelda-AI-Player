from zelda_ai.combat_learning import combat_state_key, choose_action, enemy_key, update_policy
from zelda_ai.bridge import Bridge
from zelda_ai.models import ActorObservation, RunConfig
from zelda_ai.runtime import Runtime


def enemy(actor_id=7, category=5, name="Enemy"):
    return ActorObservation(actor_id=actor_id, category=category, category_name="enemy", room=0,
        params=0, name=name, position=(0, 0, 80), focus_position=(0, 20, 80),
        distance=80, targeted=False, drawn=True)


def test_enemy_key_is_per_class_and_player_age(state):
    first = enemy(actor_id=7)
    second = enemy(actor_id=8)
    assert enemy_key(state, first) != enemy_key(state, second)
    child_key = enemy_key(state, first)
    state.player.age = "adult"
    assert enemy_key(state, first) != child_key


def test_policy_learning_changes_preference():
    state_name = combat_state_key(distance=80, threat_score=1, locked=True)
    profile = {"enemy_key": "child:5:7", "encounters": 3, "policy": {"states": {}, "best_by_state": {}}}
    policy = profile["policy"]
    for _ in range(6):
        policy = update_policy(policy, state_name, "attack", 1.0)
    for _ in range(6):
        policy = update_policy(policy, state_name, "guard", -0.8)
    profile["policy"] = policy
    assert choose_action(profile, state_name, ["attack", "guard"], priors={}, step=20) == "attack"


def test_store_persists_enemy_learning_without_cross_model_leak(store, state):
    actor = enemy(actor_id=7, name="Deku Baba")
    key = enemy_key(state, actor)
    p1 = store.combat_profile("adaptive:model-a", key, actor_id=7, category=5,
        enemy_name="Deku Baba", create=True)
    assert store.combat_profile("adaptive:model-b", key, create=False) is None
    result = {"outcome": "win", "health_lost": 0, "confirmed_hits": 1, "attacks": 2,
        "dodges": 0, "guards": 0, "duration_ms": 900,
        "learning_trace": [
            {"state": "melee|low|locked|active", "action": "attack", "reward": 1.2, "detail": "hit"},
            {"state": "melee|low|locked|active", "action": "attack", "reward": 3.0, "detail": "defeat"},
        ]}
    updated = store.record_combat_encounter("adaptive:model-a", "run-x",
        {"enemy_key": key, "actor_id": 7, "category": 5, "enemy_name": "Deku Baba",
         "params": 0, "scene": 85, "room": 0}, result)
    assert updated["encounters"] == 1 and updated["wins"] == 1
    assert updated["policy"]["best_by_state"]["melee|low|locked|active"]["action"] == "attack"
    assert store.list_combat_profiles("adaptive:model-b") == []
    encounters = store.recent_combat_encounters(p1["id"])
    assert encounters[0]["outcome"] == "win"


def test_losses_update_separate_profile_counters(store, state):
    actor = enemy(actor_id=9, name="Stalfos")
    key = enemy_key(state, actor)
    store.record_combat_encounter("adaptive:model-a", "run-y",
        {"enemy_key": key, "actor_id": 9, "category": 5, "enemy_name": "Stalfos",
         "params": 0, "scene": 5, "room": 2},
        {"outcome": "loss", "health_lost": 48, "learning_trace": [
            {"state": "point_blank|high|locked|active", "action": "attack", "reward": -4,
             "detail": "player_defeated"}]})
    profile = store.combat_profile("adaptive:model-a", key)
    assert profile["losses"] == 1
    assert profile["damage_taken"] == 48


def test_runtime_exposes_profiles_only_for_adaptive_namespace(store, state):
    actor = enemy(actor_id=7, name="Deku Baba")
    game = type(state).model_validate({**state.model_dump(), "room_actors": [actor.model_dump()],
        "room_actor_count": 1})
    runtime = Runtime(Bridge("x" * 32, True), store, {})
    runtime.config = RunConfig(provider="demo", model="deterministic-demo", memory_mode="adaptive")
    runtime.namespace = runtime.new_namespace(runtime.config)
    key = enemy_key(game, actor)
    store.combat_profile(runtime.namespace, key, actor_id=7, category=5,
        enemy_name="Deku Baba", create=True)
    context = runtime._enemy_learning_context(game)
    assert len(context) == 1
    assert context[0]["enemy_key"] == key
    runtime.config = runtime.config.model_copy(update={"memory_mode": "isolated"})
    assert runtime._enemy_learning_context(game) == []


def test_snapshot_contains_compact_combat_profiles(store, state):
    actor = enemy(actor_id=7, name="Deku Baba")
    runtime = Runtime(Bridge("x" * 32, True), store, {})
    runtime.config = RunConfig(provider="demo", model="deterministic-demo", memory_mode="adaptive")
    runtime.run_id = store.new_run(runtime.config.model_dump(), "simulator", "hash")
    runtime.namespace = runtime.new_namespace(runtime.config)
    store.combat_profile(runtime.namespace, enemy_key(state, actor), actor_id=7, category=5,
        enemy_name="Deku Baba", create=True)
    snap = runtime.snapshot()
    assert snap["combat_profiles"][0]["enemy_name"] == "Deku Baba"
    assert "policy" not in snap["combat_profiles"][0]
