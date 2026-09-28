from pathlib import Path

from zelda_ai.autonomy.champions import ChampionStore


def test_champion_store_creates_immutable_completion_snapshots(tmp_path):
    source = tmp_path / "training.pt"
    source.write_bytes(b"policy-v1")
    store = ChampionStore(tmp_path / "champions")

    first = store.capture(
        source,
        {
            "run_id": "run-1",
            "elapsed_s": 500.0,
            "updates": 10,
            "samples_trained": 2560,
        },
    )
    source.write_bytes(b"policy-v2")
    second = store.capture(
        source,
        {
            "run_id": "run-2",
            "elapsed_s": 400.0,
            "updates": 20,
            "samples_trained": 5120,
        },
    )

    assert first["id"] == "completion-0001"
    assert second["id"] == "completion-0002"
    assert (store.root / "completion-0001.pt").read_bytes() == b"policy-v1"
    assert (store.root / "completion-0002.pt").read_bytes() == b"policy-v2"

    catalog = store.catalog()
    assert catalog["count"] == 2
    assert catalog["latest"]["id"] == "completion-0002"
    assert catalog["best_completion"]["id"] == "completion-0002"
    assert (store.root / "best-completion.pt").read_bytes() == b"policy-v2"

    latest, latest_path = store.resolve()
    assert latest["id"] == "completion-0002"
    assert latest_path.name == "completion-0002.pt"

    best, best_path = store.resolve("best")
    assert best["id"] == "completion-0002"
    assert best_path.name == "completion-0002.pt"

    explicit, explicit_path = store.resolve("completion-0001")
    assert explicit["run_id"] == "run-1"
    assert explicit_path.read_bytes() == b"policy-v1"


def test_champion_store_rejects_missing_or_invalid_ids(tmp_path):
    store = ChampionStore(tmp_path / "champions")
    for champion_id in (None, "latest", "best", "../escape", "completion-x"):
        try:
            store.resolve(champion_id)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{champion_id!r} unexpectedly resolved")
