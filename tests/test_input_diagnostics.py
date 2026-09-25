from zelda_ai.control.diagnostics import direction_is_safe, summarize_receipts
from zelda_ai.models import GameState, InputReceipt


def with_probe(state: GameState, **changes):
    probe = dict(direction="back", distance=70, floor_found=True, floor_y=0, delta_y=0,
                 floor_type=0, wall_hit=False, wall_distance=None, wall_flags=0)
    probe.update(changes)
    return GameState.model_validate({**state.model_dump(), "navigation_probes": [probe]})


def receipt(seq, **changes):
    data = dict(seq=seq, owner_epoch=1, status="completed", first_tick=seq, last_tick=seq,
                pressed=0x8000, released=0x8000, apply_latency_ms=seq / 100.0,
                client_to_consume_ms=seq, reason="")
    data.update(changes)
    return InputReceipt.model_validate(data)


def test_direction_guard_requires_short_observed_floor(state):
    assert not direction_is_safe(state, "back")
    assert direction_is_safe(with_probe(state), "back")
    assert not direction_is_safe(with_probe(state, floor_found=False), "back")
    assert not direction_is_safe(with_probe(state, delta_y=-40), "back")
    assert not direction_is_safe(with_probe(state, wall_hit=True, wall_distance=20), "back")


def test_receipt_summary_counts_edges_and_latency():
    rows = [receipt(i) for i in range(1, 21)]
    summary = summarize_receipts(rows, edge_button=0x8000, expected_edges=20)
    assert summary["commands"] == summary["consumed"] == summary["completed"] == 20
    assert summary["lost"] == 0
    assert summary["presses"] == summary["releases"] == 20
    assert summary["duplicate_presses"] == summary["duplicate_releases"] == 0
    assert summary["latency_ms"]["min"] == 1
    assert summary["latency_ms"]["p95"] == 19
    assert summary["latency_ms"]["p99"] == 20
    assert summary["latency_ms"]["max"] == 20
    assert summary["native_queue_ms"]["p95"] == .19


def test_receipt_summary_does_not_turn_missing_delivery_into_success():
    rows = [receipt(1), None, receipt(3, status="cancelled", first_tick=0, last_tick=0,
                                     pressed=0, released=0, apply_latency_ms=None)]
    summary = summarize_receipts(rows, edge_button=0x8000, expected_edges=3)
    # A missing receipt is still an attempted command and must count as lost.
    assert summary["commands"] == 3
    assert summary["consumed"] == 1
    assert summary["lost"] == 2
    assert summary["presses"] == summary["releases"] == 1


def test_hop_direction_field_accepts_engine_backflip_classification(state):
    state.player.hop_direction = 2
    assert state.player.hop_direction == 2
