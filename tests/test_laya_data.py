import json

import pytest

from zelda_ai.laya_data import (
    PROFILE,
    bounded_state,
    digest,
    export_surfaces,
    load_dataset,
    majority_baseline,
    quantize,
    validate_record,
)
from zelda_ai.laya_training import prepare_base, specialization_keys


def test_specialization_keys_keep_frozen_encoder_out_of_candidate():
    keys = {'head.weight', 'temperature', 'encoder.embeddings.weight'}
    keys.update(f'encoder.layers.{i}.weight' for i in range(28))
    assert specialization_keys(keys, 0) == {'head.weight', 'temperature'}
    assert specialization_keys(keys, 2) == {
        'head.weight', 'temperature', 'encoder.layers.26.weight', 'encoder.layers.27.weight'}
    for bad in (-1, 5, True, 1.5):
        with pytest.raises(ValueError):
            specialization_keys(keys, bad)


def record(episode="one"):
    return {"profile": PROFILE, "source": "soh", "controller": "reference",
            "first_tick": 4, "buttons": 0, "successful_episode": True,
            "reference_blend": 0, "episode_id": episode, "observation_seq": 3,
            "command_seq": 1, "state": bounded_state([[1, 0, 0, .5, 0, 0, 0, 0, 1]]),
            "executed_stick": [-55, -23], "labels": [-60, -20]}


@pytest.mark.parametrize("value,label", [(-55, -60), (-23, -20), (10, 0), (70, 60), (80, 80)])
def test_native_action_quantization(value, label):
    assert quantize(value) == label


@pytest.mark.parametrize("field,value", [("source", "simulator"), ("first_tick", 0),
                                         ("successful_episode", False), ("buttons", 32768),
                                         ("controller", "candidate"), ("reference_blend", .1),
                                         ("labels", [60, 20])])
def test_training_rejects_unproven_or_mislabeled_actions(field, value):
    row = record()
    row[field] = value
    with pytest.raises(ValueError):
        validate_record(row)


def test_inputs_are_bounded_and_cannot_include_hidden_flags():
    with pytest.raises(ValueError):
        bounded_state([[0]*9]*5)
    row = record()
    row["state"]["sword_chest_content"] = True
    with pytest.raises(ValueError):
        validate_record(row)


def test_dataset_loader_rejects_episode_leakage_and_changed_bytes(tmp_path):
    hashes = {}
    for name in ("train", "validation", "test"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(json.dumps(record("same_episode")), encoding="utf-8")
        hashes[name] = digest(path)
    (tmp_path / "manifest.json").write_text(json.dumps({"profile": PROFILE,
        "split_unit": "episode", "sha256": hashes}), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        load_dataset(tmp_path)
    (tmp_path / "train.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(tmp_path)


def test_base_bootstrap_rejects_other_weights_before_downloading(tmp_path):
    wrong = tmp_path / "trading.safetensors"
    wrong.write_bytes(b"other-model")
    destination = tmp_path / "new-base"
    with pytest.raises(ValueError, match="published generic"):
        prepare_base(destination, wrong)
    assert not destination.exists()


def test_comparator_cannot_choose_its_constant_action_from_reserved_labels():
    training = [record("train")]
    reserved = record("test")
    reserved["labels"] = [80, 80]
    reserved["executed_stick"] = [80, 80]
    baseline = majority_baseline(training, [reserved])
    assert baseline["stick"] == [-60, -20]
    assert baseline["axis_accuracy"] == 0
    assert baseline["stick_mae_native_units"] == 120


def test_export_reserves_whole_real_episodes_and_checks_receipts(tmp_path):
    suite = tmp_path / "suite"
    for i in range(3):
        path = suite / "demonstrations" / f"trial-{i}" / "motor" / "task.json"
        path.parent.mkdir(parents=True)
        action = {"source": "soh", "controller": "reference", "first_tick": i+1,
                  "buttons": 0, "successful_episode": True, "reference_blend": 0,
                  "command_seq": i+1, "observation_seq": i+10,
                  "features": [1, 0, 0, .5, 0, 0, 0, 0, 1], "action": [-.6875, -.2875]}
        task = {"success": True, "source": "soh", "provider_calls": 0, "run_updates": 0,
                "objective_unchanged": True, "reference_blend": 0, "task": {"kind": "observed_cell"},
                "receipts": [{"seq": i+1, "first_tick": i+1}], "demonstrations": [action]}
        path.write_text(json.dumps(task), encoding="utf-8")
    output = tmp_path / "dataset"
    manifest = export_surfaces(suite, output)
    assert manifest["episodes"] == {"train": 1, "validation": 1, "test": 1}
    _, splits = load_dataset(output)
    assert len({r["episode_id"] for rows in splits.values() for r in rows}) == 3
    task["receipts"][0]["first_tick"] = 0
    path.write_text(json.dumps(task), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt"):
        export_surfaces(suite, tmp_path / "invalid-dataset")


def test_native_session_grouping_keeps_neighboring_tasks_together_across_families(tmp_path):
    suite = tmp_path / 'suite'
    for i in range(10):
        session = i // 2
        path = suite / 'demonstrations' / f'trial-{i:04}' / 'motor' / 'task.json'
        path.parent.mkdir(parents=True)
        action = {'source': 'soh', 'controller': 'reference', 'first_tick': i+1,
                  'buttons': 0, 'successful_episode': True, 'reference_blend': 0,
                  'command_seq': i+1, 'observation_seq': i+10,
                  'features': [1, 0, 0, .5, 0, 0, 0, 0, 1], 'action': [-.5, .25]}
        instance = {'source': 'soh', 'instance_id': f'native-{session}'}
        task = {'success': True, 'source': 'soh', 'provider_calls': 0, 'run_updates': 0,
                'objective_unchanged': True, 'reference_blend': 0,
                'initial': instance, 'final': instance,
                'task': {'kind': 'observed_cell' if i % 2 else 'stairs_or_slope_up'},
                'receipts': [{'seq': i+1, 'first_tick': i+1}], 'demonstrations': [action]}
        path.write_text(json.dumps(task), encoding='utf-8')
    output = tmp_path / 'dataset'
    manifest = export_surfaces(suite, output, group_native_sessions=True)
    assert manifest['episode_grouping'] == 'native_instance'
    assert manifest['episodes'] == {'train': 3, 'validation': 1, 'test': 1}
    assert manifest['records'] == {'train': 6, 'validation': 2, 'test': 2}
    _, splits = load_dataset(output)
    assert len({r['episode_id'] for rows in splits.values() for r in rows}) == 5
