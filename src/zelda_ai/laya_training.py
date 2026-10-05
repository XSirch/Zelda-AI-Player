"""Offline-only Laya base bootstrap, bounded specialization and latency audit.

Use the isolated optional environment. No provider, native input or trading
checkpoint is constructed. The base digest is pinned to the public checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import statistics
import time
import urllib.request
from pathlib import Path

from .laya_data import (
    PROFILE,
    STICK_BINS,
    digest,
    export_surfaces,
    load_dataset,
    majority_baseline,
    profile_features,
    profile_questions,
)
from .laya_numeric import NUMERIC_REPRESENTATION, TEXT_REPRESENTATION

BASE_REPO = "convaiinnovations/laya"
BASE_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
BASE_WEIGHTS_SHA256 = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
BASE_FILES_SHA256 = {
    "model.safetensors": BASE_WEIGHTS_SHA256,
    "rl_agent_config.json": "ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd",
    "encoder/config.json": "bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979",
    "tokenizer/tokenizer.json": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
    "tokenizer/tokenizer_config.json": "50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12",
}
LAYA_SOURCE_REVISION = "573e5b62696ba441230cd6be71d593331b5d23af"
MAX_LEN, HEAD_MAX_LEN = 512, 96
LOSS_MODES = ("choice", "stick_mse")


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def specialization_keys(keys, encoder_layers=0):
    """Save only the heads and an explicitly bounded encoder tail."""
    if type(encoder_layers) is not int or not 0 <= encoder_layers <= 4:
        raise ValueError("Encoder specialization must use zero to four tail layers")
    layers = sorted({int(k.split(".")[2]) for k in keys if k.startswith("encoder.layers.")})
    if encoder_layers > len(layers):
        raise ValueError("Requested encoder tail is absent from the pinned model")
    prefixes = tuple(f"encoder.layers.{i}." for i in layers[-encoder_layers:]) if encoder_layers else ()
    return {k for k in keys if not k.startswith("encoder.") or prefixes and k.startswith(prefixes)}


def configure_encoder_tail(model, encoder_layers):
    selected = specialization_keys(model.state_dict(), encoder_layers)
    model.encoder.requires_grad_(False)
    if encoder_layers:
        for layer in model.encoder.layers[-encoder_layers:]:
            layer.float()
        for name, parameter in model.named_parameters():
            if name in selected and name.startswith("encoder."):
                parameter.requires_grad_(True)
    return selected


def verify_base(base: Path) -> dict:
    provenance = json.loads((base / "base-origin.json").read_text(encoding="utf-8"))
    if (
        provenance.get("repo") != BASE_REPO
        or provenance.get("revision") != BASE_REVISION
        or provenance.get("sha256") != BASE_FILES_SHA256
    ):
        raise ValueError("Use the pinned generic Laya base, never a trading checkpoint")
    for name, expected in provenance["sha256"].items():
        if digest(base / name) != expected:
            raise ValueError(f"Base checksum mismatch: {name}")
    return provenance


def prepare_base(output: Path, weights_source: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError("Base preparation never overwrites a directory")
    if weights_source and digest(weights_source) != BASE_WEIGHTS_SHA256:
        raise ValueError("Source is not the published generic Laya base")
    output.mkdir(parents=True)
    names = (
        "model.safetensors",
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    )
    for name in names:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "model.safetensors" and weights_source:
            shutil.copyfile(weights_source, path)
        else:
            url = f"https://huggingface.co/{BASE_REPO}/resolve/{BASE_REVISION}/{name}"
            with urllib.request.urlopen(url, timeout=60) as source, path.open("wb") as dest:
                shutil.copyfileobj(source, dest)
    if digest(output / "model.safetensors") != BASE_WEIGHTS_SHA256:
        raise ValueError("Published weight digest mismatch")
    provenance = {
        "repo": BASE_REPO,
        "revision": BASE_REVISION,
        "training_domain": "generic_base",
        "sha256": {name: digest(output / name) for name in names},
    }
    write_json(output / "base-origin.json", provenance)
    return verify_base(output)


def load_model(base: Path, candidate: Path | None = None, *, encoder_layers=0, numeric=False, profile=PROFILE):
    profile_features(profile)
    verify_base(base)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from importlib.metadata import distribution

    import laya
    import torch
    from laya.common import build_model
    from safetensors.torch import load_file
    from transformers import PreTrainedTokenizerFast

    installed_origin = json.loads(distribution("laya").read_text("direct_url.json") or "{}")
    if (
        laya.__version__ != "0.3.5"
        or installed_origin.get("vcs_info", {}).get("commit_id") != LAYA_SOURCE_REVISION
    ):
        raise RuntimeError("Use the pinned Laya 0.3.5 source revision")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pilot; no silent CPU fallback")
    cfg = json.loads((base / "rl_agent_config.json").read_text(encoding="utf-8"))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(base / "tokenizer"), local_files_only=True)
    model = build_model(cfg, encoder_dir=str(base / "encoder"))
    model.encoder.config.reference_compile = False
    model.load_state_dict(load_file(str(base / "model.safetensors")), strict=True)
    model.encoder.requires_grad_(False)
    model.to("cuda")
    model.encoder.to(dtype=torch.bfloat16)
    model.stick_readout = "argmax"
    model.observation_profile = profile
    if candidate:
        metadata = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
        if (
            metadata.get("profile") != profile
            or metadata.get("base_sha256") != BASE_WEIGHTS_SHA256
            or metadata.get("source_revision") != LAYA_SOURCE_REVISION
            or digest(candidate / "heads.safetensors") != metadata.get("heads_sha256")
        ):
            raise ValueError("Incompatible or changed Zelda candidate")
        encoder_layers = metadata.get("encoder_layers", 0)
        if metadata.get("encoder_frozen") is not (encoder_layers == 0):
            raise ValueError("Encoder specialization metadata contradicts the saved scope")
        representation = metadata.get("representation", TEXT_REPRESENTATION)
        if metadata.get("loss_mode", "choice") not in LOSS_MODES:
            raise ValueError("Unknown supervised walking loss")
        model.stick_readout = metadata.get("stick_readout", "argmax")
        if model.stick_readout not in {"argmax", "expectation"}:
            raise ValueError("Unknown physical stick decoder")
        if representation not in {TEXT_REPRESENTATION, NUMERIC_REPRESENTATION}:
            raise ValueError("Unknown Zelda telemetry representation")
        numeric = representation == NUMERIC_REPRESENTATION
        if numeric and encoder_layers:
            raise ValueError("Numeric schema caching requires an immutable encoder")
        if numeric:
            from .laya_numeric import attach_numeric_conditioning

            attach_numeric_conditioning(model)
        expected = configure_encoder_tail(model, encoder_layers)
        heads = load_file(str(candidate / "heads.safetensors"))
        if set(heads) != expected or any(not torch.isfinite(v).all() for v in heads.values()):
            raise ValueError("Invalid isolated candidate heads")
        weights = model.state_dict()
        weights.update(heads)
        model.load_state_dict(weights, strict=True)
    else:
        configure_encoder_tail(model, encoder_layers)
        if numeric:
            from .laya_numeric import attach_numeric_conditioning

            attach_numeric_conditioning(model)
    model.eval()
    return model, tokenizer


def batch_for(tokenizer, rows, *, numeric=False):
    from laya.common import QTYPES, build_sequence, collate_items

    groups = []
    for row in rows:
        group = []
        for axis, definition in profile_questions(row["state"]["profile"]).items():
            q = {"t": definition["type"], "ins": definition["instructions"], "crit": definition["criteria"]}
            if numeric:
                from .laya_numeric import encoder_schema

                state = encoder_schema(row["state"]["profile"])
            else:
                state = row["state"]
            ids, markers = build_sequence(tokenizer, state, q, 16384, HEAD_MAX_LEN)
            if len(ids) > MAX_LEN or len(markers) != len(STICK_BINS):
                raise ValueError("Input would truncate telemetry or action options")
            item = {"ids": ids, "markers": markers, "qtype": QTYPES["choice"]}
            if "labels" in row:
                item["label"] = STICK_BINS.index(row["labels"][0 if axis == "stick_x" else 1])
            group.append(item)
        groups.append(group)
    batch = collate_items(groups, tokenizer.pad_token_id)
    batch = {k: v.to("cuda") for k, v in batch.items() if k != "meta"}
    if numeric:
        import torch

        from .laya_numeric import numeric_values

        batch["telemetry"] = torch.tensor(numeric_values(rows), dtype=torch.float32, device="cuda")
    return batch


def forward(model, batch):
    import torch

    with torch.autocast("cuda", dtype=torch.bfloat16):
        if getattr(model, "numeric_telemetry", False):
            from .laya_numeric import numeric_forward

            logits = numeric_forward(model, batch)
        else:
            logits, _ = model(**{k: v for k, v in batch.items() if k != "label"})
    return logits


def decode_stick(logits, readout="argmax"):
    import torch

    if logits.shape[-1] != len(STICK_BINS) or not torch.isfinite(logits).all():
        raise ValueError("Invalid physical stick logits")
    bins = torch.tensor(STICK_BINS, device=logits.device, dtype=torch.float32)
    if readout == "argmax":
        return bins[logits.argmax(-1)].long()
    if readout == "expectation":
        return torch.round(torch.softmax(logits.float(), -1) @ bins).clamp(-80, 80).long()
    raise ValueError("Unknown physical stick decoder")


def supervised_stick_loss(logits, labels, executed_stick, *, mode="choice"):
    """Train choices or their continuous mean against consumed native actions.

    Regression stays differentiable through probabilities. Rounded decoder
    output must never be used for gradients, nor quantized labels as its target.
    This supervised executed-action objective is separate from PPO residuals.
    """
    import torch

    if mode not in LOSS_MODES:
        raise ValueError("Unknown supervised walking loss")
    if (
        logits.ndim != 2
        or logits.shape[1] != len(STICK_BINS)
        or logits.shape[0] == 0
        or logits.shape[0] % 2
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("Invalid two-axis physical stick logits")
    if mode == "choice":
        return torch.nn.functional.cross_entropy(logits, labels)
    target = torch.as_tensor(executed_stick, dtype=torch.float32, device=logits.device)
    if (
        target.ndim != 2
        or target.shape != (logits.shape[0] // 2, 2)
        or not torch.isfinite(target).all()
        or (target.abs() > 80).any()
    ):
        raise ValueError("Invalid or misaligned consumed native actions")
    bins = torch.tensor(STICK_BINS, dtype=torch.float32, device=logits.device)
    predicted = torch.softmax(logits.float(), -1) @ (bins / 80)
    return torch.nn.functional.mse_loss(predicted, target.reshape(-1) / 80)


def derive_readout(base: Path, source: Path, output: Path, readout: str):
    if output.exists() or readout not in {"argmax", "expectation"}:
        raise ValueError("Use a new immutable output and a supported decoder")
    metadata = json.loads((source / "candidate.json").read_text(encoding="utf-8"))
    load_model(base, source, profile=metadata.get("profile"))  # Actual keys, provenance and finite weights.
    metadata.update(
        stick_readout=readout,
        derived_from_candidate_sha256=digest(source / "candidate.json"),
        derivation_training_updates=0,
        derivation="decoder_only_weights_unchanged",
    )
    metadata["readout_evaluation"] = "pending_separate_benchmark_and_native_trials"
    output.mkdir(parents=True)
    shutil.copyfile(source / "heads.safetensors", output / "heads.safetensors")
    write_json(output / "candidate.json", metadata)
    return metadata


def evaluate(model, tokenizer, rows, *, latency_samples=20):
    import torch

    correct, joint, errors, count = 0, 0, 0, 0
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(rows), 2):
            part = rows[offset : offset + 2]
            batch = batch_for(tokenizer, part, numeric=getattr(model, "numeric_telemetry", False))
            logits = forward(model, batch)
            chosen = logits.argmax(-1)
            matches = chosen == batch["label"]
            correct += int(matches.sum())
            joint += int(matches.reshape(-1, 2).all(-1).sum())
            decoded = decode_stick(logits, model.stick_readout)
            actual = torch.tensor([r["executed_stick"] for r in part], device="cuda").reshape(-1)
            errors += float((decoded - actual).abs().sum())
            count += len(part)
        timings = []
        for i in range(latency_samples + 5):
            torch.cuda.synchronize()
            started = time.perf_counter()
            logits = forward(
                model,
                batch_for(
                    tokenizer, [rows[i % len(rows)]], numeric=getattr(model, "numeric_telemetry", False)
                ),
            )
            decode_stick(logits, model.stick_readout).cpu().tolist()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            if i >= 5:
                timings.append(elapsed)
    timings.sort()
    return {
        "records": count,
        "axis_accuracy": correct / (2 * count),
        "joint_accuracy": joint / count,
        "stick_mae_native_units": errors / (2 * count),
        "latency_scope": "two_axes_tokenize_transfer_forward_readback",
        "latency_samples": len(timings),
        "p50_ms": statistics.median(timings),
        "p95_ms": timings[math.ceil(0.95 * len(timings)) - 1],
        "max_ms": max(timings),
        "physical_gameplay_evaluated": False,
        "stick_readout": model.stick_readout,
        "accuracy_scope": "choice_argmax_labels",
    }


def train(
    base: Path,
    dataset: Path,
    output: Path,
    *,
    steps=120,
    seed=4102050,
    encoder_layers=0,
    numeric=False,
    batch_size=2,
    loss_mode="choice",
    baseline_output: Path | None = None,
):
    import torch
    from safetensors.torch import save_file

    if output.exists() or not 1 <= steps <= 1000:
        raise ValueError("Use a new immutable candidate directory and 1..1000 pilot updates")
    if not 1 <= batch_size <= 16:
        raise ValueError("Use a bounded training batch of one to sixteen real observations")
    if loss_mode not in LOSS_MODES:
        raise ValueError("Unknown supervised walking loss")
    manifest, splits = load_dataset(dataset)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model, tokenizer = load_model(base, encoder_layers=encoder_layers, numeric=numeric, profile=manifest["profile"])
    if loss_mode == "stick_mse":
        model.stick_readout = "expectation"
    before = evaluate(model, tokenizer, splits["validation"])
    parameters = [p for p in model.parameters() if p.requires_grad]
    if baseline_output is not None:
        if baseline_output.exists() or baseline_output.resolve() == output.resolve():
            raise ValueError("The untouched baseline needs a separate new immutable directory")
        baseline_output.mkdir(parents=True)
        selected = specialization_keys(model.state_dict(), encoder_layers)
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in selected},
                  str(baseline_output / "heads.safetensors"))
        write_json(baseline_output / "candidate.json", {
            "profile": manifest["profile"], "base_repo": BASE_REPO, "base_revision": BASE_REVISION,
            "base_sha256": BASE_WEIGHTS_SHA256, "source_revision": LAYA_SOURCE_REVISION,
            "heads_sha256": digest(baseline_output / "heads.safetensors"), "steps": 0, "seed": seed,
            "loss_mode": loss_mode, "stick_readout": model.stick_readout,
            "encoder_layers": encoder_layers, "encoder_frozen": encoder_layers == 0,
            "representation": NUMERIC_REPRESENTATION if numeric else TEXT_REPRESENTATION,
            "dataset_sha256": digest(dataset / "manifest.json"), "provider_calls": 0,
            "promotion": "disabled", "scope": "exact_pre_training_baseline_for_separate_physical_evaluation"})
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.01)
    model.train()
    model.encoder.eval()
    losses = []
    for step in range(steps):
        rows = rng.sample(splits["train"], min(batch_size, len(splits["train"])))
        batch = batch_for(tokenizer, rows, numeric=numeric)
        optimizer.zero_grad(set_to_none=True)
        logits = forward(model, batch)
        loss = supervised_stick_loss(
            logits, batch["label"], [r["executed_stick"] for r in rows], mode=loss_mode
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite pilot training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 20 == 0:
            print(json.dumps({"update": step + 1, "loss": losses[-1]}), flush=True)
    validation = evaluate(model, tokenizer, splits["validation"])
    test = evaluate(model, tokenizer, splits["test"])
    output.mkdir(parents=True)
    selected = specialization_keys(model.state_dict(), encoder_layers)
    heads = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in selected}
    save_file(heads, str(output / "heads.safetensors"))
    metadata = {
        "profile": manifest["profile"],
        "base_repo": BASE_REPO,
        "base_revision": BASE_REVISION,
        "base_sha256": BASE_WEIGHTS_SHA256,
        "source_revision": LAYA_SOURCE_REVISION,
        "heads_sha256": digest(output / "heads.safetensors"),
        "steps": steps,
        "seed": seed,
        "batch_size": batch_size,
        "loss_mode": loss_mode,
        "stick_readout": model.stick_readout,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "encoder_frozen": encoder_layers == 0,
        "encoder_layers": encoder_layers,
        "representation": NUMERIC_REPRESENTATION if numeric else TEXT_REPRESENTATION,
        "numeric_history_usage": "current_frame_only" if numeric else None,
        "dataset_sha256": digest(dataset / "manifest.json"),
        "dataset": manifest,
        "scope": manifest["coverage"],
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "before_validation": before,
        "before_candidate_sha256": digest(baseline_output / "candidate.json") if baseline_output else None,
        "after_validation": validation,
        "reserved_test": test,
        "majority_baseline_validation": majority_baseline(splits["train"], splits["validation"]),
        "majority_baseline_test": majority_baseline(splits["train"], splits["test"]),
        "device": torch.cuda.get_device_name(0),
        "provider_calls": 0,
        "promotion": "disabled_pending_physical_gameplay_and_latency_gates",
        "max_len": MAX_LEN,
        "head_max_len": HEAD_MAX_LEN,
        "stick_bins": list(STICK_BINS),
    }
    write_json(output / "candidate.json", metadata)
    verify_base(base)
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Isolated Laya base -> Zelda walking pilot; no provider calls"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-base")
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--weights-source", type=Path)
    export = commands.add_parser("export-surfaces")
    export.add_argument("suite", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--group-native-sessions", action="store_true")
    training = commands.add_parser("train")
    training.add_argument("base", type=Path)
    training.add_argument("dataset", type=Path)
    training.add_argument("output", type=Path)
    training.add_argument("--steps", type=int, default=120)
    training.add_argument("--seed", type=int, default=4102050)
    training.add_argument("--encoder-layers", type=int, default=0, choices=range(5))
    training.add_argument("--numeric-telemetry", action="store_true")
    training.add_argument("--batch-size", type=int, default=2, choices=range(1, 17))
    training.add_argument("--loss-mode", choices=LOSS_MODES, default="choice")
    training.add_argument("--baseline-output", type=Path)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("base", type=Path)
    benchmark.add_argument("dataset", type=Path)
    benchmark.add_argument("--candidate", type=Path)
    decoder = commands.add_parser("derive-readout")
    decoder.add_argument("base", type=Path)
    decoder.add_argument("source", type=Path)
    decoder.add_argument("output", type=Path)
    decoder.add_argument("--readout", choices=("argmax", "expectation"), required=True)
    args = parser.parse_args()
    if args.command == "prepare-base":
        result = prepare_base(args.output, args.weights_source)
    elif args.command == "export-surfaces":
        result = export_surfaces(args.suite, args.output, group_native_sessions=args.group_native_sessions)
    elif args.command == "train":
        result = train(
            args.base,
            args.dataset,
            args.output,
            steps=args.steps,
            seed=args.seed,
            encoder_layers=args.encoder_layers,
            numeric=args.numeric_telemetry,
            batch_size=args.batch_size,
            loss_mode=args.loss_mode,
            baseline_output=args.baseline_output,
        )
    elif args.command == "derive-readout":
        result = derive_readout(args.base, args.source, args.output, args.readout)
    else:
        manifest, splits = load_dataset(args.dataset)
        model, tokenizer = load_model(args.base, args.candidate, profile=manifest["profile"])
        result = evaluate(model, tokenizer, splits["test"], latency_samples=50)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
