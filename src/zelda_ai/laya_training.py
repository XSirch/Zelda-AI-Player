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
    QUESTIONS,
    STICK_BINS,
    digest,
    export_surfaces,
    load_dataset,
    majority_baseline,
)

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


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_base(base: Path) -> dict:
    provenance = json.loads((base / "base-origin.json").read_text(encoding="utf-8"))
    if (provenance.get("repo") != BASE_REPO or provenance.get("revision") != BASE_REVISION
            or provenance.get("sha256") != BASE_FILES_SHA256):
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
    names = ("model.safetensors", "rl_agent_config.json", "encoder/config.json",
             "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")
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
    provenance = {"repo": BASE_REPO, "revision": BASE_REVISION, "training_domain": "generic_base",
                  "sha256": {name: digest(output / name) for name in names}}
    write_json(output / "base-origin.json", provenance)
    return verify_base(output)


def load_model(base: Path, candidate: Path | None = None):
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
    if (laya.__version__ != "0.3.5"
            or installed_origin.get("vcs_info", {}).get("commit_id") != LAYA_SOURCE_REVISION):
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
    if candidate:
        metadata = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
        if (metadata.get("profile") != PROFILE or metadata.get("base_sha256") != BASE_WEIGHTS_SHA256
                or metadata.get("source_revision") != LAYA_SOURCE_REVISION
                or digest(candidate / "heads.safetensors") != metadata.get("heads_sha256")):
            raise ValueError("Incompatible or changed Zelda candidate")
        expected = {k for k in model.state_dict() if not k.startswith("encoder.")}
        heads = load_file(str(candidate / "heads.safetensors"))
        if set(heads) != expected or any(not torch.isfinite(v).all() for v in heads.values()):
            raise ValueError("Invalid isolated candidate heads")
        weights = model.state_dict()
        weights.update(heads)
        model.load_state_dict(weights, strict=True)
    model.eval()
    return model, tokenizer


def batch_for(tokenizer, rows):
    from laya.common import QTYPES, build_sequence, collate_items

    groups = []
    for row in rows:
        group = []
        for axis, definition in QUESTIONS.items():
            q = {"t": definition["type"], "ins": definition["instructions"], "crit": definition["criteria"]}
            ids, markers = build_sequence(tokenizer, row["state"], q, 16384, HEAD_MAX_LEN)
            if len(ids) > MAX_LEN or len(markers) != len(STICK_BINS):
                raise ValueError("Input would truncate telemetry or action options")
            group.append({"ids": ids, "markers": markers, "qtype": QTYPES["choice"],
                          "label": STICK_BINS.index(row["labels"][0 if axis == "stick_x" else 1])})
        groups.append(group)
    batch = collate_items(groups, tokenizer.pad_token_id)
    return {k: v.to("cuda") for k, v in batch.items() if k != "meta"}


def forward(model, batch):
    import torch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(**{k: v for k, v in batch.items() if k != "label"})
    return logits


def evaluate(model, tokenizer, rows, *, latency_samples=20):
    import torch
    correct, joint, errors, count = 0, 0, 0, 0
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(rows), 2):
            part = rows[offset:offset+2]
            batch = batch_for(tokenizer, part)
            logits = forward(model, batch)
            chosen = logits.argmax(-1)
            matches = chosen == batch["label"]
            correct += int(matches.sum())
            joint += int(matches.reshape(-1, 2).all(-1).sum())
            bins = torch.tensor(STICK_BINS, device="cuda")
            actual = torch.tensor([r["executed_stick"] for r in part], device="cuda").reshape(-1)
            errors += float((bins[chosen] - actual).abs().sum())
            count += len(part)
        timings = []
        for i in range(latency_samples + 5):
            torch.cuda.synchronize()
            started = time.perf_counter()
            logits = forward(model, batch_for(tokenizer, [rows[i % len(rows)]]))
            logits.argmax(-1).cpu().tolist()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter()-started)*1000
            if i >= 5:
                timings.append(elapsed)
    timings.sort()
    return {"records": count, "axis_accuracy": correct/(2*count), "joint_accuracy": joint/count,
            "stick_mae_native_units": errors/(2*count), "latency_scope": "two_axes_tokenize_transfer_forward_readback",
            "latency_samples": len(timings), "p50_ms": statistics.median(timings),
            "p95_ms": timings[math.ceil(.95*len(timings))-1], "max_ms": max(timings),
            "physical_gameplay_evaluated": False}


def train(base: Path, dataset: Path, output: Path, *, steps=120, seed=4102050):
    import torch
    from safetensors.torch import save_file

    if output.exists() or not 1 <= steps <= 1000:
        raise ValueError("Use a new immutable candidate directory and 1..1000 pilot updates")
    manifest, splits = load_dataset(dataset)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model, tokenizer = load_model(base)
    before = evaluate(model, tokenizer, splits["validation"])
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
    model.train()
    model.encoder.eval()
    losses = []
    for step in range(steps):
        rows = rng.sample(splits["train"], min(2, len(splits["train"])))
        batch = batch_for(tokenizer, rows)
        optimizer.zero_grad(set_to_none=True)
        logits = forward(model, batch)
        loss = torch.nn.functional.cross_entropy(logits, batch["label"])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite pilot training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1) % 20 == 0:
            print(json.dumps({"update": step+1, "loss": losses[-1]}), flush=True)
    validation = evaluate(model, tokenizer, splits["validation"])
    test = evaluate(model, tokenizer, splits["test"])
    output.mkdir(parents=True)
    heads = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
             if not k.startswith("encoder.")}
    save_file(heads, str(output / "heads.safetensors"))
    metadata = {"profile": PROFILE, "base_repo": BASE_REPO, "base_revision": BASE_REVISION,
                "base_sha256": BASE_WEIGHTS_SHA256, "source_revision": LAYA_SOURCE_REVISION,
                "heads_sha256": digest(output / "heads.safetensors"), "steps": steps, "seed": seed,
                "trainable_parameters": sum(p.numel() for p in parameters), "encoder_frozen": True,
                "dataset_sha256": digest(dataset / "manifest.json"), "dataset": manifest,
                "scope": manifest["coverage"], "initial_loss": losses[0], "final_loss": losses[-1],
                "before_validation": before, "after_validation": validation, "reserved_test": test,
                "majority_baseline_validation": majority_baseline(splits["train"], splits["validation"]),
                "majority_baseline_test": majority_baseline(splits["train"], splits["test"]),
                "device": torch.cuda.get_device_name(0), "provider_calls": 0,
                "promotion": "disabled_pending_physical_gameplay_and_latency_gates",
                "max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN, "stick_bins": list(STICK_BINS)}
    write_json(output / "candidate.json", metadata)
    verify_base(base)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Isolated Laya base -> Zelda walking pilot; no provider calls")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-base")
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--weights-source", type=Path)
    export = commands.add_parser("export-surfaces")
    export.add_argument("suite", type=Path)
    export.add_argument("output", type=Path)
    training = commands.add_parser("train")
    training.add_argument("base", type=Path)
    training.add_argument("dataset", type=Path)
    training.add_argument("output", type=Path)
    training.add_argument("--steps", type=int, default=120)
    training.add_argument("--seed", type=int, default=4102050)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("base", type=Path)
    benchmark.add_argument("dataset", type=Path)
    benchmark.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    if args.command == "prepare-base":
        result = prepare_base(args.output, args.weights_source)
    elif args.command == "export-surfaces":
        result = export_surfaces(args.suite, args.output)
    elif args.command == "train":
        result = train(args.base, args.dataset, args.output, steps=args.steps, seed=args.seed)
    else:
        _, splits = load_dataset(args.dataset)
        model, tokenizer = load_model(args.base, args.candidate)
        result = evaluate(model, tokenizer, splits["test"], latency_samples=50)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
