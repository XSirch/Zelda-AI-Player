"""Read-only local inference process. No bridge, provider or game input access."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .laya_data import PROFILE, PROFILE_FEATURES, bounded_state, digest, profile_features
from .laya_training import batch_for, decode_stick, forward, load_model


def reply(value):
    print(json.dumps(value, allow_nan=False, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILE_FEATURES), default=PROFILE)
    args = parser.parse_args()
    model, tokenizer = load_model(args.base, args.candidate, profile=args.profile)
    import torch

    with torch.inference_mode():
        warmup = {"state": bounded_state([[0.0] * len(profile_features(args.profile))], profile=args.profile)}
        for _ in range(3):
            forward(
                model, batch_for(tokenizer, [warmup], numeric=getattr(model, "numeric_telemetry", False))
            ).argmax(-1).cpu().tolist()
        reply(
            {
                "ready": True,
                "profile": args.profile,
                "candidate_sha256": digest(args.candidate / "heads.safetensors"),
                "provider_calls": 0,
                "training_updates": 0,
            }
        )
        for line in sys.stdin:
            if len(line) > 8192:
                raise ValueError("Oversized inference request")
            request = json.loads(line)
            state = request["state"]
            if (
                set(request) != {"id", "state"}
                or type(request["id"]) is not int
                or request["id"] <= 0
                or state != bounded_state(state["history_oldest_first"], profile=args.profile)
            ):
                raise ValueError("Invalid bounded inference request")
            torch.cuda.synchronize()
            started = time.monotonic()
            logits = forward(
                model,
                batch_for(tokenizer, [{"state": state}], numeric=getattr(model, "numeric_telemetry", False)),
            )
            stick = decode_stick(logits, model.stick_readout).cpu().tolist()
            reply(
                {
                    "id": request["id"],
                    "stick": stick,
                    "inference_ms": (time.monotonic() - started) * 1000,
                }
            )


if __name__ == "__main__":
    main()
