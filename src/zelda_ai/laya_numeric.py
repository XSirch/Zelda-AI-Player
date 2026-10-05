"""Numeric telemetry conditioning of the pretrained Laya choice head.

The frozen encoder represents the constant walking schema and physical options.
Only that constant representation is cached. Actual observed values enter a
trainable projection on every decision, never the cache or a handwritten stick.
"""

from __future__ import annotations

from .laya_data import FEATURE_NAMES, PROFILE, bounded_state, profile_features

TEXT_REPRESENTATION = "text_choice_v1"
NUMERIC_REPRESENTATION = "numeric_conditioned_choice_v2"
ENCODER_SCHEMA = {"profile": PROFILE, "features": list(FEATURE_NAMES), "values": "numeric_conditioning"}


def encoder_schema(profile):
    return {"profile": profile, "features": list(profile_features(profile)), "values": "numeric_conditioning"}


def numeric_values(rows):
    values = []
    for row in rows:
        state = row["state"]
        if state != bounded_state(state["history_oldest_first"], profile=state["profile"]):
            raise ValueError("Unexpected fields in numeric walking input")
        current = state["history_oldest_first"][-1]
        values.extend([current + [1.0, 0.0], current + [0.0, 1.0]])
    return values


def attach_numeric_conditioning(model):
    import torch

    if any(p.requires_grad for p in model.encoder.parameters()):
        raise ValueError("Numeric schema caching requires a frozen encoder")
    width = model.encoder.config.hidden_size
    adapter = torch.nn.Sequential(
        torch.nn.Linear(len(profile_features(getattr(model, "observation_profile", PROFILE))) + 2, 128),
        torch.nn.SiLU(), torch.nn.Linear(128, width)
    ).to("cuda")
    torch.nn.init.zeros_(adapter[-1].weight)
    torch.nn.init.zeros_(adapter[-1].bias)
    model.add_module("telemetry_adapter", adapter)
    model.numeric_telemetry = True
    model.numeric_cache = None


def numeric_forward(model, batch):
    import torch

    n = batch["input_ids"].shape[0]
    if n % 2 or n == 0:
        raise ValueError("Numeric decisions require exactly two axis questions per observation")
    ids, mask = batch["input_ids"][:2], batch["attention_mask"][:2]
    if not torch.equal(batch["input_ids"], ids.repeat(n // 2, 1)) or not torch.equal(
        batch["attention_mask"], mask.repeat(n // 2, 1)
    ):
        raise ValueError("Frozen encoding may cache only the constant schema, never game values")
    cache = model.numeric_cache
    if cache is None:
        with torch.inference_mode(False), torch.no_grad():
            encoded = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            model.numeric_cache = (ids.clone(), mask.clone(), encoded.detach())
    else:
        if not torch.equal(ids, cache[0]) or not torch.equal(mask, cache[1]):
            raise ValueError("Constant schema or physical options changed during inference")
    encoded = model.numeric_cache[2].repeat(n // 2, 1, 1)
    return numeric_logits(model, batch, encoded)


def numeric_logits(model, batch, encoded):
    """The same head arithmetic for training and fixed-schema inference.

    Callers own schema validation. This contains only tensor operations so a
    read-only worker can capture it without a host synchronization per layer.
    The telemetry projection still runs for every new observation.
    """
    import torch

    h = (
        encoded
        + model.type_emb(batch["qtype"])[:, None, :]
        + model.telemetry_adapter(batch["telemetry"])[:, None, :]
    )
    if model.head is not None:
        for layer in model.head.layers:
            h = layer(h, src_key_padding_mask=~batch["attention_mask"].bool())
    indices = batch["marker_pos"].clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    logits = model.scorer(torch.gather(h, 1, indices)).squeeze(-1).float()
    return logits.masked_fill(~batch["marker_mask"], -1e4)
