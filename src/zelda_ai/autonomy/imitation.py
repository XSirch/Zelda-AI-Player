"""Separate off-policy imitation candidate for observed walking surfaces.

No PPO probabilities or mixed residual actions are reinterpreted here. Candidate
evaluation owns raw analog output with zero reference blending and zero online
updates. It cannot control buttons, ladders, aiming, inventory or the campaign.
"""
from __future__ import annotations

import math
from pathlib import Path

from .features import camera_world_yaw
from .ml_policy import nn, require_torch, torch

VERSION = "surface-imitation-v1"
FEATURES = ("heading_sin", "heading_cos", "mirrored", "horizontal_distance",
            "target_height", "speed", "surface_up", "surface_down", "cell")
ACTION_CONTRACT = "executed_n64_stick_div_80_no_buttons_no_reference_blend"


def encode_surface(game, task):
    point, position = task.steering_point(game), game.player.position
    dx, dz = point[0] - position[0], point[2] - position[2]
    angle = math.atan2(dx, dz) - (camera_world_yaw(game, game.player) or 0)
    return [math.sin(angle), math.cos(angle), float(game.mirrored_world),
            min(1., math.hypot(dx, dz) / 200), max(-1., min(1., (point[1] - position[1]) / 100)),
            max(0., min(1., game.player.speed_xz / 10)),
            float(task.kind == "stairs_or_slope_up"), float(task.kind == "stairs_or_slope_down"),
            # Dry approach points share this steering contract. Their physical
            # transition/landing postcondition belongs to the local task.
            float(task.kind in {"observed_cell", "observed_portal", "observed_descent_approach",
                               "observed_container_approach"})]


class SurfacePolicy:
    def __init__(self, *, seed=1):
        require_torch()
        # Avoid changing the PPO/random runtime generator when creating a local
        # candidate. Both before/after evaluation use the same initialization.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.net = nn.Sequential(nn.Linear(len(FEATURES), 64), nn.Tanh(),
                                     nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2), nn.Tanh())
        self.net.eval()
        self.seed, self.training_steps = seed, 0

    def __call__(self, game, task):
        with torch.inference_mode():
            stick = self.net(torch.tensor(encode_surface(game, task), dtype=torch.float32)).tolist()
        return tuple(round(max(-1., min(1., value)) * 80) for value in stick)

    def fit(self, rows, *, epochs=120):
        if not rows or not 1 <= epochs <= 1000:
            raise ValueError("Bounded, consumed real-game demonstrations are required")
        for row in rows:
            if (row.get("source") != "soh" or row.get("controller") != "reference"
                    or not row.get("successful_episode") or row.get("first_tick", 0) <= 0
                    or row.get("reference_blend") != 0 or row.get("buttons") != 0
                    or len(row.get("features", [])) != len(FEATURES)
                    or len(row.get("action", [])) != 2
                    or not all(math.isfinite(v) for v in row["features"] + row["action"])
                    or any(abs(v) > 1 for v in row["action"])):
                raise ValueError("Invalid or unconsumed imitation demonstration")
        observations = torch.tensor([r["features"] for r in rows], dtype=torch.float32)
        actions = torch.tensor([r["action"] for r in rows], dtype=torch.float32)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=.003)
        self.net.train()
        losses = []
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(self.net(observations), actions)
            if not torch.isfinite(loss):
                raise ValueError("Imitation training diverged")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        self.training_steps += epochs
        self.net.eval()
        return {"training_steps": epochs, "demonstrations": len(rows),
                "initial_mse": losses[0], "final_mse": losses[-1]}

    def save(self, path: Path, *, dataset_sha256):
        if path.exists():
            raise FileExistsError("Candidate checkpoints are immutable")
        torch.save({"version": VERSION, "features": FEATURES, "action_contract": ACTION_CONTRACT,
                    "normalizer": "fixed_bounded_features_v1", "seed": self.seed,
                    "training_steps": self.training_steps, "dataset_sha256": dataset_sha256,
                    "state_dict": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path):
        require_torch()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint.get("version") != VERSION or tuple(checkpoint.get("features", ())) != FEATURES
                or checkpoint.get("action_contract") != ACTION_CONTRACT
                or checkpoint.get("normalizer") != "fixed_bounded_features_v1"):
            raise ValueError("Incompatible local candidate checkpoint")
        policy = cls(seed=checkpoint["seed"])
        policy.net.load_state_dict(checkpoint["state_dict"], strict=True)
        if any(not torch.isfinite(value).all() for value in policy.net.state_dict().values()):
            raise ValueError("Nonfinite local candidate checkpoint")
        policy.training_steps = checkpoint["training_steps"]
        return policy
