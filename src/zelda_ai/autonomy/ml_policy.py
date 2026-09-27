from __future__ import annotations

import threading
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.distributions import Bernoulli, Beta
except ImportError as exc:  # pragma: no cover - explicit runtime dependency message
    torch = None
    nn = None
    Bernoulli = None
    Beta = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from .features import BUTTON_NAMES, FEATURE_DIM


def require_torch():
    if torch is None:
        raise RuntimeError(
            "Autonomy V3 requires PyTorch for online ML/RL. Run 'uv sync' after updating "
            "the project so the torch dependency is installed."
        ) from _TORCH_IMPORT_ERROR


class HybridActorCritic(nn.Module if nn is not None else object):
    def __init__(self):
        require_torch()
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(FEATURE_DIM, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
        )
        self.stick_alpha = nn.Linear(256, 2)
        self.stick_beta = nn.Linear(256, 2)
        self.button_logits = nn.Linear(256, len(BUTTON_NAMES))
        self.value_head = nn.Linear(256, 1)
        nn.init.constant_(self.button_logits.bias, -2.2)

    def distributions(self, observations):
        latent = self.trunk(observations)
        alpha = torch.nn.functional.softplus(self.stick_alpha(latent)) + 1.05
        beta = torch.nn.functional.softplus(self.stick_beta(latent)) + 1.05
        stick_dist = Beta(alpha, beta)
        button_logits = self.button_logits(latent)
        button_dist = Bernoulli(logits=button_logits)
        value = self.value_head(latent).squeeze(-1)
        return stick_dist, button_dist, value, button_logits

    def evaluate(self, observations, stick_unit, buttons):
        stick_dist, button_dist, value, button_logits = self.distributions(observations)
        stick01 = ((stick_unit + 1.0) * 0.5).clamp(1e-5, 1.0 - 1e-5)
        log_prob = stick_dist.log_prob(stick01).sum(-1) + button_dist.log_prob(buttons).sum(-1)
        entropy = stick_dist.entropy().sum(-1) + button_dist.entropy().sum(-1)
        return log_prob, entropy, value, button_logits


class RNDNetwork(nn.Module if nn is not None else object):
    def __init__(self):
        require_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, 192),
            nn.SiLU(),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Linear(96, 48),
        )

    def forward(self, observations):
        return self.net(observations)


class OnlinePPO:
    """Asynchronous actor/learner pair.

    The actor is never optimized in-place. The learner trains a private copy and
    publishes weights only after an update, so gameplay inference keeps running
    while CUDA/CPU training is in progress.
    """

    def __init__(
        self,
        checkpoint: Path,
        *,
        learning_rate: float = 3e-4,
        rnd_learning_rate: float = 2e-4,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.18,
        entropy_coef: float = 0.012,
        value_coef: float = 0.5,
        epochs: int = 4,
        minibatch_size: int = 64,
    ):
        require_torch()
        self.checkpoint = Path(checkpoint)
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.learner_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Controller inference stays on CPU so CUDA backprop cannot starve the
        # 10 Hz action sampler. The networks are intentionally small.
        self.actor_device = torch.device("cpu")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.actor_lock = threading.Lock()

        self.learner = HybridActorCritic().to(self.learner_device)
        self.actor = HybridActorCritic().to(self.actor_device)
        self.learner_rnd = RNDNetwork().to(self.learner_device)
        self.actor_rnd = RNDNetwork().to(self.actor_device)
        self.learner_rnd_target = RNDNetwork().to(self.learner_device)
        self.actor_rnd_target = RNDNetwork().to(self.actor_device)
        for parameter in self.learner_rnd_target.parameters():
            parameter.requires_grad_(False)
        for parameter in self.actor_rnd_target.parameters():
            parameter.requires_grad_(False)

        self.actor.load_state_dict(self.learner.state_dict())
        self.actor_rnd.load_state_dict(self.learner_rnd.state_dict())
        self.actor_rnd_target.load_state_dict(self.learner_rnd_target.state_dict())

        self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=learning_rate)
        self.rnd_optimizer = torch.optim.AdamW(self.learner_rnd.parameters(), lr=rnd_learning_rate)
        self.updates = 0
        self.samples_trained = 0
        self.intrinsic_scale: float | None = None
        self.last_stats: dict = {}

        self.load_error = ""
        if self.checkpoint.is_file():
            try:
                self._load()
            except Exception as exc:
                # A truncated local checkpoint must not prevent autonomous play.
                # Preserve it for inspection and start a fresh learner.
                corrupt = self.checkpoint.with_suffix(
                    self.checkpoint.suffix + f".corrupt-{int(__import__('time').time())}"
                )
                try:
                    self.checkpoint.replace(corrupt)
                except OSError:
                    pass
                self.load_error = f"{type(exc).__name__}: {str(exc)[:180]}"
                self.actor.load_state_dict(self.learner.state_dict())
                self.actor_rnd.load_state_dict(self.learner_rnd.state_dict())
                self.actor_rnd_target.load_state_dict(self.learner_rnd_target.state_dict())

    def _actor_tensor(self, observation: list[float]):
        return torch.tensor(observation, dtype=torch.float32, device=self.actor_device)

    def sample(self, observation: list[float]) -> dict:
        with self.actor_lock, torch.inference_mode():
            obs = self._actor_tensor(observation).unsqueeze(0)
            stick_dist, button_dist, value, _ = self.actor.distributions(obs)
            sampled01 = stick_dist.sample()
            buttons = button_dist.sample()
            sampled = sampled01 * 2.0 - 1.0
            # Quantize first so PPO trains on the exact N64 stick values that
            # Bridge.send will deliver, not on an unobservable pre-rounding action.
            stick = torch.round(sampled * 80.0) / 80.0
            executed01 = ((stick + 1.0) * 0.5).clamp(1e-5, 1.0 - 1e-5)
            log_prob = (
                stick_dist.log_prob(executed01).sum(-1)
                + button_dist.log_prob(buttons).sum(-1)
            )
            return {
                "stick": [float(v) for v in stick.squeeze(0).detach().cpu().tolist()],
                "buttons": [float(v) for v in buttons.squeeze(0).detach().cpu().tolist()],
                "log_prob": float(log_prob.item()),
                "value": float(value.item()),
            }

    def actor_value(self, observation: list[float]) -> float:
        with self.actor_lock, torch.inference_mode():
            obs = self._actor_tensor(observation).unsqueeze(0)
            _, _, value, _ = self.actor.distributions(obs)
            return float(value.item())

    def intrinsic_reward(self, observation: list[float]) -> float:
        with self.actor_lock, torch.inference_mode():
            obs = self._actor_tensor(observation).unsqueeze(0)
            target = self.actor_rnd_target(obs)
            prediction = self.actor_rnd(obs)
            error = torch.nn.functional.mse_loss(prediction, target).item()
        if self.intrinsic_scale is None:
            self.intrinsic_scale = max(error, 1e-6)
        else:
            self.intrinsic_scale = 0.995 * self.intrinsic_scale + 0.005 * max(error, 1e-6)
        normalized = error / max(self.intrinsic_scale, 1e-6)
        return max(0.0, min(3.0, normalized)) / 3.0

    def train_rollout(
        self,
        rollout: list[dict],
        *,
        bootstrap_value: float,
        bootstrap_done: bool,
    ) -> dict:
        if not rollout:
            return {}

        observations = torch.tensor(
            [row["observation"] for row in rollout], dtype=torch.float32, device=self.learner_device
        )
        sticks = torch.tensor(
            [row["stick"] for row in rollout], dtype=torch.float32, device=self.learner_device
        )
        buttons = torch.tensor(
            [row["buttons"] for row in rollout], dtype=torch.float32, device=self.learner_device
        )
        old_log_probs = torch.tensor(
            [row["log_prob"] for row in rollout], dtype=torch.float32, device=self.learner_device
        )
        old_values = torch.tensor(
            [row["value"] for row in rollout], dtype=torch.float32, device=self.learner_device
        )
        rewards = torch.tensor(
            [row["reward"] for row in rollout], dtype=torch.float32, device=self.learner_device
        )
        dones = torch.tensor(
            [1.0 if row["done"] else 0.0 for row in rollout],
            dtype=torch.float32,
            device=self.learner_device,
        )

        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, device=self.learner_device)
        next_value = torch.tensor(
            0.0 if bootstrap_done else bootstrap_value,
            dtype=torch.float32,
            device=self.learner_device,
        )
        for index in range(len(rollout) - 1, -1, -1):
            nonterminal = 1.0 - dones[index]
            delta = rewards[index] + self.gamma * next_value * nonterminal - old_values[index]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[index] = gae
            next_value = old_values[index]
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

        count = len(rollout)
        last_policy_loss = last_value_loss = last_entropy = 0.0
        for _ in range(self.epochs):
            permutation = torch.randperm(count, device=self.learner_device)
            for start in range(0, count, self.minibatch_size):
                batch = permutation[start:start + self.minibatch_size]
                new_log_probs, entropy, values, button_logits = self.learner.evaluate(
                    observations[batch], sticks[batch], buttons[batch]
                )
                ratio = torch.exp(new_log_probs - old_log_probs[batch])
                unclipped = ratio * advantages[batch]
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                ) * advantages[batch]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.nn.functional.smooth_l1_loss(values, returns[batch])
                entropy_mean = entropy.mean()
                button_activity = torch.sigmoid(button_logits).mean()
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_mean
                    + 0.002 * button_activity
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.learner.parameters(), 1.0)
                self.optimizer.step()
                last_policy_loss = float(policy_loss.detach().item())
                last_value_loss = float(value_loss.detach().item())
                last_entropy = float(entropy_mean.detach().item())

        with torch.no_grad():
            rnd_target = self.learner_rnd_target(observations)
        rnd_prediction = self.learner_rnd(observations)
        rnd_loss = torch.nn.functional.mse_loss(rnd_prediction, rnd_target)
        self.rnd_optimizer.zero_grad(set_to_none=True)
        rnd_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.learner_rnd.parameters(), 1.0)
        self.rnd_optimizer.step()

        self.updates += 1
        self.samples_trained += count

        with self.actor_lock:
            self.actor.load_state_dict(self.learner.state_dict())
            self.actor_rnd.load_state_dict(self.learner_rnd.state_dict())

        self.last_stats = {
            "updates": self.updates,
            "samples_trained": self.samples_trained,
            "policy_loss": round(last_policy_loss, 6),
            "value_loss": round(last_value_loss, 6),
            "entropy": round(last_entropy, 6),
            "rnd_loss": round(float(rnd_loss.detach().item()), 6),
            "mean_reward": round(float(rewards.mean().item()), 6),
        }
        self.save()
        return dict(self.last_stats)

    def save(self):
        payload = {
            "version": 1,
            "feature_dim": FEATURE_DIM,
            "button_names": BUTTON_NAMES,
            "learner": self.learner.state_dict(),
            "actor": self.actor.state_dict(),
            "learner_rnd": self.learner_rnd.state_dict(),
            "actor_rnd": self.actor_rnd.state_dict(),
            "rnd_target": self.learner_rnd_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rnd_optimizer": self.rnd_optimizer.state_dict(),
            "updates": self.updates,
            "samples_trained": self.samples_trained,
        }
        temporary = self.checkpoint.with_suffix(self.checkpoint.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(self.checkpoint)

    def _load(self):
        payload = torch.load(
            self.checkpoint,
            map_location=self.learner_device,
            weights_only=False,
        )
        if payload.get("feature_dim") != FEATURE_DIM:
            raise RuntimeError(
                f"ML checkpoint feature dimension {payload.get('feature_dim')} does not match {FEATURE_DIM}"
            )
        if tuple(payload.get("button_names") or ()) != BUTTON_NAMES:
            raise RuntimeError("ML checkpoint controller button order does not match this build")
        self.learner.load_state_dict(payload["learner"])
        self.actor.load_state_dict(payload.get("actor") or payload["learner"])
        self.learner_rnd.load_state_dict(payload["learner_rnd"])
        self.actor_rnd.load_state_dict(payload.get("actor_rnd") or payload["learner_rnd"])
        self.learner_rnd_target.load_state_dict(payload["rnd_target"])
        self.actor_rnd_target.load_state_dict(payload["rnd_target"])
        if payload.get("optimizer"):
            self.optimizer.load_state_dict(payload["optimizer"])
        if payload.get("rnd_optimizer"):
            self.rnd_optimizer.load_state_dict(payload["rnd_optimizer"])
        self.updates = int(payload.get("updates") or 0)
        self.samples_trained = int(payload.get("samples_trained") or 0)

    def stats(self) -> dict:
        return {
            "learner_device": str(self.learner_device),
            "actor_device": str(self.actor_device),
            "updates": self.updates,
            "samples_trained": self.samples_trained,
            "checkpoint_load_error": self.load_error,
            **self.last_stats,
        }
