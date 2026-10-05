"""Fixed-schema CUDA execution for an exclusively owned, frozen numeric model.

Only physical questions and frozen schema encoding are prepared once. Every
call copies fresh bounded telemetry before replaying the actual trained head.
This is separate from the gradient-bearing training path; no CPU fallback.
"""

from __future__ import annotations

from .laya_data import bounded_state, profile_features
from .laya_numeric import numeric_logits, numeric_values


class NumericInference:
    def __init__(self, model, tokenizer, *, profile):
        import torch

        from .laya_training import batch_for, forward

        profile_features(profile)
        if (
            model.training
            or not getattr(model, "numeric_telemetry", False)
            or getattr(model, "observation_profile", None) != profile
            or any(p.requires_grad for p in model.encoder.parameters())
        ):
            raise ValueError("Numeric inference requires an eval model with a frozen matching schema")
        if not torch.cuda.is_available() or any(p.device.type != "cuda" for p in model.parameters()):
            raise RuntimeError("Numeric inference requires CUDA; no CPU fallback")
        self.profile = profile
        # The worker owns this model exclusively. Heads cannot train after capture.
        model.requires_grad_(False)
        with torch.inference_mode():
            state = bounded_state([[0.0] * len(profile_features(profile))], profile=profile)
            batch = batch_for(tokenizer, [{"state": state}], numeric=True)
            forward(model, batch)  # Strict schema/cache checks before any capture.
            encoded = model.numeric_cache[2]
            self.telemetry = batch["telemetry"]

            def score():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return numeric_logits(model, batch, encoded)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    score()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            # Default global capture guards remain active; failures abort startup.
            with torch.cuda.graph(self.graph, stream=stream):
                self.logits = score()
            torch.cuda.synchronize()
        # Retain all captured tensors/weights for their stable storage lifetime.
        self.model, self.batch, self.encoded = model, batch, encoded

    def __call__(self, state):
        import torch

        if state.get("profile") != self.profile:
            raise ValueError("Numeric inference observation profile changed")
        values = numeric_values([{"state": state}])  # Exact bounded fields, no labels.
        with torch.inference_mode():
            self.telemetry.copy_(torch.tensor(values, dtype=torch.float32, device="cuda"))
            self.graph.replay()
        # Borrowed GPU output, overwritten on the next replay. Decode now.
        return self.logits
