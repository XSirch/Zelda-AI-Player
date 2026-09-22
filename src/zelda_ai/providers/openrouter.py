from __future__ import annotations

import json
import math

import httpx

from ..models import Decision, ModelInfo, RunConfig, Usage
from .base import InferenceResult, ProviderFailure, SYSTEM_PROMPT

GATEWAY_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def model_info(row: dict) -> ModelInfo:
    reasoning = row.get("reasoning") or {}
    if "supported_efforts" not in reasoning:
        efforts = []
    elif reasoning["supported_efforts"] is None:
        efforts = GATEWAY_EFFORTS.copy()
    else:
        efforts = [value for value in reasoning["supported_efforts"] if value in GATEWAY_EFFORTS]
    if reasoning.get("mandatory"):
        efforts = [e for e in efforts if e != "none"]
    return ModelInfo(id=row["id"], name=row.get("name", row["id"]), efforts=efforts,
        effort_source="catalog" if efforts else "unsupported", default_effort=reasoning.get("default_effort"),
        structured_output="structured_outputs" in row.get("supported_parameters", []),
        pricing={k: str(v) for k, v in row.get("pricing", {}).items() if v is not None})


def parse_usage(data: dict) -> Usage:
    usage = data.get("usage") or {}
    return Usage(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
        cached_input_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        reasoning_output_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        cost_usd=usage.get("cost"), generation_id=data.get("id"), actual_model=data.get("model"),
        upstream_provider=data.get("provider"))


def reserve_cost(info: ModelInfo, prompt: str, output_limit: int) -> float:
    """Conservative preflight estimate, not an assertion of a provider-enforced hard cap."""
    try:
        prices = [float(info.pricing[k]) for k in ("prompt", "completion")]
        request = float(info.pricing.get("request", "0"))
        if any(not math.isfinite(p) or p < 0 for p in [*prices, request]):
            raise ValueError
    except (KeyError, ValueError, TypeError):
        raise ProviderFailure("Model has no usable pricing; a dollar budget cannot be checked.") from None
    # Byte count deliberately overestimates normal text tokenization. Schema and system are included.
    text_bytes = len((SYSTEM_PROMPT + prompt + json.dumps(Decision.model_json_schema())).encode("utf-8"))
    return (prices[0] * (text_bytes + 1024) + prices[1] * output_limit + request) * 1.25


class OpenRouterProvider:
    def __init__(self, api_key: str = "", transport=None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(base_url="https://openrouter.ai/api/v1/", timeout=180,
            transport=transport, follow_redirects=False)

    def headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "X-Title": "Zelda AI Player"}

    async def status(self) -> dict:
        if not self.api_key:
            return {"connected": False, "message": "Configure OPENROUTER_API_KEY ou conecte pelo painel."}
        try:
            response = await self.client.get("key", headers=self.headers())
            response.raise_for_status()
            data = response.json().get("data") or {}
            return {"connected": True, "usage": data.get("usage"), "limit": data.get("limit"),
                "limit_remaining": data.get("limit_remaining"), "message": "Chave validada."}
        except (httpx.HTTPError, ValueError):
            return {"connected": False, "message": "Falha ao validar a chave OpenRouter."}

    async def models(self) -> list[ModelInfo]:
        try:
            response = await self.client.get("models", headers=self.headers() if self.api_key else {})
            response.raise_for_status()
            return [model_info(row) for row in response.json()["data"]]
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise ProviderFailure(f"OpenRouter catalog unavailable ({type(exc).__name__}).") from None

    async def decide(self, config: RunConfig, prompt: str) -> InferenceResult:
        if not self.api_key:
            raise ProviderFailure("OpenRouter is not authenticated.")
        body = {"model": config.model, "messages": [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}], "stream": False,
            "max_tokens": config.max_output_tokens,
            "provider": {"require_parameters": True, "allow_fallbacks": False},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "zelda_decision", "strict": True, "schema": Decision.model_json_schema()}}}
        if config.effort:
            body["reasoning"] = {"effort": config.effort, "exclude": True}
        try:
            response = await self.client.post("chat/completions", json=body, headers=self.headers())
            response.raise_for_status()
            data = response.json()
            usage = parse_usage(data)
            choice = data.get("choices", [{}])[0]
            content = (choice.get("message") or {}).get("content")
            if choice.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
                raise ProviderFailure("No complete decision returned; usage was recorded. Try a larger output budget "
                    "or lower effort. No automatic paid retry was made.", usage)
            return InferenceResult(content, usage)
        except httpx.HTTPStatusError as exc:
            raise ProviderFailure(f"OpenRouter HTTP {exc.response.status_code}; no automatic retry.") from None
        except (httpx.RequestError, ValueError, KeyError) as exc:
            raise ProviderFailure(f"OpenRouter request failed ({type(exc).__name__}); billing may be incomplete.") from None

    async def close(self):
        await self.client.aclose()
