"""Provider router -- distributes reasoning calls across free LLM APIs.

Strategy:
1. Maintain a priority queue of available providers
2. Route each request to the highest-priority available provider
3. On rate limit or error, demote the provider and try next
4. Track success rates and latency per provider for adaptive routing
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ProviderStats:
    """Track provider performance for adaptive routing."""

    name: str
    total_calls: int = 0
    successes: int = 0
    failures: int = 0
    rate_limits: int = 0
    avg_latency_ms: float = 0.0
    last_rate_limit: float = 0.0
    cooldown_until: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 1.0
        return self.successes / self.total_calls

    @property
    def is_available(self) -> bool:
        return time.time() > self.cooldown_until

    def record_success(self, latency_ms: float):
        self.total_calls += 1
        self.successes += 1
        self.avg_latency_ms = (self.avg_latency_ms * 0.9) + (latency_ms * 0.1)

    def record_failure(self):
        self.total_calls += 1
        self.failures += 1

    def record_rate_limit(self, cooldown_seconds: float = 60.0):
        self.total_calls += 1
        self.rate_limits += 1
        self.last_rate_limit = time.time()
        self.cooldown_until = time.time() + cooldown_seconds


# Provider configurations: base_url, model, auth header format
PROVIDER_CONFIGS = {
    "google": {
        "env_key": "GOOGLE_AI_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-2.0-flash",
        "style": "google",
        "daily_limit": 1500,
        "tier": "worker",
    },
    "groq": {
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "style": "openai",
        "daily_limit": 1000,
        "tier": "worker",
    },
    "openrouter": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "style": "openai",
        "daily_limit": 200,
        "tier": "worker",
    },
    "cerebras": {
        "env_key": "CEREBRAS_API_KEY",
        "base_url": "https://api.cerebras.ai/v1",
        "model": "llama3.3-70b",
        "style": "openai",
        "daily_limit": 43200,
        "tier": "worker",
    },
    "mistral": {
        "env_key": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-small-latest",
        "style": "openai",
        "daily_limit": 86400,
        "tier": "worker",
    },
    "together": {
        "env_key": "TOGETHER_API_KEY",
        "base_url": "https://api.together.xyz/v1",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "style": "openai",
        "daily_limit": 1000,
        "tier": "worker",
    },
    "nvidia": {
        "env_key": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "meta/llama-3.1-8b-instruct",
        "style": "openai",
        "daily_limit": 500,
        "tier": "worker",
    },
    "github": {
        "env_key": "GITHUB_TOKEN",
        "base_url": "https://models.inference.ai.azure.com",
        "model": "gpt-4o-mini",
        "style": "openai",
        "daily_limit": 200,
        "tier": "worker",
    },
    "cloudflare": {
        "env_key": "CLOUDFLARE_API_TOKEN",
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "model": "@cf/meta/llama-3.1-8b-instruct",
        "style": "cloudflare",
        "daily_limit": 500,
        "tier": "worker",
    },
    "cohere": {
        "env_key": "COHERE_API_KEY",
        "base_url": "https://api.cohere.ai/v1",
        "model": "command-r",
        "style": "cohere",
        "daily_limit": 33,
        "tier": "worker",
    },
    "hf_inference": {
        "env_key": "HF_TOKEN",
        "base_url": "https://api-inference.huggingface.co/models",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "style": "hf",
        "daily_limit": 100,
        "tier": "worker",
    },
    "glm_flash": {
        "env_key": "GLM_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "style": "openai",
        "daily_limit": 1000,
        "tier": "worker",
    },
    "glm_5_1": {
        "env_key": "ZAI_API_KEY",
        "base_url": "https://api.z.ai/v1",
        "model": "GLM-5.1",
        "style": "openai",
        "daily_limit": 500,
        "tier": "brain",
    },
}

# Tier definitions for intelligent routing
TIER_WORKER = "worker"   # free providers -- handle basic reasoning
TIER_BRAIN = "brain"     # GLM-5.1 -- handles synthesis, decomposition, complex tasks


class ProviderRouter:
    """Routes reasoning requests across multiple free-tier LLM providers.

    Automatically handles:
    - Provider selection based on availability and performance
    - Rate limit detection and cooldown
    - Failover to next available provider
    - Usage tracking to stay within free-tier limits
    """

    def __init__(self, api_keys: dict[str, str]):
        self._client = httpx.AsyncClient(timeout=30.0)
        self._stats: dict[str, ProviderStats] = {}
        self._providers: dict[str, dict] = {}
        self._daily_usage: dict[str, int] = {}

        # Register available providers based on supplied keys
        for name, config in PROVIDER_CONFIGS.items():
            env_key = config["env_key"]
            if env_key in api_keys:
                self._providers[name] = {**config, "api_key": api_keys[env_key]}
                self._stats[name] = ProviderStats(name=name)
                self._daily_usage[name] = 0
                logger.info("Registered provider: %s (%s)", name, config["model"])

        if not self._providers:
            logger.warning("No LLM providers configured. Reasoning will fail.")

    def _select_provider(self, tier: str | None = None) -> str | None:
        """Select the best available provider, optionally filtered by tier.

        Args:
            tier: If set, only consider providers of this tier.
                  None means consider all providers.
        """
        candidates = []
        for name, stats in self._stats.items():
            if not stats.is_available:
                continue
            config = self._providers[name]
            if tier and config.get("tier", TIER_WORKER) != tier:
                continue
            if self._daily_usage.get(name, 0) >= config["daily_limit"]:
                continue
            # Score: success_rate / (latency_ms + 1) -- prefer fast and reliable
            score = stats.success_rate / (stats.avg_latency_ms + 1)
            candidates.append((score, name))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        return candidates[0][1]

    def has_brain_tier(self) -> bool:
        """Check if any brain-tier provider (GLM-5.1) is configured."""
        return any(
            c.get("tier") == TIER_BRAIN
            for c in self._providers.values()
        )

    async def reason_brain(self, prompt: str, system_prompt: str = "") -> dict[str, Any]:
        """Route to brain-tier provider (GLM-5.1) for complex reasoning.

        Falls back to any available worker-tier provider if brain is
        unavailable, rate-limited, or erroring.
        """
        try:
            return await self.reason(prompt, system_prompt=system_prompt, tier=TIER_BRAIN)
        except RuntimeError:
            logger.info("Brain tier unavailable, falling back to worker tier")
            return await self.reason(prompt, system_prompt=system_prompt, tier=TIER_WORKER)

    async def reason(self, prompt: str, system_prompt: str = "", tier: str | None = None) -> dict[str, Any]:
        """Send a reasoning request to the best available provider.

        Args:
            prompt: The user/task prompt.
            system_prompt: The cognitive protocol system prompt. If empty,
                uses a generic default.
            tier: If set, restrict to providers of this tier only.
                  If no provider is available in the requested tier,
                  raises RuntimeError (caller can catch and retry
                  with a different tier).

        Returns dict with keys: text, model, provider, latency_ms, confidence, tier
        """
        system = system_prompt or (
            "You are a reasoning agent in a distributed swarm. "
            "Be concise, precise, and quantitative. State your confidence level."
        )

        tried: set[str] = set()
        max_attempts = len(self._providers)

        while len(tried) < max_attempts:
            provider_name = self._select_provider(tier=tier)
            if not provider_name:
                break

            if provider_name in tried:
                break

            tried.add(provider_name)
            config = self._providers[provider_name]
            stats = self._stats[provider_name]

            try:
                start = time.time()
                text = await self._call_provider(provider_name, config, prompt, system)
                latency = (time.time() - start) * 1000

                stats.record_success(latency)
                self._daily_usage[provider_name] = self._daily_usage.get(provider_name, 0) + 1

                return {
                    "text": text,
                    "model": config["model"],
                    "provider": provider_name,
                    "latency_ms": round(latency, 1),
                    "confidence": 0.5,  # base confidence, refined by synthesis
                    "tier": config.get("tier", TIER_WORKER),
                }

            except RateLimitError:
                stats.record_rate_limit(cooldown_seconds=60)
                logger.info("Rate limited by %s, cooling down", provider_name)

            except Exception as e:
                stats.record_failure()
                logger.warning("Provider %s error: %s", provider_name, e)

        raise RuntimeError(
            f"All providers exhausted for tier={tier or 'any'} "
            f"(tried {len(tried)} of {max_attempts})"
        )

    async def _call_provider(self, name: str, config: dict, prompt: str, system: str) -> str:
        """Make the actual API call to a specific provider."""
        style = config["style"]
        api_key = config["api_key"]

        if style == "openai":
            return await self._call_openai_compatible(
                config["base_url"], api_key, config["model"], prompt, system,
            )
        elif style == "google":
            return await self._call_google(api_key, config["model"], prompt)
        elif style == "cohere":
            return await self._call_cohere(api_key, config["model"], prompt)
        elif style == "hf":
            return await self._call_hf(api_key, config["model"], prompt)
        else:
            raise ValueError(f"Unknown provider style: {style}")

    async def _call_openai_compatible(self, base_url: str, api_key: str, model: str, prompt: str, system: str) -> str:
        """Call any OpenAI-compatible API."""
        resp = await self._client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 1024,
                "temperature": 0.7,
            },
        )

        if resp.status_code == 429:
            raise RateLimitError(f"Rate limited by {base_url}")
        resp.raise_for_status()

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def _call_google(self, api_key: str, model: str, prompt: str) -> str:
        """Call Google AI Studio / Gemini API."""
        resp = await self._client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.7},
            },
        )

        if resp.status_code == 429:
            raise RateLimitError("Google rate limit")
        resp.raise_for_status()

        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    async def _call_cohere(self, api_key: str, model: str, prompt: str) -> str:
        """Call Cohere API."""
        resp = await self._client.post(
            "https://api.cohere.ai/v2/chat",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
            },
        )

        if resp.status_code == 429:
            raise RateLimitError("Cohere rate limit")
        resp.raise_for_status()

        data = resp.json()
        return data["message"]["content"][0]["text"]

    async def _call_hf(self, api_key: str, model: str, prompt: str) -> str:
        """Call HuggingFace Inference API."""
        resp = await self._client.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 512}},
        )

        if resp.status_code == 429:
            raise RateLimitError("HuggingFace rate limit")
        resp.raise_for_status()

        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "")
        return str(data)

    def provider_status(self) -> dict[str, Any]:
        """Return status of all providers."""
        return {
            name: {
                "available": stats.is_available,
                "success_rate": round(stats.success_rate, 3),
                "avg_latency_ms": round(stats.avg_latency_ms, 1),
                "daily_usage": self._daily_usage.get(name, 0),
                "daily_limit": self._providers[name]["daily_limit"],
                "rate_limits": stats.rate_limits,
            }
            for name, stats in self._stats.items()
        }


class RateLimitError(Exception):
    """Raised when a provider returns 429."""
    pass
