import asyncio
import time
from typing import AsyncIterator, Dict, Optional

from .circuit_breaker import CircuitBreaker
from .metrics import Metrics
from .models import ChatCompletionRequest, ProviderResult
from .provider_registry import ProviderRegistry
from .providers.base import ProviderError
from .routing import RoutingPolicy


class ProviderRouter:
    def __init__(self, registry: ProviderRegistry, policy: RoutingPolicy, max_retries=1, breaker=None, metrics=None):
        self.registry = registry
        self.policy = policy
        self.max_retries = max(0, max_retries)
        self.breaker = breaker or CircuitBreaker()
        self.metrics = metrics or Metrics()

    async def complete(self, req: ChatCompletionRequest):
        errors = []
        route = self.policy.resolve(req.model)
        for index, (provider_name, provider_model) in enumerate(route.providers):
            try:
                provider = self.registry.get(provider_name)
            except KeyError:
                errors.append({'provider': provider_name, 'error': 'provider_not_configured'})
                continue
            if not await self.breaker.allow(provider.name):
                errors.append({'provider': provider.name, 'error': 'circuit_open'})
                continue
            for attempt in range(self.max_retries + 1):
                started = time.perf_counter()
                await self.metrics.provider_started(provider.name)
                try:
                    result = await provider.chat(req, provider_model)
                    await self.metrics.provider_finished(provider.name, int((time.perf_counter()-started)*1000))
                    await self.breaker.success(provider.name)
                    if index > 0:
                        await self.metrics.fallback()
                    return result, errors
                except ProviderError as e:
                    latency = int((time.perf_counter()-started)*1000)
                    await self.metrics.provider_finished(provider.name, latency, error=True)
                    if e.retryable:
                        await self.breaker.failure(provider.name)
                    errors.append({'provider': e.provider, 'status_code': e.status_code, 'error': str(e), 'attempt': attempt + 1})
                    if not e.retryable or attempt >= self.max_retries:
                        break
                    delay = e.retry_after if e.retry_after is not None else 0.35 * (attempt + 1)
                    await asyncio.sleep(min(30.0, max(0.05, delay)))
        return None, errors

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[Dict]:
        errors = []
        route = self.policy.resolve(req.model)
        for index, (provider_name, provider_model) in enumerate(route.providers):
            try:
                provider = self.registry.get(provider_name)
            except KeyError:
                errors.append({'provider': provider_name, 'error': 'provider_not_configured'})
                continue
            if not provider.supports_streaming:
                errors.append({'provider': provider.name, 'error': 'streaming_not_supported'})
                continue
            if not await self.breaker.allow(provider.name):
                errors.append({'provider': provider.name, 'error': 'circuit_open'})
                continue
            started = time.perf_counter()
            await self.metrics.provider_started(provider.name)
            emitted = False
            try:
                async for chunk in provider.stream(req, provider_model):
                    emitted = True
                    yield chunk
                await self.metrics.provider_finished(provider.name, int((time.perf_counter()-started)*1000))
                await self.breaker.success(provider.name)
                if index > 0:
                    await self.metrics.fallback()
                return
            except ProviderError as e:
                await self.metrics.provider_finished(provider.name, int((time.perf_counter()-started)*1000), error=True)
                if e.retryable:
                    await self.breaker.failure(provider.name)
                errors.append({'provider': e.provider, 'status_code': e.status_code, 'error': str(e)})
                # Once bytes/chunks reached the client, a transparent provider switch is unsafe.
                if emitted:
                    raise
        raise ProviderError('router', 'All streaming providers failed', 502, retryable=False)
