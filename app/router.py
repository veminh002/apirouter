import asyncio
import logging
import time
from typing import AsyncIterator, Dict, Optional

from .circuit_breaker import CircuitBreaker
from .metrics import Metrics
from .models import ChatCompletionRequest, ProviderResult
from .provider_registry import ProviderRegistry
from .providers.base import ProviderError
from .routing import RoutingPolicy

logger = logging.getLogger('9router.router')


class _NullAsyncContext:
    """No-op async context manager, used when no semaphore is supplied
    (e.g. in tests) so complete() doesn't need an `if self.semaphore` branch
    at every call site."""

    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc_info):
        return False


class ProviderRouter:
    def __init__(self, registry: ProviderRegistry, policy: RoutingPolicy, max_retries=1, breaker=None, metrics=None, semaphore: Optional[asyncio.Semaphore] = None):
        self.registry = registry
        self.policy = policy
        self.max_retries = max(0, max_retries)
        self.breaker = breaker or CircuitBreaker()
        self.metrics = metrics or Metrics()
        # Held only around the actual outbound HTTP call below, not around
        # retry backoff sleeps. Previously the caller (main.py) wrapped the
        # *entire* complete() call - including every `asyncio.sleep(delay)`
        # between retries - in `async with semaphore`, so a slow/failing
        # provider held a concurrency slot hostage during its own backoff,
        # exactly when the server needs that slot free for other requests.
        self.semaphore = semaphore or _NullAsyncContext()

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
                    async with self.semaphore:
                        result = await provider.chat(req, provider_model)
                    latency = int((time.perf_counter()-started)*1000)
                    await self.metrics.provider_finished(provider.name, latency)
                    await self.breaker.success(provider.name)
                    if index > 0:
                        await self.metrics.fallback()
                    logger.info('served requested_model=%s provider=%s provider_model=%s latency_ms=%s fallback=%s', req.model, provider.name, provider_model, latency, index > 0)
                    return result, errors
                except ProviderError as e:
                    latency = int((time.perf_counter()-started)*1000)
                    await self.metrics.provider_finished(provider.name, latency, error=True)
                    if e.retryable:
                        await self.breaker.failure(provider.name)
                    errors.append({'provider': e.provider, 'status_code': e.status_code, 'error': str(e), 'attempt': attempt + 1})
                    logger.warning('provider attempt failed requested_model=%s provider=%s provider_model=%s status=%s error=%s', req.model, e.provider, provider_model, e.status_code, e)
                    if not e.retryable or attempt >= self.max_retries:
                        break
                    delay = e.retry_after if e.retry_after is not None else 0.35 * (attempt + 1)
                    await asyncio.sleep(min(30.0, max(0.05, delay)))
                except Exception as e:
                    # A bug or unhandled exception inside provider.chat() must still
                    # release the breaker's half-open trial slot. Without this, a
                    # non-ProviderError failure during a half-open trial leaves
                    # half_open_in_flight=True forever, and breaker.allow() refuses
                    # every future request for this provider until process restart.
                    latency = int((time.perf_counter()-started)*1000)
                    await self.metrics.provider_finished(provider.name, latency, error=True)
                    await self.breaker.failure(provider.name)
                    errors.append({'provider': provider.name, 'status_code': None, 'error': str(e), 'attempt': attempt + 1})
                    break
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
                latency = int((time.perf_counter()-started)*1000)
                await self.metrics.provider_finished(provider.name, latency)
                await self.breaker.success(provider.name)
                if index > 0:
                    await self.metrics.fallback()
                logger.info('served (stream) requested_model=%s provider=%s provider_model=%s latency_ms=%s fallback=%s', req.model, provider.name, provider_model, latency, index > 0)
                return
            except ProviderError as e:
                await self.metrics.provider_finished(provider.name, int((time.perf_counter()-started)*1000), error=True)
                if e.retryable:
                    await self.breaker.failure(provider.name)
                errors.append({'provider': e.provider, 'status_code': e.status_code, 'error': str(e)})
                logger.warning('provider attempt failed (stream) requested_model=%s provider=%s provider_model=%s status=%s error=%s', req.model, e.provider, provider_model, e.status_code, e)
                # Once bytes/chunks reached the client, a transparent provider switch is unsafe.
                if emitted:
                    raise
            except Exception as e:
                # Same half-open deadlock risk as complete(): an unexpected exception
                # must still release the breaker's trial slot, or this provider stays
                # permanently blocked once it hits a half-open trial.
                await self.metrics.provider_finished(provider.name, int((time.perf_counter()-started)*1000), error=True)
                await self.breaker.failure(provider.name)
                errors.append({'provider': provider.name, 'status_code': None, 'error': str(e)})
                if emitted:
                    raise
        raise ProviderError('router', 'All streaming providers failed', 502, retryable=False)
