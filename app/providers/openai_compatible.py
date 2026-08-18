import json
import time
import httpx
from typing import Any, AsyncIterator, Dict

from .base import BaseProvider, ProviderError, ProviderHealth
from ..models import ProviderResult
from ..normalizer import normalize_openai_response


class OpenAICompatibleProvider(BaseProvider):
    """Shared implementation for any upstream that speaks the OpenAI
    chat-completions wire format (POST base_url, SSE streaming, GET
    models_url for health). Groq, OpenRouter, and TokenRouter were previously
    three near-identical copies of this file; a fix to one (e.g. retry-after
    parsing) would silently not apply to the others.

    Subclasses set `name`, `base_url`, `models_url`. If the upstream needs
    headers beyond `Authorization: Bearer`, override `_extra_headers`.
    """

    supports_streaming = True
    base_url: str
    models_url: str

    def __init__(self, api_key, timeout):
        self.api_key, self.timeout = api_key, timeout

    def _extra_headers(self) -> Dict[str, str]:
        return {}

    def _headers(self):
        return {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json', **self._extra_headers()}

    def _payload(self, req, provider_model, stream=False):
        payload = req.model_dump(exclude_none=True)
        payload['model'] = provider_model
        payload['stream'] = stream
        return payload

    @staticmethod
    def _retry_after(headers):
        value = headers.get('retry-after') or headers.get('Retry-After')
        try:
            return float(value) if value else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _describe_exception(e: Exception) -> str:
        # Many httpx connection-level exceptions (ConnectTimeout, ConnectError,
        # ReadTimeout, etc.) have an empty str() - so a bare `str(e)` silently
        # produced error='' with no clue what actually failed (DNS, TLS,
        # timeout, refused connection...). Always include the exception type.
        text = str(e)
        return f'{type(e).__name__}: {text}' if text else type(e).__name__

    async def chat(self, req, provider_model):
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(self.base_url, headers=self._headers(), json=self._payload(req, provider_model))
            if r.status_code >= 400:
                raise ProviderError(self.name, r.text[:500], r.status_code, r.status_code in (408, 409, 425, 429) or r.status_code >= 500, self._retry_after(r.headers))
            return ProviderResult(provider=self.name, response=normalize_openai_response(r.json(), req.model, self.name), model=provider_model, latency_ms=int((time.perf_counter()-started)*1000))
        except ProviderError:
            raise
        except httpx.HTTPError as e:
            raise ProviderError(self.name, self._describe_exception(e), retryable=True)
        except ValueError as e:
            raise ProviderError(self.name, f'Invalid JSON response: {e}', 502, True)

    async def stream(self, req, provider_model) -> AsyncIterator[Dict[str, Any]]:
        payload = self._payload(req, provider_model, stream=True)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream('POST', self.base_url, headers=self._headers(), json=payload) as r:
                    if r.status_code >= 400:
                        body = await r.aread()
                        raise ProviderError(self.name, body.decode('utf-8', 'ignore')[:500], r.status_code, r.status_code in (408, 409, 425, 429) or r.status_code >= 500, self._retry_after(r.headers))
                    async for line in r.aiter_lines():
                        if not line or not line.startswith('data:'):
                            continue
                        data = line[5:].strip()
                        if data == '[DONE]':
                            return
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        yield obj
        except ProviderError:
            raise
        except httpx.HTTPError as e:
            raise ProviderError(self.name, self._describe_exception(e), retryable=True)

    async def health(self) -> ProviderHealth:
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout, 8.0)) as client:
                r = await client.get(self.models_url, headers=self._headers())
            latency = int((time.perf_counter()-started)*1000)
            if r.status_code < 400:
                return ProviderHealth(self.name, True, True, 'healthy', latency)
            return ProviderHealth(self.name, True, False, 'unhealthy', latency, r.text[:300])
        except Exception as e:
            return ProviderHealth(self.name, True, False, 'unhealthy', int((time.perf_counter()-started)*1000), str(e))
