import json
import time
import httpx
from typing import Any, AsyncIterator, Dict

from .base import BaseProvider, ProviderError, ProviderHealth
from ..models import ChatCompletionRequest, ProviderResult
from ..normalizer import normalize_openai_response


class GroqProvider(BaseProvider):
    name = 'groq'
    supports_streaming = True
    base_url = 'https://api.groq.com/openai/v1/chat/completions'

    def __init__(self, api_key, timeout):
        self.api_key, self.timeout = api_key, timeout

    def _headers(self):
        return {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}

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
            raise ProviderError(self.name, str(e), retryable=True)
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
            raise ProviderError(self.name, str(e), retryable=True)

    async def health(self) -> ProviderHealth:
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout, 8.0)) as client:
                r = await client.get('https://api.groq.com/openai/v1/models', headers=self._headers())
            latency = int((time.perf_counter()-started)*1000)
            if r.status_code < 400:
                return ProviderHealth(self.name, True, True, 'healthy', latency)
            return ProviderHealth(self.name, True, False, 'unhealthy', latency, r.text[:300])
        except Exception as e:
            return ProviderHealth(self.name, True, False, 'unhealthy', int((time.perf_counter()-started)*1000), str(e))
