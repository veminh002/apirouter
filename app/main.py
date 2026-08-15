import asyncio
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .auth import validate_auth
from .circuit_breaker import CircuitBreaker
from .config import get_settings
from .metrics import Metrics
from .models import ChatCompletionRequest
from .provider_registry import ProviderRegistry
from .providers.chatgpt import ChatGPTProvider
from .providers.groq import GroqProvider
from .providers.openrouter import OpenRouterProvider
from .rate_limit import RateLimiter
from .router import ProviderRouter
from .routing import ModelAlias, RoutingPolicy

logger = logging.getLogger('9router')

settings = get_settings()
app = FastAPI(title='9Router', version='3.2.0')
limiter = RateLimiter(settings.rate_limit_per_minute)
semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
metrics = Metrics()
breaker = CircuitBreaker(settings.circuit_failure_threshold, settings.circuit_recovery_seconds)

registry = ProviderRegistry()
chatgpt_provider = None
if 'chatgpt' in settings.configured_providers:
    chatgpt_provider = ChatGPTProvider(
        settings.chatgpt_refresh_token,
        settings.provider_timeout,
        settings.chatgpt_token_state_file,
        settings.chatgpt_keepalive_hours,
        settings.chatgpt_web_search_mode,
        settings.chatgpt_web_search_instruction,
        settings.chatgpt_access_token,
        settings.chatgpt_access_token_expires_in,
        settings.chatgpt_client_id,
        settings.chatgpt_redirect_uri,
        settings.chatgpt_auth_url,
    )
    registry.register(chatgpt_provider)
if 'groq' in settings.configured_providers:
    registry.register(GroqProvider(settings.groq_api_key, settings.provider_timeout))
if 'openrouter' in settings.configured_providers:
    registry.register(OpenRouterProvider(settings.openrouter_api_key, settings.provider_timeout, settings.openrouter_referer, settings.openrouter_title))

policy = RoutingPolicy({
    'gpt-4o': ModelAlias('gpt-4o', settings.parse_candidates(settings.alias_gpt_4o)),
    'gpt-4o-mini': ModelAlias('gpt-4o-mini', settings.parse_candidates(settings.alias_gpt_4o_mini)),
    'gpt-4-turbo': ModelAlias('gpt-4-turbo', settings.parse_candidates(settings.alias_gpt_4_turbo)),
    'gpt-3.5-turbo': ModelAlias('gpt-3.5-turbo', settings.parse_candidates(settings.alias_gpt_3_5_turbo)),
})
router = ProviderRouter(registry, policy, settings.max_retries, breaker, metrics)


async def auth_and_limit(authorization: Optional[str]):
    validate_auth(authorization, settings)
    key = (authorization or 'anonymous').removeprefix('Bearer ').strip()
    await limiter.check(key[-64:] if key else 'anonymous')


def sse(data):
    return f'data: {json.dumps(data, ensure_ascii=False, separators=(",", ":"))}\n\n'


@app.on_event('startup')
async def startup_event():
    if chatgpt_provider is not None:
        await chatgpt_provider.start_keepalive()


@app.on_event('shutdown')
async def shutdown_event():
    if chatgpt_provider is not None:
        await chatgpt_provider.stop_keepalive()


@app.get('/health')
async def health():
    provider_health = []
    for provider in registry.all():
        result = await provider.health()
        provider_health.append(result.__dict__)
    state = await breaker.snapshot()
    healthy = all(x['healthy'] for x in provider_health) if provider_health else False
    return {
        'status': 'ok' if healthy else ('degraded' if provider_health else 'down'),
        'version': app.version,
        'providers': provider_health,
        'circuit_breakers': state,
        'time': int(time.time()),
    }


@app.get('/metrics')
async def metrics_endpoint():
    return PlainTextResponse(await metrics.prometheus(), media_type='text/plain; version=0.0.4')


@app.get('/metrics/json')
async def metrics_json():
    return await metrics.snapshot()


@app.get('/v1/models')
async def models(authorization: Optional[str] = Header(None)):
    await auth_and_limit(authorization)
    created = int(time.time())
    return {
        'object': 'list',
        'data': [
            {'id': name, 'object': 'model', 'created': created, 'owned_by': '9router'}
            for name in policy.models()
        ],
    }


@app.post('/v1/chat/completions')
async def chat(req: ChatCompletionRequest, authorization: Optional[str] = Header(None)):
    await auth_and_limit(authorization)
    if not registry.all():
        raise HTTPException(503, 'No provider is configured. Set CHATGPT_REFRESH_TOKEN and/or fallback provider API keys.')

    if not req.stream:
        async with semaphore:
            result, errors = await router.complete(req)
        if not result:
            logger.error('All providers failed for model=%s: %s', req.model, errors)
            await metrics.request('502')
            return JSONResponse(status_code=502, content={'error': {'message': 'All providers failed', 'type': 'router_error', 'code': 'provider_unavailable', 'details': errors}})
        await metrics.request('200')
        response = dict(result.response)
        response['model'] = req.model
        response['router'] = {
            'provider': result.provider,
            'provider_model': result.model,
            'latency_ms': result.latency_ms,
            'fallback_errors': errors,
        }
        return response

    async def event_stream():
        rid = f'chatcmpl-{uuid.uuid4().hex}'
        created = int(time.time())
        first = {'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model,
                 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}
        yield sse(first)
        try:
            async with semaphore:
                async for chunk in router.stream(req):
                    chunk = dict(chunk)
                    chunk['id'] = rid
                    chunk['object'] = 'chat.completion.chunk'
                    chunk['created'] = created
                    chunk['model'] = req.model
                    yield sse(chunk)
            final = {'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model,
                     'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}
            yield sse(final)
            yield 'data: [DONE]\n\n'
            await metrics.request('200')
        except Exception as exc:
            # The headers have already been sent. The client receives an SSE error event.
            logger.error('Stream failed for model=%s: %s', req.model, exc)
            error = {'error': {'message': str(exc), 'type': 'router_error', 'code': 'stream_error'}}
            yield sse(error)
            yield 'data: [DONE]\n\n'
            await metrics.request('502_stream')

    return StreamingResponse(
        event_stream(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache, no-transform', 'Connection': 'keep-alive', 'X-Accel-Buffering': 'no'},
    )


@app.get('/')
async def root():
    return {'name': '9Router', 'version': app.version, 'docs': '/docs', 'health': '/health', 'metrics': '/metrics'}
