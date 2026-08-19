import asyncio
import json
import logging
import time
import uuid
from typing import Optional
from urllib.parse import parse_qsl

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from .auth import validate_auth
from .circuit_breaker import CircuitBreaker
from .config import get_settings
from .metrics import Metrics
from .models import ChatCompletionRequest
from .provider_registry import ProviderRegistry
from .providers.base import ProviderError
from .providers.chatgpt import ChatGPTProvider
from .providers.groq import GroqProvider
from .providers.openrouter import OpenRouterProvider
from .providers.tokenrouter import TokenRouterProvider
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
if settings.enable_chatgpt:
    chatgpt_provider = ChatGPTProvider(
        settings.chatgpt_refresh_token,
        settings.provider_timeout,
        settings.chatgpt_token_state_file,
        settings.database_url,
        settings.chatgpt_token_encryption_key,
        settings.chatgpt_keepalive_hours,
        settings.chatgpt_web_search_mode,
        settings.chatgpt_web_search_instruction,
        settings.chatgpt_access_token,
        settings.chatgpt_access_token_expires_in,
        settings.chatgpt_client_id,
        settings.chatgpt_redirect_uri,
        settings.chatgpt_auth_url,
        settings.chatgpt_account_id,
        settings.chatgpt_id_token,
        settings.chatgpt_responses_url,
        settings.chatgpt_originator,
        settings.chatgpt_version,
        settings.chatgpt_oauth_scope,
    )
    registry.register(chatgpt_provider)
if 'tokenrouter' in settings.configured_providers:
    registry.register(TokenRouterProvider(settings.tokenrouter_api_key, settings.provider_timeout, settings.tokenrouter_web_search_mode, settings.tokenrouter_web_search_model))
if 'groq' in settings.configured_providers:
    registry.register(GroqProvider(settings.groq_api_key, settings.provider_timeout, settings.groq_web_search_mode, settings.groq_web_search_model))
if 'openrouter' in settings.configured_providers:
    registry.register(OpenRouterProvider(settings.openrouter_api_key, settings.provider_timeout, settings.openrouter_referer, settings.openrouter_title, settings.openrouter_web_search_mode, settings.openrouter_web_search_model))

policy = RoutingPolicy({
    'gpt-4o': ModelAlias('gpt-4o', settings.parse_candidates(settings.alias_gpt_4o)),
    'gpt-4o-mini': ModelAlias('gpt-4o-mini', settings.parse_candidates(settings.alias_gpt_4o_mini)),
    'gpt-4-turbo': ModelAlias('gpt-4-turbo', settings.parse_candidates(settings.alias_gpt_4_turbo)),
    'gpt-3.5-turbo': ModelAlias('gpt-3.5-turbo', settings.parse_candidates(settings.alias_gpt_3_5_turbo)),
})
router = ProviderRouter(registry, policy, settings.max_retries, breaker, metrics, semaphore=semaphore)


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


@app.get('/auth/chatgpt', response_class=HTMLResponse)
async def chatgpt_auth_start():
    if chatgpt_provider is None:
        return HTMLResponse('<h2>ChatGPT authentication is disabled.</h2>', status_code=503)
    authorize_url = chatgpt_provider.create_oauth_login()
    escaped = authorize_url.replace('&', '&amp;').replace('"', '&quot;')
    return HTMLResponse(f"""
<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>9Router ChatGPT Login</title>
<style>body{{font-family:system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 20px;line-height:1.5}}a,button{{font-size:16px}}textarea{{width:100%;min-height:120px;box-sizing:border-box}}.box{{padding:16px;border:1px solid #ddd;border-radius:12px;margin:16px 0}}</style>
</head><body>
<h1>Connect ChatGPT</h1>
<div class="box"><p><b>1.</b> Tap <b>Sign in with ChatGPT</b>.</p>
<p><a href="{escaped}" target="_blank">Sign in with ChatGPT</a></p></div>
<div class="box"><p><b>2.</b> After login, the browser will redirect to <code>http://localhost:1455/auth/callback</code>. On a phone that page may not open. That is expected.</p>
<p><b>3.</b> Copy the complete URL from the browser address bar.</p></div>
<div class="box"><p><b>4.</b> Paste the callback URL below.</p>
<form method="post" action="/auth/chatgpt/callback"><textarea name="callback_url" required placeholder="http://localhost:1455/auth/callback?code=...&state=..."></textarea><br><br><button type="submit">Complete connection</button></form></div>
<p>Never paste access tokens, cookies, or session tokens here.</p>
</body></html>""")


async def _finish_chatgpt_oauth(callback_url: str):
    if chatgpt_provider is None:
        return HTMLResponse('<h2>ChatGPT authentication is disabled.</h2>', status_code=503)
    try:
        result = await chatgpt_provider.complete_oauth_callback(callback_url)
    except ProviderError as exc:
        logger.warning('ChatGPT OAuth callback failed: %s', exc)
        safe = str(exc).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        return HTMLResponse(f'<h2>ChatGPT login failed</h2><pre>{safe}</pre><p><a href="/auth/chatgpt">Try again</a></p>', status_code=exc.status_code or 400)
    diagnostic = json.dumps(result.get('diagnostic', {}), indent=2)
    return HTMLResponse(f'<h2>ChatGPT connected</h2><p>Account ID detected: <b>{result.get("account_id") or "yes"}</b></p><p>You can now use <code>/v1/chat/completions</code>.</p><pre>{diagnostic}</pre>')


@app.get('/auth/chatgpt/callback', response_class=HTMLResponse)
async def chatgpt_auth_callback(callback_url: Optional[str] = None, request: Request = None):
    # Supports a browser GET with the callback URL in the query string.
    if callback_url:
        return await _finish_chatgpt_oauth(callback_url)
    if request is not None:
        params = dict(parse_qsl(request.url.query, keep_blank_values=True))
        if 'code' in params or 'error' in params:
            return await _finish_chatgpt_oauth(str(request.url))
    return HTMLResponse('<h2>Missing OAuth callback</h2><p>Return to <a href="/auth/chatgpt">ChatGPT login</a> and try again.</p>', status_code=400)


@app.post('/auth/chatgpt/callback', response_class=HTMLResponse)
async def chatgpt_auth_callback_post(request: Request):
    # The login page submits application/x-www-form-urlencoded. Parse it without
    # requiring python-multipart so the existing minimal image keeps working.
    body = (await request.body()).decode('utf-8', errors='replace')
    form = dict(parse_qsl(body, keep_blank_values=True))
    callback_url = form.get('callback_url', '').strip()
    if not callback_url:
        return HTMLResponse('<h2>Missing callback URL</h2><p>Paste the complete localhost callback URL.</p>', status_code=400)
    return await _finish_chatgpt_oauth(callback_url)


@app.get('/auth/chatgpt/status')
async def chatgpt_auth_status():
    if chatgpt_provider is None:
        return {'enabled': False}
    try:
        token = await chatgpt_provider.access_token()
        diagnostic = chatgpt_provider.token_diagnostic(
            token,
            chatgpt_provider.account_id or chatgpt_provider._extract_account_id(token, chatgpt_provider.id_token) or '',
        )
        return {
            'enabled': True,
            'authenticated': not diagnostic['token_expired'] and not diagnostic['missing_scopes'] and bool(diagnostic['account_id']),
            'account_id_present': bool(diagnostic['account_id']),
            'diagnostic': diagnostic,
        }
    except ProviderError as exc:
        return {
            'enabled': True,
            'authenticated': False,
            'account_id_present': bool(chatgpt_provider.account_id),
            'error': str(exc),
            'status_code': exc.status_code,
        }


@app.api_route('/health', methods=['GET', 'HEAD'])
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
        try:
            result, errors = await router.complete(req)
        except Exception as exc:
            logger.exception('Unhandled /v1/chat/completions failure for model=%s', req.model)
            await metrics.request('500')
            return JSONResponse(
                status_code=500,
                content={'error': {'message': 'Internal router error', 'type': 'internal_error', 'code': 'router_exception'}}
            )
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


@app.api_route('/', methods=['GET', 'HEAD'])
async def root():
    return {'name': '9Router', 'version': app.version, 'docs': '/docs', 'health': '/health', 'metrics': '/metrics'}
