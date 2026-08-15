import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from typing import Any, AsyncIterator, Dict

from .base import BaseProvider, ProviderError, ProviderHealth
from ..models import ChatCompletionRequest, ProviderResult
from ..normalizer import flatten_for_chatgpt
from ..realtime import detect_realtime


class ChatGPTProvider(BaseProvider):
    """ChatGPT subscription provider using the current Codex Responses backend."""

    name = 'chatgpt'
    supports_streaming = True

    AUTH_URL = 'https://auth.openai.com/oauth/token'
    RESPONSES_URL = 'https://chatgpt.com/backend-api/codex/responses'
    DEFAULT_REDIRECT_URI = 'http://localhost:1455/auth/callback'

    def __init__(
        self,
        refresh_token: str,
        timeout: float,
        token_state_file: str = '',
        keepalive_hours: float = 6.0,
        web_search_mode: str = 'auto',
        web_search_instruction: str = '',
        access_token: str = '',
        access_token_expires_in: int = 0,
        client_id: str = '',
        redirect_uri: str = '',
        auth_url: str = AUTH_URL,
        account_id: str = '',
        responses_url: str = RESPONSES_URL,
        model_map: Dict[str, str] | None = None,
        originator: str = 'codex_cli_rs',
        version: str = '0.142.3',
    ):
        self.timeout = timeout
        self.token_state_file = token_state_file.strip()
        self.keepalive_hours = max(0.25, float(keepalive_hours))
        self.web_search_mode = (web_search_mode or 'auto').strip().lower()
        self.web_search_instruction = (web_search_instruction or '').strip()
        self.cached_access_token = (access_token or '').strip() or None
        self.expires_at = time.time() + max(0, int(access_token_expires_in or 0)) if self.cached_access_token and access_token_expires_in else 0.0
        self.lock = asyncio.Lock()
        self.keepalive_task = None
        self.log = logging.getLogger('9router.chatgpt')
        self.refresh_token = self._load_refresh_token(refresh_token or '')
        self.client_id = (client_id or '').strip()
        self.redirect_uri = (redirect_uri or self.DEFAULT_REDIRECT_URI).strip()
        self.auth_url = (auth_url or self.AUTH_URL).strip()
        self.account_id = (account_id or '').strip()
        self.responses_url = (responses_url or self.RESPONSES_URL).strip()
        self.originator = (originator or 'codex_cli_rs').strip()
        self.version = (version or '0.142.3').strip()
        self.model_map = {str(k).lower(): str(v) for k, v in (model_map or {}).items() if v}

    def _load_refresh_token(self, fallback: str) -> str:
        if not self.token_state_file:
            return fallback
        try:
            if os.path.exists(self.token_state_file):
                with open(self.token_state_file, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                token = data.get('refresh_token')
                if token:
                    return str(token)
        except Exception as exc:
            self.log.warning('Could not load ChatGPT token state: %s', exc)
        return fallback

    def _persist_refresh_token(self, token: str) -> None:
        if not self.token_state_file or not token:
            return
        path = os.path.abspath(self.token_state_file)
        directory = os.path.dirname(path) or '.'
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='.chatgpt-token-', dir=directory, text=True)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                json.dump({'refresh_token': token, 'updated_at': int(time.time())}, fh)
                fh.flush()
                os.fchmod(fh.fileno(), 0o600)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    async def start_keepalive(self):
        if self.keepalive_task is None or self.keepalive_task.done():
            self.keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def stop_keepalive(self):
        task = self.keepalive_task
        self.keepalive_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _keepalive_loop(self):
        interval = self.keepalive_hours * 3600.0
        await asyncio.sleep(min(300.0, interval))
        while True:
            try:
                await asyncio.sleep(interval)
                if self.refresh_token and self.client_id:
                    await self.force_refresh()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.log.warning('ChatGPT token keep-alive failed: %s', exc)

    async def force_refresh(self) -> str:
        async with self.lock:
            self.expires_at = 0.0
        return await self.access_token()

    @staticmethod
    def _requests():
        try:
            from curl_cffi import requests as cffi_requests
        except ImportError as exc:
            raise ProviderError('chatgpt', 'curl_cffi is required for ChatGPT provider', 503, False) from exc
        return cffi_requests

    @staticmethod
    def _jwt_payload(token: str) -> Dict[str, Any]:
        try:
            parts = token.split('.')
            if len(parts) != 3:
                return {}
            raw = parts[1] + '=' * (-len(parts[1]) % 4)
            return json.loads(base64.urlsafe_b64decode(raw.encode()).decode('utf-8'))
        except Exception:
            return {}

    @classmethod
    def extract_account_id(cls, token: str) -> str:
        payload = cls._jwt_payload(token)
        auth = payload.get('https://api.openai.com/auth') or {}
        return str(
            payload.get('chatgpt_account_id')
            or auth.get('chatgpt_account_id')
            or payload.get('https://api.openai.com/auth.chatgpt_account_id')
            or ''
        ).strip()

    @classmethod
    def token_diagnostics(cls, token: str) -> Dict[str, Any]:
        payload = cls._jwt_payload(token)
        exp = payload.get('exp')
        scopes = payload.get('scope') or payload.get('scopes') or ''
        if isinstance(scopes, list):
            scopes = ' '.join(str(x) for x in scopes)
        account_id = cls.extract_account_id(token)
        return {
            'jwt': bool(payload),
            'expired': bool(exp and float(exp) <= time.time()),
            'account_id': bool(account_id),
            'scopes': str(scopes),
        }

    async def access_token(self) -> str:
        async with self.lock:
            now = time.time()
            if self.cached_access_token and (self.expires_at == 0.0 or now < self.expires_at - 60):
                self._ensure_account_id(self.cached_access_token)
                return self.cached_access_token

            if not self.refresh_token:
                raise ProviderError(self.name, 'ChatGPT authentication is not configured. Set CHATGPT_ACCESS_TOKEN or CHATGPT_REFRESH_TOKEN.', 503, False)
            if not self.client_id:
                raise ProviderError(self.name, 'CHATGPT_CLIENT_ID is required for refresh-token authentication. For direct testing, set CHATGPT_ACCESS_TOKEN.', 503, False)

            cffi_requests = self._requests()
            payload = {
                'grant_type': 'refresh_token',
                'client_id': self.client_id,
                'refresh_token': self.refresh_token,
            }
            if self.redirect_uri:
                payload['redirect_uri'] = self.redirect_uri

            def refresh():
                return cffi_requests.post(
                    self.auth_url,
                    data=payload,
                    headers={'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json', 'User-Agent': 'codex_cli_rs/0.142.3'},
                    impersonate='chrome120',
                    timeout=15,
                )

            try:
                response = await asyncio.to_thread(refresh)
            except Exception as exc:
                raise ProviderError(self.name, f'token refresh failed: {exc}', retryable=True) from exc

            if response.status_code != 200:
                retryable = response.status_code == 429 or response.status_code >= 500
                raise ProviderError(self.name, f'token refresh HTTP {response.status_code}: {response.text[:500]}', response.status_code, retryable)

            try:
                data = response.json()
            except Exception as exc:
                raise ProviderError(self.name, f'invalid token response: {exc}', 502, True) from exc

            token = data.get('access_token')
            if not token:
                raise ProviderError(self.name, 'No access_token in refresh response', 502, False)

            rotated_refresh = data.get('refresh_token')
            if rotated_refresh and rotated_refresh != self.refresh_token:
                self.refresh_token = str(rotated_refresh)
                try:
                    self._persist_refresh_token(self.refresh_token)
                except Exception as exc:
                    raise ProviderError(self.name, 'ChatGPT refresh token rotated but could not be persisted', 500, False) from exc

            self.cached_access_token = str(token)
            self.expires_at = time.time() + max(60, int(data.get('expires_in', 3600)))
            self._ensure_account_id(self.cached_access_token)
            return self.cached_access_token

    def _ensure_account_id(self, token: str) -> None:
        if not self.account_id:
            self.account_id = self.extract_account_id(token)
        if not self.account_id:
            raise ProviderError(
                self.name,
                'ChatGPT access token is valid-looking but does not contain chatgpt_account_id. Set CHATGPT_ACCOUNT_ID explicitly or use a current Codex ChatGPT OAuth access token.',
                401,
                False,
            )

    def _provider_model(self, requested: str) -> str:
        return self.model_map.get(requested.lower(), requested)

    def _payload(self, req: ChatCompletionRequest, provider_model: str) -> Dict[str, Any]:
        prompt = flatten_for_chatgpt(req.messages)
        decision = detect_realtime(req.messages)
        mode = self.web_search_mode
        if mode == 'always' or (mode == 'auto' and decision.needs_fresh_info):
            if self.web_search_instruction:
                prompt = self.web_search_instruction + '\n\nUser request follows.\n' + prompt

        return {
            'model': provider_model,
            'store': False,
            'stream': True,
            'instructions': '',
            'input': [{
                'role': 'user',
                'content': [{'type': 'input_text', 'text': prompt}],
            }],
            'reasoning': {'effort': 'none'},
        }

    def _headers(self, token: str, session_id: str | None = None) -> Dict[str, str]:
        self._ensure_account_id(token)
        sid = session_id or str(uuid.uuid4())
        return {
            'Authorization': f'Bearer {token}',
            'ChatGPT-Account-ID': self.account_id,
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
            'OpenAI-Beta': 'responses=experimental',
            'Originator': self.originator,
            'Version': self.version,
            'User-Agent': f'{self.originator}/{self.version}',
            'Origin': 'https://chatgpt.com',
            'Referer': 'https://chatgpt.com/',
            'session-id': sid,
            'x-client-request-id': str(uuid.uuid4()),
        }

    @staticmethod
    def _text_from_event(event: Dict[str, Any]) -> str:
        typ = event.get('type', '')
        if typ == 'response.output_text.delta':
            return str(event.get('delta') or '')
        if typ == 'response.output_text.done':
            return ''
        # Be tolerant of response shapes observed by older Responses clients.
        if typ in ('response.completed', 'response.done'):
            return ''
        return ''

    @staticmethod
    def _error_text(response) -> str:
        body = response.text[:800]
        if 'Just a moment' in body or 'cf-chl-' in body or 'Cloudflare' in body:
            return f'Cloudflare challenge (HTTP {response.status_code})'
        try:
            data = response.json()
            if isinstance(data, dict):
                err = data.get('error') or data
                if isinstance(err, dict):
                    return str(err.get('message') or err.get('detail') or err)[:800]
        except Exception:
            pass
        return body

    def _raise_http(self, response):
        status = response.status_code
        detail = self._error_text(response)
        retryable = status in (408, 409, 425, 429) or status >= 500
        if status == 401:
            detail = f'ChatGPT Responses authentication rejected: {detail or "Unauthorized"}. Verify access token and ChatGPT-Account-ID.'
            retryable = False
        elif status == 403:
            detail = f'ChatGPT Responses forbidden: {detail or "Forbidden"}. The ChatGPT Web/Codex backend may be blocking this runtime or account route.'
            retryable = False
        elif status == 404:
            detail = f'ChatGPT Responses model/route not found: {detail or "Not Found"}'
            retryable = False
        raise ProviderError(self.name, detail, status, retryable)

    async def chat(self, req: ChatCompletionRequest, provider_model: str) -> ProviderResult:
        started = time.perf_counter()
        token = await self.access_token()
        provider_model = self._provider_model(provider_model)
        payload = self._payload(req, provider_model)
        cffi_requests = self._requests()
        session_id = str(uuid.uuid4())

        def call():
            return cffi_requests.post(
                self.responses_url,
                headers=self._headers(token, session_id),
                json=payload,
                impersonate='chrome120',
                timeout=self.timeout,
            )

        try:
            response = await asyncio.to_thread(call)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self.name, str(exc), 503, True) from exc

        if response.status_code >= 400:
            self._raise_http(response)

        final_text = ''
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode('utf-8', 'ignore')
                if not line.startswith('data:'):
                    continue
                raw = line[5:].strip()
                if raw == '[DONE]':
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                delta = self._text_from_event(event)
                if delta:
                    final_text += delta
        except Exception as exc:
            raise ProviderError(self.name, f'Invalid ChatGPT Responses SSE: {exc}', 502, True) from exc

        return ProviderResult(
            provider=self.name,
            response={
                'id': f'chatcmpl-{uuid.uuid4().hex}',
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': req.model,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': final_text}, 'finish_reason': 'stop'}],
            },
            model=provider_model,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def stream(self, req: ChatCompletionRequest, provider_model: str) -> AsyncIterator[Dict[str, Any]]:
        token = await self.access_token()
        provider_model = self._provider_model(provider_model)
        payload = self._payload(req, provider_model)
        cffi_requests = self._requests()
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        sentinel = object()
        session_id = str(uuid.uuid4())

        def worker():
            try:
                response = cffi_requests.post(
                    self.responses_url,
                    headers=self._headers(token, session_id),
                    json=payload,
                    impersonate='chrome120',
                    timeout=self.timeout,
                    stream=True,
                )
                if response.status_code >= 400:
                    try:
                        self._raise_http(response)
                    except ProviderError as err:
                        loop.call_soon_threadsafe(queue.put_nowait, err)
                    return
                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode('utf-8', 'ignore')
                    if not line.startswith('data:'):
                        continue
                    raw = line[5:].strip()
                    if raw == '[DONE]':
                        continue
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait, json.loads(raw))
                    except json.JSONDecodeError:
                        continue
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ProviderError(self.name, str(exc), 503, True))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        task = asyncio.create_task(asyncio.to_thread(worker))
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, ProviderError):
                    raise item
                delta = self._text_from_event(item)
                if delta:
                    yield {'choices': [{'index': 0, 'delta': {'content': delta}, 'finish_reason': None}]}
        finally:
            if not task.done():
                task.cancel()

    async def health(self) -> ProviderHealth:
        started = time.perf_counter()
        try:
            token = await self.access_token()
            diagnostics = self.token_diagnostics(token)
            if diagnostics['expired']:
                raise ProviderError(self.name, 'ChatGPT access token is expired', 401, False)
            self._ensure_account_id(token)
            return ProviderHealth(self.name, True, True, 'healthy', int((time.perf_counter() - started) * 1000), 'authenticated')
        except ProviderError as exc:
            return ProviderHealth(self.name, True, False, 'unhealthy', int((time.perf_counter() - started) * 1000), str(exc))
