import asyncio
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
    """Primary provider: ChatGPT Web backend authenticated by OAuth refresh token.

    This intentionally keeps the token-refresh flow from the original 9Router repo.
    The web backend is private/non-public and may change independently of the public API.
    """

    name = 'chatgpt'
    supports_streaming = True

    AUTH_URL = 'https://auth0.openai.com/oauth/token'
    CONVERSATION_URL = 'https://chatgpt.com/backend-api/conversation'
    CLIENT_ID = 'pdlLIX2Y72MIlIKCdACjhgptvBDjhSp8'
    REDIRECT_URI = 'com.openai.chat://auth0.openai.com/ios/com.openai.chat/callback'

    def __init__(self, refresh_token: str, access_token: str = '', access_token_expires_in: int = 0, timeout: float = 30.0, token_state_file: str = '', keepalive_hours: float = 6.0, web_search_mode: str = 'auto', web_search_instruction: str = ''):
        self.timeout = timeout
        self.token_state_file = token_state_file.strip()
        self.keepalive_hours = max(0.25, float(keepalive_hours))
        self.web_search_mode = (web_search_mode or 'auto').strip().lower()
        self.web_search_instruction = (web_search_instruction or '').strip()
        self.cached_access_token = access_token.strip() or None
        self.expires_at = time.time() + max(0, int(access_token_expires_in)) if self.cached_access_token and access_token_expires_in else 0.0
        self.lock = asyncio.Lock()
        self.keepalive_task = None
        self.log = logging.getLogger('9router.chatgpt')
        self.refresh_token = self._load_refresh_token(refresh_token or '')

    def _load_refresh_token(self, fallback: str) -> str:
        if not self.token_state_file:
            return fallback
        try:
            path = self.token_state_file
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as fh:
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
        # A successful refresh can reset an issuer's idle refresh-token lifetime.
        # It cannot extend an issuer-defined absolute/max lifetime.
        interval = self.keepalive_hours * 3600.0
        await asyncio.sleep(min(300.0, interval))
        while True:
            try:
                await asyncio.sleep(interval)
                if self.refresh_token:
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
            raise ProviderError(
                'chatgpt',
                'curl_cffi is required for ChatGPT-Web provider',
                503,
                retryable=False,
            ) from exc
        return cffi_requests

    async def access_token(self) -> str:
        if self.cached_access_token and (self.expires_at == 0.0 or time.time() < self.expires_at - 60):
            return self.cached_access_token
        if not self.refresh_token:
            raise ProviderError(self.name, 'ChatGPT authentication is not configured: set CHATGPT_REFRESH_TOKEN or CHATGPT_ACCESS_TOKEN', 503, False)

        async with self.lock:
            now = time.time()
            if self.cached_access_token and (self.expires_at == 0.0 or now < self.expires_at - 60):
                return self.cached_access_token

            cffi_requests = self._requests()
            payload = {
                'redirect_uri': self.REDIRECT_URI,
                'grant_type': 'refresh_token',
                'client_id': self.CLIENT_ID,
                'refresh_token': self.refresh_token,
            }

            def refresh():
                return cffi_requests.post(
                    self.AUTH_URL,
                    headers={
                        'Accept': 'application/json',
                        'Content-Type': 'application/x-www-form-urlencoded',
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    },
                    data=payload,
                    impersonate='chrome120',
                    timeout=15,
                )

            try:
                response = await asyncio.to_thread(refresh)
            except Exception as exc:
                raise ProviderError(self.name, f'token refresh failed: {exc}', retryable=True) from exc

            if response.status_code != 200:
                if response.status_code in (400, 401, 403):
                    detail = 'authentication rejected'
                    body = response.text[:300].replace('\n', ' ')
                    if 'Just a moment' in response.text or 'cf-chl-' in response.text:
                        detail = 'authentication endpoint returned a Cloudflare challenge; verify the OAuth flow/credential from a supported client rather than retrying'
                    raise ProviderError(self.name, f'token refresh HTTP {response.status_code}: {detail}: {body}', response.status_code, False)
                retryable = response.status_code == 429 or response.status_code >= 500
                raise ProviderError(
                    self.name,
                    f'token refresh HTTP {response.status_code}: {response.text[:300]}',
                    response.status_code,
                    retryable,
                )

            try:
                data = response.json()
            except Exception as exc:
                raise ProviderError(self.name, f'invalid token response: {exc}', 502, True) from exc

            token = data.get('access_token')
            if not token:
                raise ProviderError(self.name, 'No access_token in refresh response', 502, True)

            # If rotation is enabled, the newest refresh token must replace the old one.
            rotated_refresh = data.get('refresh_token')
            if rotated_refresh and rotated_refresh != self.refresh_token:
                self.refresh_token = str(rotated_refresh)
                try:
                    self._persist_refresh_token(self.refresh_token)
                except Exception as exc:
                    self.log.error('Failed to persist rotated ChatGPT refresh token: %s', exc)
                    raise ProviderError(
                        self.name,
                        'ChatGPT refresh token rotated but could not be persisted',
                        500,
                        retryable=False,
                    ) from exc

            self.cached_access_token = token
            self.expires_at = time.time() + max(60, int(data.get('expires_in', 3600)))
            return token

    def _payload(self, req: ChatCompletionRequest, provider_model: str) -> Dict[str, Any]:
        prompt = flatten_for_chatgpt(req.messages)
        decision = detect_realtime(req.messages)
        mode = self.web_search_mode
        should_hint_search = mode == 'always' or (mode == 'auto' and decision.needs_fresh_info)

        if should_hint_search and self.web_search_instruction:
            prompt = (
                self.web_search_instruction
                + '\n\nUser request follows.\n'
                + prompt
            )

        payload = {
            'action': 'next',
            'messages': [{
                'id': str(uuid.uuid4()),
                'author': {'role': 'user'},
                'content': {'content_type': 'text', 'parts': [prompt]},
            }],
            'model': provider_model,
            'parent_message_id': str(uuid.uuid4()),
            'history_and_training_disabled': True,
            'conversation_mode': {'kind': 'primary_assistant'},
        }

        # No external search service is called here. This metadata is only a
        # local observability hint and may be ignored by ChatGPT Web.
        if should_hint_search:
            payload['metadata'] = {'9router_web_search_hint': True, 'mode': mode}
        return payload

    def _headers(self, token: str):
        return {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
            'User-Agent': 'Mozilla/5.0',
        }

    @staticmethod
    def _extract_text(obj: Dict[str, Any]) -> str:
        parts = obj.get('message', {}).get('content', {}).get('parts', [])
        return parts[0] if parts and isinstance(parts[0], str) else ''

    async def chat(self, req: ChatCompletionRequest, provider_model: str) -> ProviderResult:
        started = time.perf_counter()
        token = await self.access_token()
        payload = self._payload(req, provider_model)
        cffi_requests = self._requests()

        def call():
            return cffi_requests.post(
                self.CONVERSATION_URL,
                headers=self._headers(token),
                json=payload,
                impersonate='chrome120',
                timeout=self.timeout,
            )

        try:
            response = await asyncio.to_thread(call)
        except Exception as exc:
            raise ProviderError(self.name, str(exc), retryable=True) from exc

        if response.status_code >= 400:
            retryable = response.status_code in (408, 409, 425, 429) or response.status_code >= 500
            if response.status_code in (401, 403):
                retryable = False
            raise ProviderError(
                self.name,
                f'ChatGPT HTTP {response.status_code}: {response.text[:300].replace(chr(10), " ")}',
                response.status_code,
                retryable,
            )

        final_text = ''
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode('utf-8', 'ignore')
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    continue
                try:
                    final_text = self._extract_text(json.loads(data)) or final_text
                except json.JSONDecodeError:
                    continue
        except Exception as exc:
            raise ProviderError(self.name, f'Invalid SSE response: {exc}', 502, True) from exc

        response_data = {
            'id': f'chatcmpl-{uuid.uuid4().hex}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': req.model,
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': final_text},
                'finish_reason': 'stop',
            }],
        }
        return ProviderResult(
            provider=self.name,
            response=response_data,
            model=provider_model,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def stream(self, req: ChatCompletionRequest, provider_model: str) -> AsyncIterator[Dict[str, Any]]:
        """Forward ChatGPT Web SSE incrementally instead of buffering the whole response."""
        token = await self.access_token()
        payload = self._payload(req, provider_model)
        cffi_requests = self._requests()
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        sentinel = object()

        def worker():
            try:
                response = cffi_requests.post(
                    self.CONVERSATION_URL,
                    headers=self._headers(token),
                    impersonate='chrome120',
                    timeout=self.timeout,
                    stream=True,
                )
                if response.status_code >= 400:
                    retryable = response.status_code in (408, 409, 425, 429) or response.status_code >= 500
                    if response.status_code in (401, 403):
                        retryable = False
                    err = ProviderError(
                        self.name,
                        f'ChatGPT HTTP {response.status_code}: {response.text[:300].replace(chr(10), " ")}',
                        response.status_code,
                        retryable,
                    )
                    loop.call_soon_threadsafe(queue.put_nowait, err)
                    return

                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode('utf-8', 'ignore')
                    if not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    loop.call_soon_threadsafe(queue.put_nowait, obj)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ProviderError(self.name, str(exc), retryable=True))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        task = asyncio.create_task(asyncio.to_thread(worker))
        previous_text = ''
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, ProviderError):
                    raise item
                text = self._extract_text(item)
                if not text:
                    continue
                # ChatGPT Web sends cumulative message content. Emit only the delta.
                delta = text[len(previous_text):] if text.startswith(previous_text) else text
                previous_text = text
                if delta:
                    yield {
                        'choices': [{
                            'index': 0,
                            'delta': {'content': delta},
                            'finish_reason': None,
                        }]
                    }
        finally:
            if not task.done():
                task.cancel()

    async def health(self) -> ProviderHealth:
        started = time.perf_counter()
        try:
            await self.access_token()
            return ProviderHealth(
                self.name,
                True,
                True,
                'healthy',
                int((time.perf_counter() - started) * 1000),
            )
        except ProviderError as exc:
            return ProviderHealth(
                self.name,
                True,
                False,
                'unhealthy',
                int((time.perf_counter() - started) * 1000),
                str(exc),
            )
