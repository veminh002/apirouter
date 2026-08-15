import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional, Tuple

from .base import BaseProvider, ProviderError, ProviderHealth
from ..models import ChatCompletionRequest, ProviderResult
from ..normalizer import flatten_for_chatgpt
from ..realtime import detect_realtime


class ChatGPTProvider(BaseProvider):
    """ChatGPT subscription provider using the ChatGPT/Codex Responses backend.

    The endpoint is a private ChatGPT Web backend and can change independently
    of the public OpenAI API. Authentication is expected to come from a
    current ChatGPT/Codex OAuth credential.
    """

    name = 'chatgpt'
    supports_streaming = True

    AUTH_URL = 'https://auth.openai.com/oauth/token'
    RESPONSES_URL = 'https://chatgpt.com/backend-api/codex/responses'
    DEFAULT_CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann'
    DEFAULT_REDIRECT_URI = 'http://localhost:1455/auth/callback'
    DEFAULT_ORIGINATOR = 'codex_cli_rs'
    DEFAULT_VERSION = '0.144.1'

    def __init__(
        self,
        refresh_token: str,
        timeout: float,
        token_state_file: str = '',
        keepalive_hours: float = 6.0,
        web_search_mode: str = 'auto',
        web_search_instruction: str = '',
        access_token: str = '',
        id_token: str = '',
        account_id: str = '',
        access_token_expires_in: int = 0,
        client_id: str = DEFAULT_CLIENT_ID,
        redirect_uri: str = DEFAULT_REDIRECT_URI,
        auth_url: str = AUTH_URL,
        responses_url: str = RESPONSES_URL,
        originator: str = DEFAULT_ORIGINATOR,
        version: str = DEFAULT_VERSION,
    ):
        self.timeout = timeout
        self.token_state_file = token_state_file.strip()
        self.keepalive_hours = max(0.25, float(keepalive_hours))
        self.web_search_mode = (web_search_mode or 'auto').strip().lower()
        self.web_search_instruction = (web_search_instruction or '').strip()

        self.cached_access_token = (access_token or '').strip() or None
        self.cached_id_token = (id_token or '').strip() or None
        self.explicit_account_id = (account_id or '').strip() or None
        self.expires_at = (
            time.time() + max(0, int(access_token_expires_in or 0))
            if self.cached_access_token and access_token_expires_in else 0.0
        )

        self.lock = asyncio.Lock()
        self.keepalive_task = None
        self.log = logging.getLogger('9router.chatgpt')

        self.refresh_token = self._load_refresh_token(refresh_token or '')
        if self.cached_access_token and not self.expires_at:
            claims = self._decode_jwt(self.cached_access_token)
            if claims.get('exp'):
                self.expires_at = float(claims['exp'])
        self.client_id = (client_id or self.DEFAULT_CLIENT_ID).strip()
        self.redirect_uri = (redirect_uri or self.DEFAULT_REDIRECT_URI).strip()
        self.auth_url = (auth_url or self.AUTH_URL).strip()
        self.responses_url = (responses_url or self.RESPONSES_URL).strip()
        self.originator = (originator or self.DEFAULT_ORIGINATOR).strip()
        self.version = (version or self.DEFAULT_VERSION).strip()

        self.account_id: Optional[str] = self.explicit_account_id
        self.claims: Dict[str, Any] = {}
        self.missing_scopes: list[str] = []
        self._refresh_identity()

    @staticmethod
    def _decode_jwt(token: str) -> Dict[str, Any]:
        if not token or token.count('.') != 2:
            return {}
        try:
            part = token.split('.')[1]
            part += '=' * (-len(part) % 4)
            raw = base64.urlsafe_b64decode(part.encode('ascii'))
            obj = json.loads(raw.decode('utf-8'))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    @classmethod
    def _extract_identity(cls, token: str, id_token: str = '') -> Tuple[Optional[str], Dict[str, Any]]:
        """Extract account identity from the same JWT locations used by Codex."""
        id_claims = cls._decode_jwt(id_token)
        access_claims = cls._decode_jwt(token)

        for claims in (id_claims, access_claims):
            auth = claims.get('https://api.openai.com/auth')
            if isinstance(auth, dict):
                account_id = (
                    auth.get('chatgpt_account_id')
                    or auth.get('chatgpt_account_user_id')
                )
                if account_id:
                    return str(account_id), claims

            for key in (
                'chatgpt_account_id',
                'https://api.openai.com/auth.chatgpt_account_id',
            ):
                if claims.get(key):
                    return str(claims[key]), claims

            organizations = claims.get('organizations')
            if isinstance(organizations, list) and organizations:
                first = organizations[0]
                if isinstance(first, dict) and first.get('id'):
                    return str(first['id']), claims

        return None, id_claims or access_claims

    def _refresh_identity(self):
        token = self.cached_access_token or ''
        account_id, claims = self._extract_identity(token, self.cached_id_token or '')
        self.account_id = self.explicit_account_id or account_id
        self.claims = claims

        scopes = claims.get('scp', [])
        if isinstance(scopes, str):
            scopes = scopes.split()
        if not isinstance(scopes, list):
            scopes = []

        # Current ChatGPT/Codex integrations have used these connector scopes
        # for subscription-backed agent access. Do not block here: the backend
        # remains the authority and can return a more precise 401/403.
        required = {'api.connectors.read', 'api.connectors.invoke'}
        self.missing_scopes = sorted(required - set(str(x) for x in scopes))

    def _diagnostic(self) -> Dict[str, Any]:
        claims = self.claims or {}
        auth = claims.get('https://api.openai.com/auth')
        if not isinstance(auth, dict):
            auth = {}

        return {
            'account_id': bool(self.account_id),
            'token_expired': bool(
                claims.get('exp') and int(claims.get('exp')) <= int(time.time())
            ),
            'issuer': claims.get('iss'),
            'audience': claims.get('aud'),
            'plan_type': auth.get('chatgpt_plan_type'),
            'compute_residency': auth.get('chatgpt_compute_residency'),
            'missing_scopes': self.missing_scopes,
        }

    def _load_refresh_token(self, fallback: str) -> str:
        if not self.token_state_file:
            return fallback
        try:
            with open(self.token_state_file, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            token = data.get('refresh_token')
            if token:
                if not self.cached_id_token and data.get('id_token'):
                    self.cached_id_token = str(data['id_token'])
                if not self.explicit_account_id and data.get('account_id'):
                    self.explicit_account_id = str(data['account_id'])
                return str(token)
        except FileNotFoundError:
            pass
        except Exception as exc:
            self.log.warning('Could not load ChatGPT token state: %s', exc)
        return fallback

    def _persist_token_state(self, refresh_token: str, id_token: str = '', account_id: str = '') -> None:
        if not self.token_state_file:
            return

        path = os.path.abspath(self.token_state_file)
        directory = os.path.dirname(path) or '.'
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='.chatgpt-token-', dir=directory, text=True)
        try:
            state = {
                'refresh_token': refresh_token,
                'id_token': id_token or None,
                'account_id': account_id or None,
                'updated_at': int(time.time()),
            }
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                json.dump(state, fh, separators=(',', ':'))
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
                'curl_cffi is required for ChatGPT Web provider',
                503,
                retryable=False,
            ) from exc
        return cffi_requests

    async def access_token(self) -> str:
        async with self.lock:
            now = time.time()
            if self.cached_access_token and (
                self.expires_at == 0.0 or now < self.expires_at - 60
            ):
                self._refresh_identity()
                if self.account_id:
                    return self.cached_access_token

            if not self.refresh_token:
                raise ProviderError(
                    self.name,
                    'ChatGPT authentication is not configured. Set CHATGPT_ACCESS_TOKEN or CHATGPT_REFRESH_TOKEN.',
                    503,
                    False,
                )

            cffi_requests = self._requests()
            payload = {
                'grant_type': 'refresh_token',
                'client_id': self.client_id,
                'refresh_token': self.refresh_token,
            }

            def refresh():
                return cffi_requests.post(
                    self.auth_url,
                    data=payload,
                    headers={
                        'Content-Type': 'application/x-www-form-urlencoded',
                        'Accept': 'application/json',
                        'User-Agent': f'codex-cli/{self.version}',
                    },
                    impersonate='chrome120',
                    timeout=15,
                )

            try:
                response = await asyncio.to_thread(refresh)
            except Exception as exc:
                raise ProviderError(self.name, f'token refresh failed: {exc}', retryable=True) from exc

            if response.status_code != 200:
                body = response.text[:500]
                retryable = response.status_code == 429 or response.status_code >= 500
                raise ProviderError(
                    self.name,
                    f'token refresh HTTP {response.status_code}: {body}',
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

            rotated_refresh = data.get('refresh_token')
            if rotated_refresh and rotated_refresh != self.refresh_token:
                self.refresh_token = str(rotated_refresh)

            self.cached_access_token = str(token)
            self.cached_id_token = str(data.get('id_token') or self.cached_id_token or '')
            self.expires_at = time.time() + max(60, int(data.get('expires_in', 3600)))

            self._refresh_identity()

            if not self.account_id:
                diagnostic = self._diagnostic()
                raise ProviderError(
                    self.name,
                    'ChatGPT OAuth succeeded but no chatgpt_account_id was found in the ID/access token. '
                    'Set CHATGPT_ACCOUNT_ID explicitly or use a current Codex ChatGPT OAuth credential. '
                    f'diagnostic={json.dumps(diagnostic, separators=(",", ":"))}',
                    401,
                    False,
                )

            if self.token_state_file:
                try:
                    self._persist_token_state(
                        self.refresh_token,
                        self.cached_id_token or '',
                        self.account_id or '',
                    )
                except Exception as exc:
                    self.log.error('Failed to persist ChatGPT token state: %s', exc)

            return self.cached_access_token

    def _payload(self, req: ChatCompletionRequest, provider_model: str) -> Dict[str, Any]:
        prompt = flatten_for_chatgpt(req.messages)
        decision = detect_realtime(req.messages)
        should_hint_search = (
            self.web_search_mode == 'always'
            or (self.web_search_mode == 'auto' and decision.needs_fresh_info)
        )

        instructions = ''
        if should_hint_search and self.web_search_instruction:
            instructions = self.web_search_instruction

        user_content: Any = prompt
        payload: Dict[str, Any] = {
            'model': provider_model,
            'store': False,
            'stream': True,
            'instructions': instructions,
            'input': [{
                'role': 'user',
                'content': prompt,
            }],
            'reasoning': {'effort': 'none'},
        }

        if req.temperature is not None:
            payload['temperature'] = req.temperature
        if req.max_completion_tokens is not None:
            payload['max_output_tokens'] = req.max_completion_tokens
        elif req.max_tokens is not None:
            payload['max_output_tokens'] = req.max_tokens

        return payload

    def _headers(self, token: str) -> Dict[str, str]:
        if not self.account_id:
            raise ProviderError(
                self.name,
                'ChatGPT account ID is missing. Set CHATGPT_ACCOUNT_ID or provide a current ChatGPT/Codex OAuth token.',
                401,
                False,
            )

        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
            'User-Agent': f'codex-cli/{self.version}',
            'Originator': self.originator,
            'Version': self.version,
            'ChatGPT-Account-ID': self.account_id,
            'session-id': str(uuid.uuid4()),
            'x-client-request-id': str(uuid.uuid4()),
        }

        residency = self.claims.get('https://api.openai.com/auth', {})
        if isinstance(residency, dict):
            compute_residency = residency.get('chatgpt_compute_residency')
            if compute_residency and compute_residency != 'no_constraint':
                headers['x-openai-internal-codex-residency'] = str(compute_residency)

        return headers

    @staticmethod
    def _extract_text_event(obj: Dict[str, Any]) -> str:
        event_type = obj.get('type')
        if event_type in {
            'response.output_text.delta',
            'response.refusal.delta',
            'response.reasoning_summary_text.delta',
        }:
            value = obj.get('delta')
            return value if isinstance(value, str) else ''

        response = obj.get('response')
        if isinstance(response, dict):
            output = response.get('output')
            if isinstance(output, list):
                texts = []
                for item in output:
                    if not isinstance(item, dict):
                        continue
                    content = item.get('content')
                    if not isinstance(content, list):
                        continue
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get('text'), str):
                            texts.append(part['text'])
                return ''.join(texts)

        return ''

    @staticmethod
    def _error_message(response) -> str:
        body = (response.text or '').strip()
        if not body:
            return f'HTTP {response.status_code} with empty response body'

        try:
            obj = response.json()
            if isinstance(obj, dict):
                err = obj.get('error')
                if isinstance(err, dict):
                    message = err.get('message') or err.get('detail') or err.get('code')
                    if message:
                        return str(message)
                detail = obj.get('detail')
                if detail:
                    return str(detail)
        except Exception:
            pass

        lowered = body.lower()
        if '<html' in lowered or '<!doctype' in lowered:
            if 'just a moment' in lowered or 'cloudflare' in lowered:
                return f'ChatGPT HTTP {response.status_code}: Cloudflare challenge HTML'
            return f'ChatGPT HTTP {response.status_code}: HTML response from ChatGPT Web backend'

        return body[:500]

    async def chat(self, req: ChatCompletionRequest, provider_model: str) -> ProviderResult:
        started = time.perf_counter()
        token = await self.access_token()
        payload = self._payload(req, provider_model)
        headers = self._headers(token)
        cffi_requests = self._requests()

        def call():
            return cffi_requests.post(
                self.responses_url,
                headers=headers,
                json=payload,
                impersonate='chrome120',
                timeout=self.timeout,
            )

        try:
            response = await asyncio.to_thread(call)
        except Exception as exc:
            raise ProviderError(self.name, f'ChatGPT request failed: {exc}', 502, True) from exc

        if response.status_code >= 400:
            message = self._error_message(response)
            retryable = response.status_code in (408, 409, 425, 429) or response.status_code >= 500

            if response.status_code == 401:
                diagnostic = self._diagnostic()
                message = f'{message}; auth_diagnostic={json.dumps(diagnostic, separators=(",", ":"))}'
                retryable = False
            elif response.status_code == 403:
                retryable = False

            raise ProviderError(
                self.name,
                message,
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
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                text = self._extract_text_event(event)
                if text:
                    final_text += text
        except Exception as exc:
            raise ProviderError(self.name, f'Invalid Responses SSE response: {exc}', 502, True) from exc

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
        token = await self.access_token()
        payload = self._payload(req, provider_model)
        headers = self._headers(token)
        cffi_requests = self._requests()

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        sentinel = object()

        def worker():
            response = None
            try:
                response = cffi_requests.post(
                    self.responses_url,
                    headers=headers,
                    json=payload,
                    impersonate='chrome120',
                    timeout=self.timeout,
                    stream=True,
                )
                if response.status_code >= 400:
                    message = self._error_message(response)
                    if response.status_code == 401:
                        message += f'; auth_diagnostic={json.dumps(self._diagnostic(), separators=(",", ":"))}'
                    err = ProviderError(
                        self.name,
                        message,
                        response.status_code,
                        response.status_code in (408, 409, 425, 429) or response.status_code >= 500,
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
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ProviderError(self.name, f'ChatGPT stream failed: {exc}', 502, True),
                )
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

                event_type = item.get('type')
                text = self._extract_text_event(item)
                if text:
                    yield {
                        'choices': [{
                            'index': 0,
                            'delta': {'content': text},
                            'finish_reason': None,
                        }]
                    }

                if event_type == 'response.completed':
                    yield {
                        'choices': [{
                            'index': 0,
                            'delta': {},
                            'finish_reason': 'stop',
                        }]
                    }
        finally:
            if not task.done():
                task.cancel()

    async def health(self) -> ProviderHealth:
        started = time.perf_counter()
        try:
            token = await self.access_token()
            self._refresh_identity()
            if not self.account_id:
                raise ProviderError(
                    self.name,
                    f'ChatGPT token authenticated but account_id is missing; diagnostic={json.dumps(self._diagnostic(), separators=(",", ":"))}',
                    401,
                    False,
                )

            detail = 'authenticated'
            if self.missing_scopes:
                detail += f'; token_missing_scopes={",".join(self.missing_scopes)}'

            return ProviderHealth(
                self.name,
                True,
                True,
                'healthy',
                int((time.perf_counter() - started) * 1000),
                detail,
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
