import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import tempfile
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional
from urllib.parse import parse_qs, urlencode, urlparse

from .base import BaseProvider, ProviderError, ProviderHealth
from ..models import ChatCompletionRequest, ProviderResult
from ..normalizer import flatten_for_chatgpt
from ..realtime import detect_realtime


class ChatGPTProvider(BaseProvider):
    """ChatGPT subscription provider using the Codex OAuth + Responses route."""

    name = 'chatgpt'
    supports_streaming = True

    AUTH_URL = 'https://auth.openai.com/oauth/token'
    AUTHORIZE_URL = 'https://auth.openai.com/oauth/authorize'
    RESPONSES_URL = 'https://chatgpt.com/backend-api/codex/responses'
    DEFAULT_CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann'
    DEFAULT_REDIRECT_URI = 'http://localhost:1455/auth/callback'
    DEFAULT_SCOPE = 'openid profile email offline_access api.connectors.read api.connectors.invoke'

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
        id_token: str = '',
        responses_url: str = RESPONSES_URL,
        originator: str = 'codex_cli_rs',
        version: str = '0.144.1',
        oauth_scope: str = DEFAULT_SCOPE,
    ):
        self.timeout = timeout
        self.token_state_file = token_state_file.strip()
        self.keepalive_hours = max(0.25, float(keepalive_hours))
        self.web_search_mode = (web_search_mode or 'auto').strip().lower()
        self.web_search_instruction = (web_search_instruction or '').strip()
        self.cached_access_token: Optional[str] = None
        self.expires_at = 0.0
        self.refresh_token = ''
        self.id_token = (id_token or '').strip()
        self.account_id = (account_id or '').strip()
        self.lock = asyncio.Lock()
        self.keepalive_task = None
        self.log = logging.getLogger('9router.chatgpt')
        self.client_id = (client_id or self.DEFAULT_CLIENT_ID).strip()
        self.redirect_uri = (redirect_uri or self.DEFAULT_REDIRECT_URI).strip()
        self.auth_url = (auth_url or self.AUTH_URL).strip()
        self.responses_url = (responses_url or self.RESPONSES_URL).strip()
        self.originator = (originator or 'codex_cli_rs').strip()
        self.version = (version or '0.144.1').strip()
        self.oauth_scope = (oauth_scope or self.DEFAULT_SCOPE).strip()
        self._oauth_sessions: Dict[str, Dict[str, str]] = {}

        state = self._load_token_state()
        self.refresh_token = state.get('refresh_token') or (refresh_token or '').strip()
        self.id_token = state.get('id_token') or self.id_token
        self.account_id = state.get('account_id') or self.account_id
        state_access = state.get('access_token') or ''
        self.cached_access_token = state_access or ((access_token or '').strip() or None)
        state_expires = float(state.get('expires_at') or 0)
        if state_expires:
            self.expires_at = state_expires
        elif self.cached_access_token and access_token_expires_in:
            self.expires_at = time.time() + max(0, int(access_token_expires_in))

        if self.cached_access_token and not self.account_id:
            self.account_id = self._extract_account_id(self.cached_access_token, self.id_token) or ''

    def _load_token_state(self) -> Dict[str, Any]:
        if not self.token_state_file:
            return {}
        try:
            if os.path.exists(self.token_state_file):
                with open(self.token_state_file, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                return data if isinstance(data, dict) else {}
        except Exception as exc:
            self.log.warning('Could not load ChatGPT token state: %s', exc)
        return {}

    def _restore_token_state_if_needed(self) -> bool:
        """Reload persisted OAuth state when this process has no in-memory credential."""
        if self.cached_access_token or self.refresh_token or not self.token_state_file:
            return bool(self.cached_access_token or self.refresh_token)
        state = self._load_token_state()
        if not state:
            return False
        self.refresh_token = str(state.get('refresh_token') or '')
        self.id_token = str(state.get('id_token') or self.id_token)
        self.account_id = str(state.get('account_id') or self.account_id)
        self.cached_access_token = str(state.get('access_token') or '') or None
        try:
            self.expires_at = float(state.get('expires_at') or 0)
        except (TypeError, ValueError):
            self.expires_at = 0.0
        if self.cached_access_token and not self.account_id:
            self.account_id = self._extract_account_id(self.cached_access_token, self.id_token) or ''
        return bool(self.cached_access_token or self.refresh_token)

    def _persist_token_state(self) -> None:
        if not self.token_state_file:
            return
        path = os.path.abspath(self.token_state_file)
        directory = os.path.dirname(path) or '.'
        os.makedirs(directory, exist_ok=True)
        data = {
            'refresh_token': self.refresh_token,
            'access_token': self.cached_access_token or '',
            'expires_at': self.expires_at,
            'id_token': self.id_token,
            'account_id': self.account_id,
            'updated_at': int(time.time()),
        }
        fd, tmp = tempfile.mkstemp(prefix='.chatgpt-token-', dir=directory, text=True)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                json.dump(data, fh)
                fh.flush()
                os.fchmod(fh.fileno(), 0o600)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @staticmethod
    def _b64decode(value: str) -> bytes:
        value += '=' * (-len(value) % 4)
        return base64.urlsafe_b64decode(value.encode())

    @classmethod
    def _decode_jwt(cls, token: str) -> Dict[str, Any]:
        try:
            parts = token.split('.')
            if len(parts) != 3:
                return {}
            return json.loads(cls._b64decode(parts[1]).decode('utf-8'))
        except Exception:
            return {}

    @classmethod
    def _extract_account_id(cls, access_token: str = '', id_token: str = '') -> Optional[str]:
        for token in (id_token, access_token):
            if not token:
                continue
            payload = cls._decode_jwt(token)
            auth = payload.get('https://api.openai.com/auth') or {}
            if isinstance(auth, dict):
                for key in ('chatgpt_account_id', 'account_id'):
                    value = auth.get(key)
                    if value:
                        return str(value)
            for key in ('chatgpt_account_id', 'account_id'):
                value = payload.get(key)
                if value:
                    return str(value)
        return None

    @classmethod
    def token_diagnostic(cls, access_token: str, account_id: str = '') -> Dict[str, Any]:
        payload = cls._decode_jwt(access_token)
        auth = payload.get('https://api.openai.com/auth') or {}
        if not isinstance(auth, dict):
            auth = {}
        scopes = payload.get('scp') or payload.get('scope') or []
        if isinstance(scopes, str):
            scopes = scopes.split()
        account = account_id or cls._extract_account_id(access_token) or ''
        exp = payload.get('exp')
        return {
            'account_id': bool(account),
            'token_expired': bool(exp and float(exp) <= time.time()),
            'issuer': payload.get('iss'),
            'audience': payload.get('aud'),
            'plan_type': auth.get('chatgpt_plan_type'),
            'compute_residency': auth.get('chatgpt_compute_residency'),
            'missing_scopes': [
                scope for scope in ('api.connectors.read', 'api.connectors.invoke') if scope not in scopes
            ],
        }

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
            raise ProviderError('chatgpt', 'curl_cffi is required for ChatGPT-Web provider', 503, False) from exc
        return cffi_requests

    def create_oauth_login(self) -> str:
        """Create a Codex-compatible PKCE login URL for a phone/browser."""
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
        state = secrets.token_urlsafe(32)
        self._oauth_sessions[state] = {'verifier': verifier, 'created_at': str(time.time())}
        # Keep only recent sessions.
        cutoff = time.time() - 900
        for key, item in list(self._oauth_sessions.items()):
            if float(item.get('created_at', 0)) < cutoff:
                self._oauth_sessions.pop(key, None)
        params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': self.oauth_scope,
            'code_challenge': challenge,
            'code_challenge_method': 'S256',
            'id_token_add_organizations': 'true',
            'codex_cli_simplified_flow': 'true',
            'state': state,
            'originator': self.originator,
        }
        return f'{self.AUTHORIZE_URL}?{urlencode(params)}'

    def parse_callback_url(self, callback_url: str) -> Dict[str, str]:
        parsed = urlparse(callback_url.strip())
        query = parse_qs(parsed.query)
        return {key: values[0] for key, values in query.items() if values}

    async def complete_oauth_callback(self, callback_url: str) -> Dict[str, Any]:
        params = self.parse_callback_url(callback_url)
        state = params.get('state', '')
        code = params.get('code', '')
        if params.get('error'):
            raise ProviderError(self.name, f'ChatGPT OAuth authorization failed: {params.get("error_description") or params.get("error")}', 400, False)
        session = self._oauth_sessions.pop(state, None)
        if not state or not session:
            raise ProviderError(self.name, 'Invalid or expired ChatGPT OAuth state. Start /auth/chatgpt again.', 400, False)
        if not code:
            raise ProviderError(self.name, 'OAuth callback did not contain an authorization code.', 400, False)

        payload = {
            'grant_type': 'authorization_code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'code': code,
            'code_verifier': session['verifier'],
        }
        cffi_requests = self._requests()

        def exchange():
            return cffi_requests.post(
                self.auth_url,
                data=payload,
                headers={'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'},
                impersonate='chrome120',
                timeout=20,
            )

        try:
            response = await asyncio.to_thread(exchange)
        except Exception as exc:
            raise ProviderError(self.name, f'ChatGPT OAuth token exchange failed: {exc}', 502, True) from exc
        if response.status_code >= 400:
            raise ProviderError(self.name, f'ChatGPT OAuth token exchange HTTP {response.status_code}: {response.text[:300]}', response.status_code, False)
        try:
            data = response.json()
        except Exception as exc:
            raise ProviderError(self.name, f'Invalid OAuth token response: {exc}', 502, False) from exc

        access_token = str(data.get('access_token') or '')
        refresh_token = str(data.get('refresh_token') or '')
        id_token = str(data.get('id_token') or '')
        if not access_token:
            raise ProviderError(self.name, 'OAuth token response did not contain access_token.', 502, False)
        account_id = self._extract_account_id(access_token, id_token) or self.account_id
        diagnostic = self.token_diagnostic(access_token, account_id)
        if diagnostic['missing_scopes']:
            raise ProviderError(
                self.name,
                f'New ChatGPT OAuth token is missing required Codex scopes: {", ".join(diagnostic["missing_scopes"])}',
                401,
                False,
            )

        self.cached_access_token = access_token
        self.refresh_token = refresh_token or self.refresh_token
        self.id_token = id_token or self.id_token
        self.account_id = account_id or ''
        self.expires_at = time.time() + max(60, int(data.get('expires_in', 3600)))
        self._persist_token_state()
        return {'account_id': self.account_id, 'expires_at': self.expires_at, 'diagnostic': diagnostic}

    async def access_token(self) -> str:
        async with self.lock:
            # OAuth callback and API requests can be handled by different
            # workers/instances. Reload the persisted credential before
            # deciding that authentication is missing.
            self._restore_token_state_if_needed()
            now = time.time()

            # A Web session access token can be JWT-valid and unexpired while
            # still lacking the Codex connector scopes required by the
            # /backend-api/codex/responses route. Never keep using such a token
            # when a refresh token is available: refresh first and re-evaluate
            # the newly issued credential.
            if self.cached_access_token and (self.expires_at == 0.0 or now < self.expires_at - 60):
                diagnostic = self.token_diagnostic(
                    self.cached_access_token,
                    self.account_id or self._extract_account_id(self.cached_access_token, self.id_token) or '',
                )
                if not diagnostic['missing_scopes']:
                    return self.cached_access_token
                if not self.refresh_token:
                    raise ProviderError(
                        self.name,
                        'ChatGPT token missing Codex scopes; sign in again via /auth/chatgpt to obtain a Codex OAuth token. '
                        f'auth_diagnostic={json.dumps(diagnostic, separators=(",", ":"))}',
                        401,
                        False,
                    )
                self.log.info(
                    'Cached ChatGPT token lacks Codex scopes; refreshing OAuth credential instead of using the Web session token.'
                )
                self.cached_access_token = None
                self.expires_at = 0.0

            if not self.refresh_token:
                raise ProviderError(self.name, 'ChatGPT authentication is not configured. Open /auth/chatgpt to sign in.', 503, False)

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
                    headers={'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'},
                    impersonate='chrome120',
                    timeout=15,
                )

            try:
                response = await asyncio.to_thread(refresh)
            except Exception as exc:
                raise ProviderError(self.name, f'token refresh failed: {exc}', 502, True) from exc
            if response.status_code != 200:
                retryable = response.status_code == 429 or response.status_code >= 500
                raise ProviderError(self.name, f'token refresh HTTP {response.status_code}: {response.text[:300]}', response.status_code, retryable)
            try:
                data = response.json()
            except Exception as exc:
                raise ProviderError(self.name, f'invalid token response: {exc}', 502, False) from exc
            token = data.get('access_token')
            if not token:
                raise ProviderError(self.name, 'No access_token in refresh response', 502, False)

            new_access_token = str(token)
            new_id_token = str(data.get('id_token') or self.id_token)
            new_account_id = self._extract_account_id(new_access_token, new_id_token) or self.account_id
            diagnostic = self.token_diagnostic(new_access_token, new_account_id)
            if diagnostic['missing_scopes']:
                raise ProviderError(
                    self.name,
                    'Refreshed ChatGPT token is still missing Codex scopes; the OAuth client/authorization flow did not grant the required scopes. '
                    f'auth_diagnostic={json.dumps(diagnostic, separators=(",", ":"))}',
                    401,
                    False,
                )

            self.cached_access_token = new_access_token
            self.refresh_token = str(data.get('refresh_token') or self.refresh_token)
            self.id_token = new_id_token
            self.account_id = new_account_id
            self.expires_at = time.time() + max(60, int(data.get('expires_in', 3600)))
            try:
                self._persist_token_state()
            except Exception as exc:
                self.log.warning('Failed to persist ChatGPT token state: %s', exc)
            return self.cached_access_token

    def _payload(self, req: ChatCompletionRequest, provider_model: str) -> Dict[str, Any]:
        prompt = flatten_for_chatgpt(req.messages)
        decision = detect_realtime(req.messages)
        mode = self.web_search_mode
        should_hint_search = mode == 'always' or (mode == 'auto' and decision.needs_fresh_info)
        if should_hint_search and self.web_search_instruction:
            prompt = self.web_search_instruction + '\n\nUser request follows.\n' + prompt
        payload: Dict[str, Any] = {
            'model': provider_model,
            'instructions': 'You are a helpful assistant.',
            'store': False,
            'stream': True,
            'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': prompt}]}],
        }
        # Codex Responses rejects the Chat Completions sampling parameters
        # `temperature` and `top_p`; intentionally omit them.
        if req.max_completion_tokens is not None:
            payload['max_output_tokens'] = req.max_completion_tokens
        elif req.max_tokens is not None:
            payload['max_output_tokens'] = req.max_tokens
        if should_hint_search:
            payload['metadata'] = {'9router_web_search_hint': True, 'mode': mode}
        return payload

    def _headers(self, token: str) -> Dict[str, str]:
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
            'User-Agent': '9Router/3 Codex-Compatible',
            'Originator': self.originator,
            'Version': self.version,
            'Origin': 'https://chatgpt.com',
            'Referer': 'https://chatgpt.com/',
        }
        if self.account_id:
            headers['ChatGPT-Account-ID'] = self.account_id
        return headers

    @staticmethod
    def _extract_text(obj: Dict[str, Any]) -> str:
        if obj.get('type') == 'response.output_text.delta':
            return str(obj.get('delta') or '')
        if obj.get('type') == 'response.completed':
            return ''
        if isinstance(obj.get('delta'), str):
            return obj['delta']
        return ''

    async def chat(self, req: ChatCompletionRequest, provider_model: str) -> ProviderResult:
        started = time.perf_counter()
        token = await self.access_token()
        self.account_id = self.account_id or self._extract_account_id(token, self.id_token) or ''
        diagnostic = self.token_diagnostic(token, self.account_id)
        if not self.account_id:
            raise ProviderError(
                self.name,
                f'ChatGPT authentication missing account ID; auth_diagnostic={json.dumps(diagnostic, separators=(",", ":"))}',
                401,
                False,
            )
        if diagnostic['missing_scopes']:
            raise ProviderError(
                self.name,
                f'ChatGPT token missing Codex scopes; auth_diagnostic={json.dumps(diagnostic, separators=(",", ":"))}',
                401,
                False,
            )

        payload = self._payload(req, provider_model)
        # /v1/chat/completions with stream=false must use the non-streaming
        # Responses representation. The previous implementation requested an
        # SSE stream and then consumed it synchronously, which could raise
        # curl_cffi's "stream mode is not enabled" assertion or leak an
        # unhandled provider exception as HTTP 500.
        payload['stream'] = False
        cffi_requests = self._requests()

        def call():
            return cffi_requests.post(
                self.responses_url,
                headers={
                    **self._headers(token),
                    'Accept': 'application/json',
                },
                json=payload,
                impersonate='chrome120',
                timeout=self.timeout,
                stream=False,
            )

        try:
            response = await asyncio.to_thread(call)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self.name, f'ChatGPT request failed: {exc}', 502, True) from exc

        try:
            if response.status_code >= 400:
                text = (response.text or '')[:500]
                retryable = response.status_code in (408, 409, 425, 429) or response.status_code >= 500
                raise ProviderError(
                    self.name,
                    f'ChatGPT Responses HTTP {response.status_code}: {text}',
                    response.status_code,
                    retryable,
                )

            try:
                obj = response.json()
            except Exception as exc:
                text = (response.text or '')[:500]
                raise ProviderError(
                    self.name,
                    f'ChatGPT Responses returned invalid JSON: {exc}; body={text}',
                    502,
                    True,
                ) from exc

            final_text = self._extract_response_text(obj)
            if not final_text:
                raise ProviderError(
                    self.name,
                    f'ChatGPT Responses returned no assistant text; response_type={obj.get("type") if isinstance(obj, dict) else type(obj).__name__}',
                    502,
                    True,
                )

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
                self.name,
                response_data,
                provider_model,
                int((time.perf_counter() - started) * 1000),
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self.name, f'ChatGPT response parsing failed: {exc}', 502, True) from exc
        finally:
            try:
                response.close()
            except Exception:
                pass

    @staticmethod
    def _extract_response_text(obj: Any) -> str:
        """Extract assistant text from Responses API JSON, tolerating variants."""
        if not isinstance(obj, dict):
            return ''

        # Common Responses API convenience field.
        output_text = obj.get('output_text')
        if isinstance(output_text, str) and output_text:
            return output_text

        parts = []
        output = obj.get('output') or []
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get('content') or []
                if isinstance(content, str):
                    parts.append(content)
                    continue
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    text = block.get('text')
                    if isinstance(text, str):
                        parts.append(text)
                    elif isinstance(block.get('value'), str):
                        parts.append(block['value'])

        # OpenAI-compatible fallback.
        if not parts:
            choices = obj.get('choices') or []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get('message') or {}
                content = message.get('content')
                if isinstance(content, str):
                    return content

        return ''.join(parts).strip()
    async def stream(self, req: ChatCompletionRequest, provider_model: str) -> AsyncIterator[Dict[str, Any]]:
        token = await self.access_token()
        self.account_id = self.account_id or self._extract_account_id(token, self.id_token) or ''
        diagnostic = self.token_diagnostic(token, self.account_id)
        if not self.account_id or diagnostic['missing_scopes']:
            raise ProviderError(self.name, f'ChatGPT authentication rejected; auth_diagnostic={json.dumps(diagnostic, separators=(",", ":"))}', 401, False)
        payload = self._payload(req, provider_model)
        cffi_requests = self._requests()
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        sentinel = object()

        def worker():
            try:
                response = cffi_requests.post(self.responses_url, headers=self._headers(token), json=payload, impersonate='chrome120', timeout=self.timeout, stream=True)
                if response.status_code >= 400:
                    retryable = response.status_code in (408, 409, 425, 429) or response.status_code >= 500
                    loop.call_soon_threadsafe(queue.put_nowait, ProviderError(self.name, f'ChatGPT Responses HTTP {response.status_code}: {response.text[:500]}', response.status_code, retryable))
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
                        loop.call_soon_threadsafe(queue.put_nowait, json.loads(data))
                    except json.JSONDecodeError:
                        continue
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ProviderError(self.name, str(exc), 502, True))
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
                text = self._extract_text(item)
                if text:
                    yield {'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]}
        finally:
            if not task.done():
                task.cancel()

    async def health(self) -> ProviderHealth:
        started = time.perf_counter()
        if not self.cached_access_token and not self.refresh_token:
            return ProviderHealth(self.name, False, False, 'authentication_required', int((time.perf_counter() - started) * 1000), 'Open /auth/chatgpt to sign in')
        try:
            token = await self.access_token()
            diagnostic = self.token_diagnostic(token, self.account_id or self._extract_account_id(token, self.id_token) or '')
            if diagnostic['missing_scopes']:
                return ProviderHealth(self.name, True, False, 'missing_scopes', int((time.perf_counter() - started) * 1000), json.dumps(diagnostic, separators=(',', ':')))
            return ProviderHealth(self.name, True, True, 'healthy', int((time.perf_counter() - started) * 1000))
        except ProviderError as exc:
            return ProviderHealth(self.name, True, False, 'unhealthy', int((time.perf_counter() - started) * 1000), str(exc))
