import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from .base import BaseProvider, ProviderError, ProviderHealth
from ..models import ChatCompletionRequest, ProviderResult
from ..realtime import detect_realtime


class ChatGPTProvider(BaseProvider):
    """ChatGPT-authenticated Codex Responses transport.

    The ChatGPT Web conversation endpoint is intentionally not used as the
    primary transport. Current Codex clients use the Responses endpoint at
    /backend-api/codex/responses, with WebSocket transport first and HTTPS/SSE
    fallback. This endpoint is private and may change independently of the
    public OpenAI API.
    """

    name = "chatgpt"
    supports_streaming = True

    AUTH_URL = "https://auth0.openai.com/oauth/token"
    RESPONSES_HTTP_URL = "https://chatgpt.com/backend-api/codex/responses"
    RESPONSES_WS_URL = "wss://chatgpt.com/backend-api/codex/responses"
    DEFAULT_REDIRECT_URI = "com.openai.chat://auth0.openai.com/ios/com.openai.chat/callback"
    DEFAULT_ORIGINATOR = "codex_cli_rs"
    DEFAULT_VERSION = "0.0.1"
    WS_BETA = "responses_websockets=2026-02-06"

    def __init__(
        self,
        refresh_token: str,
        timeout: float,
        token_state_file: str = "",
        keepalive_hours: float = 6.0,
        web_search_mode: str = "auto",
        web_search_instruction: str = "",
        access_token: str = "",
        access_token_expires_in: int = 0,
        client_id: str = "",
        redirect_uri: str = "",
        auth_url: str = AUTH_URL,
        account_id: str = "",
        originator: str = DEFAULT_ORIGINATOR,
        version: str = DEFAULT_VERSION,
        responses_ws_url: str = RESPONSES_WS_URL,
        responses_http_url: str = RESPONSES_HTTP_URL,
    ):
        self.timeout = max(5.0, float(timeout))
        self.token_state_file = token_state_file.strip()
        self.keepalive_hours = max(0.25, float(keepalive_hours))
        self.web_search_mode = (web_search_mode or "auto").strip().lower()
        self.web_search_instruction = (web_search_instruction or "").strip()
        self.lock = asyncio.Lock()
        self.keepalive_task: Optional[asyncio.Task] = None
        self.log = logging.getLogger("9router.chatgpt")

        self.refresh_token = self._load_refresh_token(refresh_token or "")
        self.cached_access_token = (access_token or "").strip() or None
        self.expires_at = (
            time.time() + max(0, int(access_token_expires_in or 0))
            if self.cached_access_token and access_token_expires_in
            else 0.0
        )

        self.client_id = (client_id or "").strip()
        self.redirect_uri = (redirect_uri or self.DEFAULT_REDIRECT_URI).strip()
        self.auth_url = (auth_url or self.AUTH_URL).strip()
        self.account_id = (account_id or "").strip()

        self.originator = (originator or self.DEFAULT_ORIGINATOR).strip()
        self.version = (version or self.DEFAULT_VERSION).strip()
        self.responses_ws_url = (responses_ws_url or self.RESPONSES_WS_URL).strip()
        self.responses_http_url = (responses_http_url or self.RESPONSES_HTTP_URL).strip()

    def _load_refresh_token(self, fallback: str) -> str:
        if not self.token_state_file:
            return fallback
        try:
            if os.path.exists(self.token_state_file):
                with open(self.token_state_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                token = data.get("refresh_token")
                if token:
                    return str(token)
        except Exception as exc:
            self.log.warning("Could not load ChatGPT token state: %s", exc)
        return fallback

    def _persist_refresh_token(self, token: str) -> None:
        if not self.token_state_file or not token:
            return
        path = os.path.abspath(self.token_state_file)
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".chatgpt-token-", dir=directory, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(
                    {"refresh_token": token, "updated_at": int(time.time())},
                    fh,
                )
                fh.flush()
                os.fchmod(fh.fileno(), 0o600)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    async def start_keepalive(self) -> None:
        if self.keepalive_task is None or self.keepalive_task.done():
            self.keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def stop_keepalive(self) -> None:
        task = self.keepalive_task
        self.keepalive_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _keepalive_loop(self) -> None:
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
                self.log.warning("ChatGPT token keep-alive failed: %s", exc)

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
                "chatgpt",
                "curl_cffi is required for ChatGPT-Web provider",
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
                return self.cached_access_token

            if not self.refresh_token:
                raise ProviderError(
                    self.name,
                    "ChatGPT authentication is not configured. Set CHATGPT_ACCESS_TOKEN or CHATGPT_REFRESH_TOKEN.",
                    503,
                    False,
                )

            if not self.client_id:
                raise ProviderError(
                    self.name,
                    "CHATGPT_CLIENT_ID is required for refresh-token authentication. "
                    "Alternatively set CHATGPT_ACCESS_TOKEN for direct testing.",
                    503,
                    False,
                )

            cffi_requests = self._requests()
            payload = {
                "redirect_uri": self.redirect_uri,
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "refresh_token": self.refresh_token,
            }

            def refresh():
                return cffi_requests.post(
                    self.auth_url,
                    data=payload,
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                        "User-Agent": "Mozilla/5.0",
                    },
                    impersonate="chrome120",
                    timeout=15,
                )

            try:
                response = await asyncio.to_thread(refresh)
            except Exception as exc:
                raise ProviderError(
                    self.name, f"token refresh failed: {exc}", retryable=True
                ) from exc

            if response.status_code != 200:
                retryable = response.status_code == 429 or response.status_code >= 500
                raise ProviderError(
                    self.name,
                    f"token refresh HTTP {response.status_code}: {response.text[:500]}",
                    response.status_code,
                    retryable,
                )

            try:
                data = response.json()
            except Exception as exc:
                raise ProviderError(
                    self.name, f"invalid token response: {exc}", 502, True
                ) from exc

            token = data.get("access_token")
            if not token:
                raise ProviderError(
                    self.name, "No access_token in refresh response", 502, True
                )

            rotated_refresh = data.get("refresh_token")
            if rotated_refresh and rotated_refresh != self.refresh_token:
                self.refresh_token = str(rotated_refresh)
                try:
                    self._persist_refresh_token(self.refresh_token)
                except Exception as exc:
                    raise ProviderError(
                        self.name,
                        "ChatGPT refresh token rotated but could not be persisted",
                        500,
                        False,
                    ) from exc

            self.cached_access_token = str(token)
            self.expires_at = time.time() + max(60, int(data.get("expires_in", 3600)))
            return self.cached_access_token

    @staticmethod
    def _jwt_claims(token: str) -> Dict[str, Any]:
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return {}
            raw = parts[1]
            raw += "=" * (-len(raw) % 4)
            return json.loads(base64.urlsafe_b64decode(raw.encode("ascii")).decode("utf-8"))
        except Exception:
            return {}

    def _resolved_account_id(self, token: str) -> str:
        if self.account_id:
            return self.account_id
        auth_claim = self._jwt_claims(token).get("https://api.openai.com/auth") or {}
        if isinstance(auth_claim, dict):
            account_id = auth_claim.get("chatgpt_account_id")
            if isinstance(account_id, str) and account_id.strip():
                return account_id.strip()
        return ""

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
                    continue
                nested = item.get("content")
                if isinstance(nested, str):
                    chunks.append(nested)
            return "".join(chunks)
        if content is None:
            return ""
        return str(content)

    def _build_input(self, req: ChatCompletionRequest) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for message in req.messages:
            role = message.role
            if role not in {"system", "developer", "user", "assistant"}:
                role = "user"
            text = self._content_to_text(message.content)
            if text:
                items.append({"role": role, "content": text})
        return items

    def _payload(self, req: ChatCompletionRequest, provider_model: str) -> Dict[str, Any]:
        decision = detect_realtime(req.messages)
        mode = self.web_search_mode
        should_hint_search = mode == "always" or (
            mode == "auto" and decision.needs_fresh_info
        )

        instructions = ""
        if should_hint_search and self.web_search_instruction:
            instructions = self.web_search_instruction

        payload: Dict[str, Any] = {
            "type": "response.create",
            "model": provider_model,
            "input": self._build_input(req),
            "store": False,
        }
        if instructions:
            payload["instructions"] = instructions
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.top_p is not None:
            payload["top_p"] = req.top_p
        max_output_tokens = getattr(req, "max_output_tokens", None) or req.max_completion_tokens
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        return payload

    def _ws_headers(self, token: str, session_id: str) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "OpenAI-Beta": self.WS_BETA,
            "Originator": self.originator,
            "Version": self.version,
            "session_id": session_id,
            "User-Agent": f"{self.originator}/{self.version} (Linux; x86_64) unknown",
        }
        account_id = self._resolved_account_id(token)
        if account_id:
            headers["ChatGPT-Account-ID"] = account_id
        return headers

    def _http_headers(self, token: str) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Originator": self.originator,
            "Version": self.version,
            "User-Agent": f"{self.originator}/{self.version} (Linux; x86_64) unknown",
        }
        account_id = self._resolved_account_id(token)
        if account_id:
            headers["ChatGPT-Account-ID"] = account_id
        return headers

    @staticmethod
    def _event_text(event: Dict[str, Any]) -> str:
        event_type = event.get("type")
        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            delta = event.get("delta")
            return delta if isinstance(delta, str) else ""

        if event_type in {"response.output_text.done", "response.refusal.done"}:
            text = event.get("text")
            return text if isinstance(text, str) else ""

        return ""

    @classmethod
    def _completed_text(cls, event: Dict[str, Any]) -> str:
        response = event.get("response")
        if not isinstance(response, dict):
            response = event
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        output = response.get("output")
        if not isinstance(output, list):
            return ""

        parts: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        return "".join(parts)

    @staticmethod
    def _provider_error_from_event(event: Dict[str, Any]) -> ProviderError:
        error = event.get("error")
        if isinstance(error, dict):
            message = error.get("message") or event.get("message") or "ChatGPT Responses error"
            code = error.get("code") or error.get("type")
        else:
            message = event.get("message") or "ChatGPT Responses error"
            code = None
        status = event.get("status") or event.get("status_code")
        try:
            status_code = int(status) if status is not None else None
        except (TypeError, ValueError):
            status_code = None

        retryable = (
            status_code in {408, 409, 425, 429}
            or (status_code is not None and status_code >= 500)
        )
        if status_code == 403:
            retryable = False
        suffix = f" code={code}" if code else ""
        return ProviderError(
            "chatgpt",
            f"ChatGPT Responses error: {message}{suffix}",
            status_code,
            retryable,
        )

    async def _stream_ws(
        self, req: ChatCompletionRequest, provider_model: str, token: str
    ) -> AsyncIterator[Dict[str, Any]]:
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise ProviderError(
                self.name,
                "websockets package is required for ChatGPT Responses WebSocket transport",
                503,
                False,
            ) from exc

        session_id = str(uuid.uuid4())
        headers = self._ws_headers(token, session_id)
        payload = self._payload(req, provider_model)

        try:
            async with connect(
                self.responses_ws_url,
                additional_headers=headers,
                open_timeout=min(15.0, self.timeout),
                close_timeout=2.0,
                ping_interval=20.0,
                ping_timeout=20.0,
                max_size=8 * 1024 * 1024,
            ) as ws:
                await ws.send(json.dumps(payload, separators=(",", ":")))

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                    except asyncio.TimeoutError as exc:
                        raise ProviderError(
                            self.name,
                            "ChatGPT Responses WebSocket timed out waiting for an event",
                            504,
                            True,
                        ) from exc

                    if raw is None:
                        raise ProviderError(
                            self.name,
                            "ChatGPT Responses WebSocket closed before response.completed",
                            502,
                            True,
                        )

                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", "replace")

                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")

                    if event_type == "error":
                        raise self._provider_error_from_event(event)

                    if event_type in {
                        "response.completed",
                        "response.done",
                        "response.failed",
                        "response.incomplete",
                    }:
                        if event_type == "response.failed":
                            raise self._provider_error_from_event(event)
                        yield {
                            "__complete__": True,
                            "text": self._completed_text(event),
                        }
                        return

                    text = self._event_text(event)
                    if text:
                        yield {"__delta__": text}

        except ProviderError:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            message = str(exc)
            retryable = True
            if status == 403 or "403" in message or "Forbidden" in message:
                retryable = False
            raise ProviderError(
                self.name,
                f"ChatGPT Responses WebSocket failed: {message}",
                status or 502,
                retryable,
            ) from exc

    async def _stream_http(
        self, req: ChatCompletionRequest, provider_model: str, token: str
    ) -> AsyncIterator[Dict[str, Any]]:
        cffi_requests = self._requests()
        payload = self._payload(req, provider_model)
        payload.pop("type", None)
        payload["stream"] = True

        def call():
            return cffi_requests.post(
                self.responses_http_url,
                headers=self._http_headers(token),
                json=payload,
                impersonate="chrome120",
                timeout=self.timeout,
                stream=True,
            )

        try:
            response = await asyncio.to_thread(call)
        except Exception as exc:
            raise ProviderError(self.name, f"ChatGPT Responses HTTP failed: {exc}", 502, True) from exc

        if response.status_code >= 400:
            body = response.text[:500]
            retryable = response.status_code in {408, 409, 425, 429} or response.status_code >= 500
            if response.status_code == 403:
                retryable = False
            raise ProviderError(
                self.name,
                f"ChatGPT Responses HTTP {response.status_code}: {body}",
                response.status_code,
                retryable,
            )

        for line in response.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", "replace")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type", "")
            if event_type == "error":
                raise self._provider_error_from_event(event)
            if event_type == "response.failed":
                raise self._provider_error_from_event(event)
            if event_type in {"response.completed", "response.done", "response.incomplete"}:
                yield {"__complete__": True, "text": self._completed_text(event)}
                return
            text = self._event_text(event)
            if text:
                yield {"__delta__": text}

    async def _responses_stream(
        self, req: ChatCompletionRequest, provider_model: str, token: str
    ) -> AsyncIterator[Tuple[str, str]]:
        ws_error: Optional[ProviderError] = None
        try:
            async for event in self._stream_ws(req, provider_model, token):
                if event.get("__delta__"):
                    yield ("delta", event["__delta__"])
                elif event.get("__complete__"):
                    yield ("complete", event.get("text", ""))
            return
        except ProviderError as exc:
            ws_error = exc
            self.log.warning(
                "ChatGPT Responses WebSocket failed status=%s: %s; trying HTTPS fallback",
                exc.status_code,
                exc,
            )

        try:
            async for event in self._stream_http(req, provider_model, token):
                if event.get("__delta__"):
                    yield ("delta", event["__delta__"])
                elif event.get("__complete__"):
                    yield ("complete", event.get("text", ""))
            return
        except ProviderError as http_exc:
            if ws_error is not None and http_exc.status_code == 403 and ws_error.status_code:
                message = (
                    f"ChatGPT Responses transports failed. "
                    f"WebSocket: {ws_error}; HTTPS: {http_exc}"
                )
                raise ProviderError(
                    self.name,
                    message,
                    http_exc.status_code,
                    retryable=False,
                ) from http_exc
            raise

    async def chat(self, req: ChatCompletionRequest, provider_model: str) -> ProviderResult:
        started = time.perf_counter()
        token = await self.access_token()
        chunks: List[str] = []
        final_text = ""

        async for kind, text in self._responses_stream(req, provider_model, token):
            if kind == "delta" and text:
                chunks.append(text)
            elif kind == "complete" and text:
                final_text = text

        if not final_text:
            final_text = "".join(chunks)

        return ProviderResult(
            provider=self.name,
            response={
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": final_text},
                    "finish_reason": "stop",
                }],
            },
            model=provider_model,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def stream(
        self, req: ChatCompletionRequest, provider_model: str
    ) -> AsyncIterator[Dict[str, Any]]:
        token = await self.access_token()
        previous = ""
        emitted = False

        async for kind, text in self._responses_stream(req, provider_model, token):
            if kind == "delta":
                if not text:
                    continue
                emitted = True
                yield {
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }]
                }
                previous += text
            elif kind == "complete" and text and not emitted:
                # Some transports may only expose the final accumulated text.
                delta = text
                if previous and text.startswith(previous):
                    delta = text[len(previous):]
                if delta:
                    yield {
                        "choices": [{
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": None,
                        }]
                    }

    async def health(self) -> ProviderHealth:
        started = time.perf_counter()
        try:
            token = await self.access_token()
            account_id = self._resolved_account_id(token)
            return ProviderHealth(
                self.name,
                True,
                True,
                "authenticated",
                int((time.perf_counter() - started) * 1000),
                f"account_id={'present' if account_id else 'not_present'}; transport=websocket->https",
            )
        except ProviderError as exc:
            return ProviderHealth(
                self.name,
                True,
                False,
                "unhealthy",
                int((time.perf_counter() - started) * 1000),
                str(exc),
            )
