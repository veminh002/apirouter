# 9Router v3

OpenAI-compatible AI gateway with provider registry, model aliases, fallback routing, real SSE streaming, circuit breakers, health checks and metrics.

## Architecture

```text
Client
  -> FastAPI /v1/chat/completions
  -> Auth + rate limit
  -> RoutingPolicy / ModelAlias
  -> ProviderRouter
       -> CircuitBreaker
       -> retry / Retry-After
       -> ProviderRegistry
            -> ChatGPT-Web (PRIMARY, refresh-token auth, true SSE)
            -> Groq (fallback)
            -> OpenRouter (fallback)
  -> OpenAI-compatible response / SSE
```

## Main endpoints

- `GET /health` - provider health + circuit state
- `GET /metrics` - Prometheus text metrics
- `GET /metrics/json` - JSON metrics
- `GET /v1/models` - configured logical model aliases
- `POST /v1/chat/completions` - OpenAI-compatible completion and true SSE streaming

## Model aliases

The client can continue sending logical names such as `gpt-4o-mini` while the router chooses the real provider model.

Routing can be overridden with environment variables such as:

```env
ALIAS_GPT_4O_MINI=groq:llama-3.1-8b-instant,openrouter:google/gemini-2.0-flash-001
```

For direct routing, `provider:model` syntax is supported, e.g. `openrouter:google/gemini-2.5-flash`.

## Circuit breaker

A provider opens its circuit after `CIRCUIT_FAILURE_THRESHOLD` retryable failures. After `CIRCUIT_RECOVERY_SECONDS`, a single half-open request is allowed to probe recovery.

## Streaming

ChatGPT-Web is the primary streaming provider. It uses the OAuth refresh token to obtain an access token, calls the ChatGPT Web conversation endpoint with SSE, and forwards incremental deltas. Groq and OpenRouter remain streaming fallbacks.

A transparent provider fallback is possible before the first stream chunk. Once chunks have reached the client, switching providers would corrupt the conversation stream, so the router emits a stream error instead.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --host 0.0.0.0 --port 10000
```


## Routing order
ChatGPT Web is the primary provider. For each logical model alias, routing uses ChatGPT first, then Groq, then OpenRouter.

## ChatGPT authentication
Set `CHATGPT_REFRESH_TOKEN` in `.env`. The application refreshes an access token through the same Auth0 refresh-token flow used by the original 9Router implementation and caches the access token until near expiry. Never commit the refresh token to source control.

## Streaming
When `stream=true`, ChatGPT Web SSE is consumed incrementally and converted to OpenAI-compatible SSE chunks. Because the ChatGPT Web endpoint sends cumulative text, the provider emits only the newly-added text delta. If a stream has already emitted data, the router does not switch providers mid-stream.


## ChatGPT token longevity

ChatGPT Web remains the primary provider. The router now caches access tokens, accepts and persists a rotated refresh token when returned, and can run a background keep-alive refresh. Set `CHATGPT_TOKEN_STATE_FILE` to a persistent storage path so a rotated token survives restarts, and use `CHATGPT_KEEPALIVE_HOURS` to control refresh cadence (default 6). The keep-alive can reset an issuer's idle lifetime when allowed, but no client can extend an issuer-defined absolute/max lifetime.

## Native ChatGPT Web realtime/search (100% free)

9Router does not call Tavily, Brave, Bing, Google Search, or any other external search API.
When `CHATGPT_WEB_SEARCH_MODE=auto`, realtime-looking requests receive a small instruction hint
inside the ChatGPT Web request asking the native ChatGPT Web assistant to use its own browsing/search
capability when available. The router itself never fetches web pages.

This is deliberately a best-effort bridge to the private ChatGPT Web backend. It does **not** guarantee
that every request will trigger native search because that decision is controlled by ChatGPT Web itself.
No paid or external search dependency is added.
