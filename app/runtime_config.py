import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .circuit_breaker import CircuitBreaker
from .config import Settings
from .metrics import Metrics
from .provider_registry import ProviderRegistry
from .providers.chatgpt import ChatGPTProvider
from .providers.groq import GroqProvider
from .providers.nvidia import NvidiaProvider
from .providers.openrouter import OpenRouterProvider
from .providers.tokenrouter import TokenRouterProvider
from .router import ProviderRouter
from .routing import ModelAlias, RoutingPolicy
from .tavily import TavilyClient

# Whitelist of fields the admin dashboard may override at runtime. Anything
# not in this set (timeouts, circuit breaker knobs, chatgpt OAuth fields...)
# stays env-only - chatgpt already has its own /auth/chatgpt login flow and
# its own persisted credential store, so it's deliberately not duplicated
# here.
EDITABLE_FIELDS = {
    'router9_api_key': str,
    'nvidia_api_key': str,
    'groq_api_key': str,
    'openrouter_api_key': str,
    'tokenrouter_api_key': str,
    'tavily_api_key': str,
    'enable_chatgpt': bool,
    'enable_nvidia': bool,
    'enable_groq': bool,
    'enable_openrouter': bool,
    'enable_tokenrouter': bool,
    'enable_tavily': bool,
    'model_aliases': dict,
}

# The 4 built-in alias names and the env field each one falls back to when
# no admin override (or tombstone) exists for that name.
LEGACY_ALIAS_FIELDS = {
    'gpt-4o': 'alias_gpt_4o',
    'gpt-4o-mini': 'alias_gpt_4o_mini',
    'gpt-4-turbo': 'alias_gpt_4_turbo',
    'gpt-3.5-turbo': 'alias_gpt_3_5_turbo',
}


def effective_aliases(settings: Settings) -> Dict[str, str]:
    """Merge the 4 built-in env-defined aliases with admin-added/edited
    ones. An override value of '' tombstones the name - including one of
    the 4 built-ins - so admins can remove a default alias entirely."""
    merged = {name: getattr(settings, field) for name, field in LEGACY_ALIAS_FIELDS.items()}
    for name, candidates in (settings.model_aliases or {}).items():
        if candidates:
            merged[name] = candidates
        else:
            merged.pop(name, None)
    return merged


def effective_settings(base: Settings, overrides: Dict[str, Any]) -> Settings:
    clean = {k: v for k, v in overrides.items() if k in EDITABLE_FIELDS}
    return base.model_copy(update=clean) if clean else base


@dataclass
class AppState:
    settings: Settings
    registry: ProviderRegistry
    policy: RoutingPolicy
    router: ProviderRouter
    breaker: CircuitBreaker
    metrics: Metrics
    chatgpt_provider: Optional[ChatGPTProvider]


def build_state(settings: Settings, semaphore: Optional[asyncio.Semaphore] = None, metrics: Optional[Metrics] = None) -> AppState:
    metrics = metrics or Metrics()
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
    if 'nvidia' in settings.configured_providers:
        registry.register(NvidiaProvider(settings.nvidia_api_key, settings.provider_timeout, settings.nvidia_web_search_mode, settings.nvidia_web_search_model))
    if 'tokenrouter' in settings.configured_providers:
        registry.register(TokenRouterProvider(settings.tokenrouter_api_key, settings.provider_timeout, settings.tokenrouter_web_search_mode, settings.tokenrouter_web_search_model))
    if 'groq' in settings.configured_providers:
        registry.register(GroqProvider(settings.groq_api_key, settings.provider_timeout, settings.groq_web_search_mode, settings.groq_web_search_model))
    if 'openrouter' in settings.configured_providers:
        registry.register(OpenRouterProvider(settings.openrouter_api_key, settings.provider_timeout, settings.openrouter_referer, settings.openrouter_title, settings.openrouter_web_search_mode, settings.openrouter_web_search_model))

    policy = RoutingPolicy({
        name: ModelAlias(name, settings.parse_candidates(candidates))
        for name, candidates in effective_aliases(settings).items()
    })
    router = ProviderRouter(
        registry, policy, settings.max_retries, breaker, metrics, semaphore=semaphore,
        search_client=TavilyClient(settings.tavily_api_key, settings.provider_timeout, settings.tavily_max_results) if settings.enable_tavily and settings.tavily_api_key else None,
        search_mode=settings.tavily_search_mode,
    )
    return AppState(settings, registry, policy, router, breaker, metrics, chatgpt_provider)
