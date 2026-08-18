from functools import lru_cache
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', case_sensitive=False)

    router9_api_key: str = ''
    chatgpt_refresh_token: str = ''
    chatgpt_access_token: str = ''
    chatgpt_access_token_expires_in: int = 0
    chatgpt_client_id: str = ''
    chatgpt_redirect_uri: str = ''
    chatgpt_auth_url: str = 'https://auth.openai.com/oauth/token'
    chatgpt_account_id: str = ''
    chatgpt_id_token: str = ''
    chatgpt_responses_url: str = 'https://chatgpt.com/backend-api/codex/responses'
    chatgpt_originator: str = 'codex_cli_rs'
    chatgpt_version: str = '0.144.1'
    chatgpt_oauth_scope: str = 'openid profile email offline_access api.connectors.read api.connectors.invoke'
    chatgpt_token_state_file: str = ''
    database_url: str = ''
    chatgpt_token_encryption_key: str = ''
    chatgpt_keepalive_hours: float = 6.0
    chatgpt_web_search_mode: str = 'auto'
    chatgpt_web_search_instruction: str = (
        'When this request depends on information that may have changed recently, '
        'use ChatGPT Web native search/browsing when available in this conversation. '
        'Prefer current web-verified information over memory, and include source links or citations when the platform provides them. '
        'Never invent a search result or citation.'
    )
    groq_api_key: str = ''
    openrouter_api_key: str = ''
    tokenrouter_api_key: str = ''

    request_timeout: float = 45.0
    provider_timeout: float = 30.0
    max_retries: int = 1
    rate_limit_per_minute: int = 60
    max_concurrent_requests: int = 8

    enable_chatgpt: bool = True
    enable_groq: bool = True
    enable_openrouter: bool = True
    enable_tokenrouter: bool = True

    circuit_failure_threshold: int = 5
    circuit_recovery_seconds: float = 30.0

    openrouter_referer: str = 'https://github.com/9router/9router'
    openrouter_title: str = '9Router v3'

    # Comma-separated provider:model candidates, highest priority first.
    # Example: "openai:gpt-4o,groq:openai/gpt-oss-120b,openrouter:openai/gpt-4o-mini"
    # Kept as env-friendly strings so the routing table can evolve without code edits.
    alias_gpt_4o: str = 'chatgpt:gpt-5.6-terra,tokenrouter:qwen/qwen3.8-max-free,groq:openai/gpt-oss-120b,openrouter:openai/gpt-4o'
    alias_gpt_4o_mini: str = 'chatgpt:gpt-5.6-luna,tokenrouter:qwen/qwen3.8-max-free,groq:llama-3.1-8b-instant,openrouter:google/gemini-2.5-flash-lite'
    alias_gpt_4_turbo: str = 'chatgpt:gpt-5.6-terra,tokenrouter:qwen/qwen3.8-max-free,groq:openai/gpt-oss-120b,openrouter:openai/gpt-4o'
    alias_gpt_3_5_turbo: str = 'chatgpt:gpt-5.6-luna,tokenrouter:qwen/qwen3.8-max-free,groq:llama-3.1-8b-instant,openrouter:google/gemini-2.5-flash-lite'

    @field_validator('enable_chatgpt', 'enable_groq', 'enable_openrouter', 'enable_tokenrouter', mode='before')
    @classmethod
    def normalize_boolean(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            aliases = {'fasle': 'false', 'ture': 'true', 'flase': 'false', 'treu': 'true'}
            value = aliases.get(normalized, normalized)
        return value

    @property
    def configured_providers(self) -> List[str]:
        providers = []
        if self.enable_chatgpt:
            providers.append('chatgpt')
        if self.enable_tokenrouter and self.tokenrouter_api_key:
            providers.append('tokenrouter')
        if self.enable_groq and self.groq_api_key:
            providers.append('groq')
        if self.enable_openrouter and self.openrouter_api_key:
            providers.append('openrouter')
        return providers

    @staticmethod
    def parse_candidates(value: str):
        candidates = []
        for item in (value or '').split(','):
            item = item.strip()
            if not item or ':' not in item:
                continue
            provider, model = item.split(':', 1)
            if provider.strip() and model.strip():
                candidates.append((provider.strip(), model.strip()))
        return candidates


@lru_cache
def get_settings():
    return Settings()
