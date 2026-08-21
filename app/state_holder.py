import asyncio
from typing import Any, Dict

from .config import Settings
from .metrics import Metrics
from .runtime_config import AppState, build_state, effective_settings
from .settings_store import SettingsStore


class StateHolder:
    def __init__(self, env_settings: Settings, max_concurrent_requests: int):
        self.env_settings = env_settings
        self.store = SettingsStore(env_settings.database_url, env_settings.chatgpt_token_encryption_key)
        self.metrics = Metrics()
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.overrides: Dict[str, Any] = self.store.load()
        self.state: AppState = self._build()

    def _build(self) -> AppState:
        settings = effective_settings(self.env_settings, self.overrides)
        return build_state(settings, semaphore=self.semaphore, metrics=self.metrics)

    async def apply(self, overrides: Dict[str, Any]) -> AppState:
        merged = {**self.overrides, **overrides}
        old_chatgpt = self.state.chatgpt_provider
        if old_chatgpt is not None:
            await old_chatgpt.stop_keepalive()
        self.store.save(merged)
        self.overrides = merged
        self.state = self._build()
        if self.state.chatgpt_provider is not None:
            await self.state.chatgpt_provider.start_keepalive()
        return self.state

    async def reset(self) -> AppState:
        old_chatgpt = self.state.chatgpt_provider
        if old_chatgpt is not None:
            await old_chatgpt.stop_keepalive()
        self.store.clear()
        self.overrides = {}
        self.state = self._build()
        if self.state.chatgpt_provider is not None:
            await self.state.chatgpt_provider.start_keepalive()
        return self.state
