from typing import Dict, Iterable
from .providers.base import BaseProvider


class ProviderRegistry:
    def __init__(self, providers: Iterable[BaseProvider] = ()):
        self._providers: Dict[str, BaseProvider] = {p.name: p for p in providers}

    def register(self, provider: BaseProvider):
        self._providers[provider.name] = provider

    def get(self, name: str) -> BaseProvider:
        return self._providers[name]

    def all(self):
        return list(self._providers.values())

    def names(self):
        return list(self._providers.keys())
