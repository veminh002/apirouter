from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

from ..models import ChatCompletionRequest, ProviderResult


class ProviderError(Exception):
    def __init__(
        self,
        provider: str,
        message: str,
        status_code: Optional[int] = None,
        retryable: bool = True,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after


@dataclass
class ProviderHealth:
    provider: str
    configured: bool = True
    healthy: bool = True
    status: str = "unknown"
    latency_ms: Optional[int] = None
    error: Optional[str] = None


class BaseProvider(ABC):
    name: str
    supports_streaming: bool = False

    @abstractmethod
    async def chat(self, req: ChatCompletionRequest, provider_model: str) -> ProviderResult:
        raise NotImplementedError

    async def stream(self, req: ChatCompletionRequest, provider_model: str) -> AsyncIterator[Dict[str, Any]]:
        raise ProviderError(self.name, "Provider does not support true streaming", 501, retryable=False)
        if False:
            yield {}

    async def health(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, status="configured")
