import asyncio
import time
from collections import Counter
from typing import Dict


class Metrics:
    def __init__(self):
        self._lock = asyncio.Lock()
        self.requests = Counter()
        self.provider_requests = Counter()
        self.provider_errors = Counter()
        self.provider_latency_ms = Counter()
        self.provider_latency_count = Counter()
        self.fallbacks = 0

    async def request(self, status: str):
        async with self._lock:
            self.requests[status] += 1

    async def provider_started(self, provider: str):
        async with self._lock:
            self.provider_requests[provider] += 1

    async def provider_finished(self, provider: str, latency_ms: int, error: bool = False):
        async with self._lock:
            self.provider_latency_ms[provider] += latency_ms
            self.provider_latency_count[provider] += 1
            if error:
                self.provider_errors[provider] += 1

    async def fallback(self):
        async with self._lock:
            self.fallbacks += 1

    async def snapshot(self) -> Dict:
        async with self._lock:
            avg = {
                p: round(self.provider_latency_ms[p] / max(1, self.provider_latency_count[p]), 2)
                for p in self.provider_latency_count
            }
            return {
                "requests": dict(self.requests),
                "providers": {
                    p: {
                        "requests": self.provider_requests[p],
                        "errors": self.provider_errors[p],
                        "avg_latency_ms": avg.get(p, 0),
                    }
                    for p in set(self.provider_requests) | set(self.provider_errors)
                },
                "fallbacks": self.fallbacks,
                "timestamp": int(time.time()),
            }

    async def prometheus(self) -> str:
        data = await self.snapshot()
        lines = [
            "# HELP router_requests_total Total router requests by status",
            "# TYPE router_requests_total counter",
        ]
        for status, value in data["requests"].items():
            lines.append(f'router_requests_total{{status="{status}"}} {value}')
        lines += [
            "# HELP router_fallbacks_total Total fallback events",
            "# TYPE router_fallbacks_total counter",
            f"router_fallbacks_total {data['fallbacks']}",
            "# HELP router_provider_requests_total Provider attempts",
            "# TYPE router_provider_requests_total counter",
        ]
        for provider, values in data["providers"].items():
            lines.append(f'router_provider_requests_total{{provider="{provider}"}} {values["requests"]}')
            lines.append(f'router_provider_errors_total{{provider="{provider}"}} {values["errors"]}')
            lines.append(f'router_provider_avg_latency_ms{{provider="{provider}"}} {values["avg_latency_ms"]}')
        return "\n".join(lines) + "\n"
