import asyncio
import time
from dataclasses import dataclass
from typing import Dict


@dataclass
class CircuitState:
    failures: int = 0
    state: str = "closed"
    opened_at: float = 0.0
    half_open_in_flight: bool = False


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_seconds: float = 30.0):
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_seconds = max(1.0, recovery_seconds)
        self._states: Dict[str, CircuitState] = {}
        self._lock = asyncio.Lock()

    async def allow(self, provider: str) -> bool:
        async with self._lock:
            state = self._states.setdefault(provider, CircuitState())
            if state.state == "closed":
                return True
            if state.state == "open":
                if time.monotonic() - state.opened_at < self.recovery_seconds:
                    return False
                state.state = "half-open"
                state.half_open_in_flight = False
            if state.state == "half-open":
                if state.half_open_in_flight:
                    return False
                state.half_open_in_flight = True
                return True
            return False

    async def success(self, provider: str) -> None:
        async with self._lock:
            self._states[provider] = CircuitState()

    async def failure(self, provider: str) -> None:
        async with self._lock:
            state = self._states.setdefault(provider, CircuitState())
            state.failures += 1
            if state.state == "half-open" or state.failures >= self.failure_threshold:
                state.state = "open"
                state.opened_at = time.monotonic()
                state.half_open_in_flight = False

    async def snapshot(self):
        async with self._lock:
            now = time.monotonic()
            return {
                name: {
                    "state": s.state,
                    "failures": s.failures,
                    "retry_in_seconds": round(max(0.0, self.recovery_seconds - (now - s.opened_at)), 2)
                    if s.state == "open" else 0.0,
                }
                for name, s in self._states.items()
            }
