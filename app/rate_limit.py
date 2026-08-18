import asyncio, time
from collections import defaultdict, deque
from fastapi import HTTPException

class RateLimiter:
    def __init__(self, per_minute: int, sweep_interval: int = 500):
        self.per_minute = max(1, per_minute)
        self.events = defaultdict(deque)
        self.lock = asyncio.Lock()
        # Only the requesting key's deque is trimmed on each check(). A key
        # that is checked once and never again (rotating API keys, random
        # anonymous callers) would otherwise sit in `events` forever and grow
        # memory unbounded over the process lifetime. Sweep all keys
        # periodically instead of on every call to keep this cheap.
        self.sweep_interval = max(1, sweep_interval)
        self._requests_since_sweep = 0

    async def check(self, key: str):
        now = time.monotonic()
        async with self.lock:
            q = self.events[key]
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                retry = max(1, int(60 - (now - q[0])))
                raise HTTPException(429, 'Rate limit exceeded', headers={'Retry-After': str(retry)})
            q.append(now)
            self._requests_since_sweep += 1
            if self._requests_since_sweep >= self.sweep_interval:
                self._requests_since_sweep = 0
                stale_keys = [k for k, dq in self.events.items() if not dq or now - dq[-1] >= 60]
                for stale_key in stale_keys:
                    del self.events[stale_key]
