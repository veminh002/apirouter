import asyncio, time
from collections import defaultdict, deque
from fastapi import HTTPException

class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = max(1, per_minute)
        self.events = defaultdict(deque)
        self.lock = asyncio.Lock()

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
