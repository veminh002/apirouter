"""Small zero-cost realtime intent helper.

This does not perform any external web search. It only marks requests that are
likely to need fresh information so the ChatGPT Web primary provider can be
explicitly instructed to use its native browsing/search capability when that
capability is available in the current ChatGPT Web session.
"""

from dataclasses import dataclass
import re
from typing import Any


_REALTIME_PATTERNS = [
    r"\b(?:today|tonight|now|currently|current|right now|latest|recent|newest)\b",
    r"\b(?:hôm nay|hien tai|hiện tại|bây giờ|luc nay|lúc này|mới nhất|moi nhat|gần đây|gan day|vừa mới|vua moi)\b",
    r"\b(?:tin tức|tin tuc|news|breaking|update|updates)\b",
    r"\b(?:giá|gia)\s+(?:hiện tại|hien tai|today|now)\b",
    r"\b(?:thời tiết|thoi tiet|weather)\b",
    r"\b(?:tỷ giá|ty gia|exchange rate|forex)\b",
    r"\b(?:kết quả|ket qua|score|scores|standings|ranking|rankings)\b",
    r"\b(?:live|trực tiếp|truc tiep)\b",
    r"\b(?:search|web search|tìm trên web|tim tren web|tra cứu trên web|tra cuu tren web)\b",
    r"\b(?:as of|this week|this month|today's|tomorrow|yesterday)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _REALTIME_PATTERNS]


@dataclass(frozen=True)
class RealtimeDecision:
    needs_fresh_info: bool
    reason: str


def _text_from_message(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content or "")


def detect_realtime(messages: list[Any]) -> RealtimeDecision:
    """Only the current (last user) turn decides intent - not the whole
    history. Scanning every past message meant a keyword anywhere in prior
    turns (or client-injected memory/context riding along as a user
    message) could mark an unrelated new question as needing search."""
    last_user_text = ""
    for m in reversed(messages):
        if getattr(m, "role", None) == "user":
            last_user_text = _text_from_message(m)
            break
    for pattern in _COMPILED:
        if pattern.search(last_user_text):
            return RealtimeDecision(True, "keyword_or_phrase")
    return RealtimeDecision(False, "no_realtime_signal")
