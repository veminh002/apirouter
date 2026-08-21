from typing import Dict, List
import httpx


class TavilyClient:
    def __init__(self, api_key: str, timeout: float = 10.0, max_results: int = 5):
        self.api_key = api_key
        self.timeout = timeout
        self.max_results = max_results

    async def search(self, query: str) -> List[Dict[str, str]]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                'https://api.tavily.com/search',
                headers={'Authorization': f'Bearer {self.api_key}'},
                json={
                    'query': query,
                    'max_results': self.max_results,
                    'include_answer': False,
                },
            )
        if r.status_code >= 400:
            raise httpx.HTTPStatusError(f'{r.status_code} {r.text[:300]}', request=r.request, response=r)
        return [
            {'title': item.get('title', ''), 'url': item.get('url', ''), 'content': item.get('content', '')}
            for item in r.json().get('results', [])
        ]


def format_search_context(query: str, results: List[Dict[str, str]]) -> str:
    if not results:
        return ''
    lines = [f'Kết quả tìm kiếm web cho: "{query}"', '']
    for i, item in enumerate(results, 1):
        lines.append(f'[{i}] {item["title"]} ({item["url"]})')
        if item['content']:
            lines.append(item['content'])
        lines.append('')
    lines.append('Dùng thông tin trên để trả lời, trích nguồn [số] khi phù hợp. Nếu không đủ thông tin, nói rõ là không chắc.')
    return '\n'.join(lines)


def extract_last_user_text(messages) -> str:
    for m in reversed(messages):
        if m.role == 'user':
            content = m.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return '\n'.join(p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text')
    return ''


def inject_context(messages, context: str):
    """Prepend search context into the last user message's own content,
    instead of inserting a separate system message mid-conversation.

    Some models (Mistral, others with strict chat templates) reject a
    system-role message that appears after an assistant turn - only the
    very first message may be system. Folding the context into the
    existing user turn avoids introducing any new role ordering at all,
    so it works the same way across every provider/model."""
    messages = list(messages)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == 'user':
            content = messages[i].content
            if isinstance(content, str):
                new_content = f'{context}\n\n{content}'
            elif isinstance(content, list):
                new_content = [{'type': 'text', 'text': context}, *content]
            else:
                return messages
            messages[i] = messages[i].model_copy(update={'content': new_content})
            return messages
    return messages
