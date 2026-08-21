from typing import Dict, List
import httpx


class TavilyClient:
    def __init__(self, api_key: str, timeout: float = 10.0, max_results: int = 5):
        self.api_key = api_key
        self.timeout = timeout
        self.max_results = max_results

    async def search(self, query: str) -> List[Dict[str, str]]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post('https://api.tavily.com/search', json={
                'api_key': self.api_key,
                'query': query,
                'max_results': self.max_results,
                'include_answer': False,
            })
        r.raise_for_status()
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
