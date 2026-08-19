from typing import Dict

from .openai_compatible import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    name = 'openrouter'
    base_url = 'https://openrouter.ai/api/v1/chat/completions'
    models_url = 'https://openrouter.ai/api/v1/models'

    def __init__(self, api_key, timeout, referer, title, web_search_mode='off', web_search_model=''):
        super().__init__(api_key, timeout, web_search_mode, web_search_model)
        self.referer, self.title = referer, title

    def _extra_headers(self) -> Dict[str, str]:
        return {'HTTP-Referer': self.referer, 'X-Title': self.title}

    def _search_tool(self):
        # Server tool: the model decides if/when to search, unlike the
        # deprecated ":online" suffix / "plugins": [{"id": "web"}], which
        # ran a paid search on every single request regardless of need.
        return {'type': 'openrouter:web_search'}
