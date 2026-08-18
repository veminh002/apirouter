from typing import Dict

from .openai_compatible import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    name = 'openrouter'
    base_url = 'https://openrouter.ai/api/v1/chat/completions'
    models_url = 'https://openrouter.ai/api/v1/models'

    def __init__(self, api_key, timeout, referer, title):
        super().__init__(api_key, timeout)
        self.referer, self.title = referer, title

    def _extra_headers(self) -> Dict[str, str]:
        return {'HTTP-Referer': self.referer, 'X-Title': self.title}
