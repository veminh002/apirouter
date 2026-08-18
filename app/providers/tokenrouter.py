from .openai_compatible import OpenAICompatibleProvider


class TokenRouterProvider(OpenAICompatibleProvider):
    name = 'tokenrouter'
    base_url = 'https://api.tokenrouter.com/v1/chat/completions'
    models_url = 'https://api.tokenrouter.com/v1/models'
