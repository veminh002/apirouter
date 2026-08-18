from .openai_compatible import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    name = 'groq'
    base_url = 'https://api.groq.com/openai/v1/chat/completions'
    models_url = 'https://api.groq.com/openai/v1/models'
