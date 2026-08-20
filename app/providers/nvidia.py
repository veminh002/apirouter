from .openai_compatible import OpenAICompatibleProvider


class NvidiaProvider(OpenAICompatibleProvider):
    name = 'nvidia'
    base_url = 'https://integrate.api.nvidia.com/v1/chat/completions'
    models_url = 'https://integrate.api.nvidia.com/v1/models'
