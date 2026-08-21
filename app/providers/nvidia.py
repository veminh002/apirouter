from .openai_compatible import OpenAICompatibleProvider


class NvidiaProvider(OpenAICompatibleProvider):
    name = 'nvidia'
    base_url = 'https://integrate.api.nvidia.com/v1/chat/completions'
    models_url = 'https://integrate.api.nvidia.com/v1/models'

    def _payload(self, req, provider_model, stream=False):
        payload = super()._payload(req, provider_model, stream)
        # Reasoning-hybrid NIM models (nemotron-3-ultra, etc.) default to
        # thinking mode on and leak raw chain-of-thought into `content`
        # instead of a clean answer unless this is explicitly turned off.
        # Non-reasoning models (mistral-nemotron) just ignore the unused
        # template kwarg. setdefault so a caller's own value still wins.
        payload.setdefault('chat_template_kwargs', {}).setdefault('enable_thinking', False)
        return payload
