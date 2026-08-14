from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ModelAlias:
    name: str
    providers: List[Tuple[str, str]]


class RoutingPolicy:
    def __init__(self, aliases: Dict[str, ModelAlias]):
        self.aliases = {k.lower(): v for k, v in aliases.items()}

    def resolve(self, requested_model: str) -> ModelAlias:
        model = requested_model.lower()
        if model in self.aliases:
            return self.aliases[model]
        # Explicit provider:model syntax is supported as an escape hatch.
        if ":" in requested_model:
            provider, provider_model = requested_model.split(":", 1)
            return ModelAlias(requested_model, [(provider, provider_model)])
        # Unknown models go to OpenRouter first because it can expose a broad model catalog.
        return ModelAlias(requested_model, [("openrouter", requested_model), ("groq", requested_model)])

    def models(self):
        return list(self.aliases.keys())
