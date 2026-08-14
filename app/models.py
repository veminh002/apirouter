from typing import Any, List, Optional, Union, Dict
from pydantic import BaseModel, Field, ConfigDict

Content = Union[str, List[Dict[str, Any]]]

class Message(BaseModel):
    model_config = ConfigDict(extra='allow')
    role: str
    content: Content
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra='allow')
    model: str = 'gpt-4o-mini'
    messages: List[Message] = Field(min_length=1)
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None

class ProviderResult(BaseModel):
    provider: str
    response: Dict[str, Any]
    model: str
    latency_ms: int
