import time, uuid

def now_ts():
    return int(time.time())

def normalize_openai_response(data, requested_model: str, provider: str):
    if isinstance(data, dict) and data.get('object') == 'chat.completion':
        data.setdefault('model', requested_model)
        data.setdefault('provider', provider)
        return data
    choices = data.get('choices') or [] if isinstance(data, dict) else []
    return {
        'id': f'chatcmpl-{uuid.uuid4().hex}',
        'object': 'chat.completion',
        'created': now_ts(),
        'model': requested_model,
        'choices': choices,
        'usage': data.get('usage', {}) if isinstance(data, dict) else {},
        'provider': provider,
    }

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                parts.append(str(item.get('text', '')))
        return '\n'.join(parts)
    return str(content)

def to_responses_input(messages):
    """Convert chat messages (possibly multimodal) into Responses API `input` items.

    Keeps each message as its own input item and preserves image_url content
    blocks as `input_image` items, instead of collapsing everything into one
    text-only blob (which would silently drop any images the caller sent).
    Each Responses input item looks like:
      {"role": "user"|"assistant", "content": [{"type": "input_text"|"input_image", ...}]}
    """
    valid_roles = {'user', 'assistant', 'developer'}
    items = []
    for m in messages:
        role = m.role if m.role in valid_roles else 'user'
        content = m.content
        blocks = []
        if isinstance(content, str):
            if content:
                blocks.append({'type': 'input_text', 'text': content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get('type')
                if ptype == 'text':
                    text = part.get('text', '')
                    if text:
                        blocks.append({'type': 'input_text', 'text': text})
                elif ptype == 'image_url':
                    # Standard OpenAI chat-completions shape:
                    # {"type": "image_url", "image_url": {"url": "..."}}
                    # (a bare string value is also tolerated defensively.)
                    image_url = part.get('image_url')
                    url = image_url.get('url') if isinstance(image_url, dict) else image_url
                    if url:
                        block = {'type': 'input_image', 'image_url': url}
                        detail = image_url.get('detail') if isinstance(image_url, dict) else None
                        if detail:
                            block['detail'] = detail
                        blocks.append(block)
                elif ptype in ('input_text', 'input_image'):
                    # Already Responses-native; pass through untouched.
                    blocks.append(part)
        if blocks:
            items.append({'role': role, 'content': blocks})
    return items

def split_system_instructions(messages):
    """Pull system-role messages out of the conversation.

    The ChatGPT Responses API has a dedicated `instructions` field for
    system-level guidance, distinct from the conversational `input`. Before
    this, every message (including role=system) was flattened together into
    one text blob under `input`, and `instructions` was a hardcoded generic
    string - so a caller's actual system prompt was demoted to a "[SYSTEM]"
    label buried in user-turn text instead of being treated as instructions.

    Returns (instructions, remaining_messages): `instructions` joins every
    system message's text in order (empty string if none); `remaining_messages`
    is every other message, unchanged and in original order.
    """
    system_chunks = []
    remaining = []
    for m in messages:
        if m.role == 'system':
            text = extract_text(m.content)
            if text:
                system_chunks.append(text)
        else:
            remaining.append(m)
    return '\n\n'.join(system_chunks), remaining
