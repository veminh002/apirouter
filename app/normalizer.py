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

def flatten_for_chatgpt(messages):
    chunks = []
    for m in messages:
        role = m.role.upper()
        text = extract_text(m.content)
        chunks.append(f'[{role}]\n{text}')
    return '\n\n'.join(chunks)

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
