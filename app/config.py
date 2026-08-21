from functools import lru_cache
from typing import Dict, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', case_sensitive=False)

    router9_api_key: str = ''

    # /admin dashboard (Basic Auth). Empty password keeps the dashboard
    # disabled - it must be turned on explicitly, since it can edit API
    # keys and provider routing at runtime.
    admin_username: str = 'admin'
    admin_password: str = ''

    tavily_api_key: str = ''
    # Chạy trước router.complete()/stream(), kết quả chèn vào messages cho
    # MỌI provider (chatgpt, nvidia, tokenrouter, groq, openrouter) dùng
    # chung - thay cho từng provider tự search riêng lẻ. off|auto|always,
    # cùng semantics với *_web_search_mode.
    tavily_search_mode: str = 'auto'
    tavily_max_results: int = 5

    chatgpt_refresh_token: str = ''
    chatgpt_access_token: str = ''
    chatgpt_access_token_expires_in: int = 0
    chatgpt_client_id: str = ''
    chatgpt_redirect_uri: str = ''
    chatgpt_auth_url: str = 'https://auth.openai.com/oauth/token'
    chatgpt_account_id: str = ''
    chatgpt_id_token: str = ''
    chatgpt_responses_url: str = 'https://chatgpt.com/backend-api/codex/responses'
    chatgpt_originator: str = 'codex_cli_rs'
    chatgpt_version: str = '0.144.1'
    chatgpt_oauth_scope: str = 'openid profile email offline_access api.connectors.read api.connectors.invoke'
    chatgpt_token_state_file: str = ''
    database_url: str = ''
    chatgpt_token_encryption_key: str = ''
    chatgpt_keepalive_hours: float = 6.0
    chatgpt_web_search_mode: str = 'off'
    chatgpt_web_search_instruction: str = (
        'When this request depends on information that may have changed recently, '
        'use ChatGPT Web native search/browsing when available in this conversation. '
        'Prefer current web-verified information over memory, and include source links or citations when the platform provides them. '
        'Never invent a search result or citation.'
    )
    groq_api_key: str = ''
    openrouter_api_key: str = ''
    tokenrouter_api_key: str = ''
    nvidia_api_key: str = ''

    # NVIDIA NIM (build.nvidia.com) chỉ là inference thuần, không model nào
    # có tool duyệt web/tìm kiếm gốc đi kèm - khác ChatGPT/Groq. Off theo
    # mặc định như TokenRouter; chỉ bật auto/always nếu tự trỏ
    # nvidia_web_search_model sang một model NIM cụ thể có hỗ trợ search.
    nvidia_web_search_mode: str = 'off'
    nvidia_web_search_model: str = ''

    # Groq's "compound" models run their own built-in web-search tool
    # server-side, so search here just means swapping to that model.
    # Off by default: Tavily now provides web context centrally for all
    # providers, so no single provider needs to search on its own.
    groq_web_search_mode: str = 'off'
    groq_web_search_model: str = 'groq/compound-mini'

    # OpenRouter exposes web search as a server tool, but unlike ChatGPT/Groq
    # it is billed per search call — off by default to keep the router fully
    # free out of the box. Set to 'auto'/'always' only if you accept that cost.
    openrouter_web_search_mode: str = 'off'
    openrouter_web_search_model: str = ''

    # TokenRouter's upstream search support depends on which model it
    # proxies to; off by default until a search-capable model is set.
    tokenrouter_web_search_mode: str = 'off'
    tokenrouter_web_search_model: str = ''

    request_timeout: float = 45.0
    provider_timeout: float = 30.0
    max_retries: int = 1
    rate_limit_per_minute: int = 60
    max_concurrent_requests: int = 8

    enable_chatgpt: bool = True
    enable_nvidia: bool = True
    enable_groq: bool = True
    enable_openrouter: bool = True
    enable_tokenrouter: bool = True
    enable_tavily: bool = True

    circuit_failure_threshold: int = 5
    circuit_recovery_seconds: float = 30.0

    openrouter_referer: str = 'https://github.com/9router/9router'
    openrouter_title: str = '9Router v3'

    # Alias candidate chains: model đứng đầu là ưu tiên cao nhất.
    #
    # groq:llama-3.1-8b-instant (giá trị cũ) đã bị Groq KHAI TỬ (shutdown
    # 16/08/2026 - console.groq.com/docs/deprecations). Groq khuyến nghị
    # thay bằng openai/gpt-oss-20b - đây cũng chính là model lananh-main
    # (GROQ_MODEL) đang dùng, đã verify hoạt động.
    #
    # openrouter:openai/gpt-4o và openrouter:google/gemini-2.5-flash-lite
    # (giá trị cũ) đều là model TRẢ PHÍ trên OpenRouter. Thay bằng đúng 2
    # model free lananh-main đang dùng và đã verify còn tồn tại/free qua
    # openrouter.ai/collections/free-models tại thời điểm sửa (19/08/2026):
    # nvidia/nemotron-3-super-120b-a12b:free (OPENROUTER_MODEL của lananh)
    # và google/gemma-4-26b-a4b-it:free (OPENROUTER_VISION_MODEL của lananh).
    #
    # tokenrouter:qwen/qwen3.8-max-free giữ nguyên - KHÔNG verify được vì
    # TokenRouter không có danh mục model công khai để tra cứu độc lập.
    #
    # chatgpt:gpt-5.6-terra / gpt-5.6-luna đã verify ĐÚNG và hiện hành qua
    # developers.openai.com/codex/models (gpt-5.4/gpt-5.4-mini retire khỏi
    # Codex-với-ChatGPT-login ngày 31/08/2026, thay bằng đúng 2 tên này).
    #
    # nvidia:nvidia/nemotron-3-ultra-550b-a55b là tầng mới, chèn ngay sau
    # chatgpt - Free Endpoint xác nhận trên build.nvidia.com (20/08/2026),
    # không phụ phí tìm kiếm hay token. Dùng cho cả 4 alias (kể cả mini):
    # mistralai/mistral-nemotron ban đầu dùng cho alias mini nhưng model nhỏ
    # đó tiếng Việt kém, hay bịa persona/dữ liệu cho câu hỏi đơn giản -
    # nemotron-3-ultra lớn hơn, đa ngôn ngữ ổn định hơn.
    alias_gpt_4o: str = 'chatgpt:gpt-5.6-terra,nvidia:nvidia/nemotron-3-ultra-550b-a55b,tokenrouter:qwen/qwen3.8-max-free,groq:openai/gpt-oss-120b,openrouter:nvidia/nemotron-3-super-120b-a12b:free'
    alias_gpt_4o_mini: str = 'chatgpt:gpt-5.6-luna,nvidia:nvidia/nemotron-3-ultra-550b-a55b,tokenrouter:qwen/qwen3.8-max-free,groq:openai/gpt-oss-20b,openrouter:google/gemma-4-26b-a4b-it:free'
    alias_gpt_4_turbo: str = 'chatgpt:gpt-5.6-terra,nvidia:nvidia/nemotron-3-ultra-550b-a55b,tokenrouter:qwen/qwen3.8-max-free,groq:openai/gpt-oss-120b,openrouter:nvidia/nemotron-3-super-120b-a12b:free'
    alias_gpt_3_5_turbo: str = 'chatgpt:gpt-5.6-luna,nvidia:nvidia/nemotron-3-ultra-550b-a55b,tokenrouter:qwen/qwen3.8-max-free,groq:openai/gpt-oss-20b,openrouter:google/gemma-4-26b-a4b-it:free'

    # Admin-editable aliases (see admin.py /admin/api/aliases), layered on
    # top of the 4 defaults above. Key = alias name (any string, matched
    # case-insensitively), value = candidate chain string, or '' to delete
    # (including deleting one of the 4 built-in names above).
    model_aliases: Dict[str, str] = Field(default_factory=dict)

    @field_validator('enable_chatgpt', 'enable_nvidia', 'enable_groq', 'enable_openrouter', 'enable_tokenrouter', 'enable_tavily', mode='before')
    @classmethod
    def normalize_boolean(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            aliases = {'fasle': 'false', 'ture': 'true', 'flase': 'false', 'treu': 'true'}
            value = aliases.get(normalized, normalized)
        return value

    @property
    def configured_providers(self) -> List[str]:
        providers = []
        if self.enable_chatgpt:
            providers.append('chatgpt')
        if self.enable_nvidia and self.nvidia_api_key:
            providers.append('nvidia')
        if self.enable_tokenrouter and self.tokenrouter_api_key:
            providers.append('tokenrouter')
        if self.enable_groq and self.groq_api_key:
            providers.append('groq')
        if self.enable_openrouter and self.openrouter_api_key:
            providers.append('openrouter')
        return providers

    @staticmethod
    def parse_candidates(value: str):
        candidates = []
        for item in (value or '').split(','):
            item = item.strip()
            if not item or ':' not in item:
                continue
            provider, model = item.split(':', 1)
            if provider.strip() and model.strip():
                candidates.append((provider.strip(), model.strip()))
        return candidates


@lru_cache
def get_settings():
    return Settings()
