import secrets
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from .runtime_config import EDITABLE_FIELDS, effective_aliases
from .state_holder import StateHolder

security = HTTPBasic()
STATIC_DIR = Path(__file__).parent / 'static'

SECRET_FIELDS = {'router9_api_key', 'nvidia_api_key', 'groq_api_key', 'openrouter_api_key', 'tokenrouter_api_key', 'tavily_api_key'}


def _mask(value: str) -> str:
    value = value or ''
    if len(value) <= 4:
        return '*' * len(value)
    return '*' * (len(value) - 4) + value[-4:]


class SettingsUpdate(BaseModel):
    router9_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    tokenrouter_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    enable_chatgpt: Optional[bool] = None
    enable_nvidia: Optional[bool] = None
    enable_groq: Optional[bool] = None
    enable_openrouter: Optional[bool] = None
    enable_tokenrouter: Optional[bool] = None
    enable_tavily: Optional[bool] = None


class AliasUpdate(BaseModel):
    name: str
    candidates: str


def build_admin_router(holder: StateHolder) -> APIRouter:
    router = APIRouter(prefix='/admin', tags=['admin'])

    def require_auth(credentials: HTTPBasicCredentials = Depends(security)):
        base = holder.env_settings
        if not base.admin_password:
            raise HTTPException(503, 'Admin dashboard disabled: set ADMIN_PASSWORD to enable it.')
        user_ok = secrets.compare_digest(credentials.username, base.admin_username)
        pass_ok = secrets.compare_digest(credentials.password, base.admin_password)
        if not (user_ok and pass_ok):
            raise HTTPException(401, 'Invalid admin credentials', headers={'WWW-Authenticate': 'Basic'})

    def snapshot():
        settings = holder.state.settings
        data = {name: getattr(settings, name) for name in EDITABLE_FIELDS if name != 'model_aliases'}
        for field in SECRET_FIELDS:
            data[field] = _mask(data[field])
        return {
            'settings': data,
            'aliases': effective_aliases(settings),
            'not_persisted_without_database': not holder.store.enabled,
            'providers': holder.state.registry.names(),
            'models': holder.state.policy.models(),
        }

    @router.get('', dependencies=[Depends(require_auth)])
    async def dashboard():
        return FileResponse(STATIC_DIR / 'admin.html')

    @router.get('/api/state', dependencies=[Depends(require_auth)])
    async def get_state():
        return snapshot()

    @router.post('/api/settings', dependencies=[Depends(require_auth)])
    async def update_settings(update: SettingsUpdate):
        overrides = update.model_dump(exclude_unset=True)
        if not overrides:
            return snapshot()
        try:
            await holder.apply(overrides)
        except RuntimeError as exc:
            raise HTTPException(400, str(exc))
        return snapshot()

    @router.post('/api/settings/reset', dependencies=[Depends(require_auth)])
    async def reset_settings():
        await holder.reset()
        return snapshot()

    @router.post('/api/aliases', dependencies=[Depends(require_auth)])
    async def upsert_alias(update: AliasUpdate):
        name = update.name.strip()
        candidates = update.candidates.strip()
        if not name or not candidates:
            raise HTTPException(400, 'name and candidates are both required')
        if not holder.state.settings.parse_candidates(candidates):
            raise HTTPException(400, "candidates must look like 'provider:model,provider:model'")
        aliases = dict(holder.overrides.get('model_aliases', {}))
        aliases[name] = candidates
        try:
            await holder.apply({'model_aliases': aliases})
        except RuntimeError as exc:
            raise HTTPException(400, str(exc))
        return snapshot()

    @router.delete('/api/aliases/{name}', dependencies=[Depends(require_auth)])
    async def delete_alias(name: str):
        aliases = dict(holder.overrides.get('model_aliases', {}))
        aliases[name] = ''  # tombstone: removes admin-added AND built-in aliases alike
        try:
            await holder.apply({'model_aliases': aliases})
        except RuntimeError as exc:
            raise HTTPException(400, str(exc))
        return snapshot()

    @router.post('/api/circuit-breaker/reset', dependencies=[Depends(require_auth)])
    async def reset_breaker(provider: Optional[str] = None):
        if provider:
            await holder.state.breaker.success(provider)
        else:
            for name in holder.state.registry.names():
                await holder.state.breaker.success(name)
        return await holder.state.breaker.snapshot()

    return router
