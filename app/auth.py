import secrets
from fastapi import Header, HTTPException
from typing import Optional
from .config import Settings

def validate_auth(authorization: Optional[str], settings: Settings):
    if not settings.router9_api_key:
        raise HTTPException(500, 'ROUTER9_API_KEY is not configured')
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(401, 'Missing Bearer token')
    token = authorization[7:].strip()
    if not secrets.compare_digest(token, settings.router9_api_key):
        raise HTTPException(403, 'Invalid ROUTER9_API_KEY')
