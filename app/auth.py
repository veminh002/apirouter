import secrets
from fastapi import Header, HTTPException
from typing import Optional
from .config import Settings

def validate_auth(authorization: Optional[str], settings: Settings):
    # Check the caller's token shape before revealing server misconfiguration.
    # Checking ROUTER9_API_KEY first meant an unauthenticated caller with no
    # Bearer header at all could distinguish "server misconfigured" (500) from
    # "server configured, my token is wrong" (401/403) without ever proving
    # who they are.
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(401, 'Missing Bearer token')
    if not settings.router9_api_key:
        raise HTTPException(500, 'ROUTER9_API_KEY is not configured')
    token = authorization[7:].strip()
    if not secrets.compare_digest(token, settings.router9_api_key):
        raise HTTPException(403, 'Invalid ROUTER9_API_KEY')
