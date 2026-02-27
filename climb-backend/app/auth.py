"""
Clerk JWT verification for FastAPI.

Verifies JWTs using Clerk's JWKS (JSON Web Key Set) endpoint.
Keys are cached to avoid hitting Clerk on every request.
"""
import time
import httpx
import jwt as pyjwt
from jwt.algorithms import RSAAlgorithm
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from fastapi import Depends, HTTPException, Request, Query
from functools import lru_cache
from app.config import settings
from app.services import services

# Cache JWKS keys for 1 hour
_jwks_cache: dict = {"keys": None, "fetched_at": 0}
JWKS_CACHE_TTL = 3600  # seconds


def _get_clerk_jwks_url() -> str:
    """Construct the JWKS URL from Clerk's issuer."""
    # Your Clerk frontend API domain, e.g. "https://your-app.clerk.accounts.dev"
    return f"{settings.CLERK_ISSUER}/.well-known/jwks.json"


def _fetch_jwks() -> list[dict]:
    """Fetch and cache JWKS from Clerk."""
    now = time.time()
    if _jwks_cache["keys"] and (now - _jwks_cache["fetched_at"]) < JWKS_CACHE_TTL:
        return _jwks_cache["keys"]

    resp = httpx.get(_get_clerk_jwks_url(), timeout=10)
    resp.raise_for_status()
    keys = resp.json()["keys"]
    _jwks_cache["keys"] = keys
    _jwks_cache["fetched_at"] = now
    return keys


def _verify_clerk_token(token: str) -> dict:
    """
    Verify a Clerk-issued JWT and return the payload.
    
    Clerk JWTs use RS256 and include 'sub' (user ID), 'email', etc.
    """
    jwks = _fetch_jwks()

    # Decode the JWT header to find the key ID
    unverified_header = pyjwt.get_unverified_header(token)
    kid = unverified_header.get("kid")

    # Find the matching public key
    matching_key = None
    for key_data in jwks:
        if key_data["kid"] == kid:
            matching_key = RSAAlgorithm.from_jwk(key_data)
            if not isinstance(matching_key, RSAPublicKey):
                raise ValueError("Expected RSA public key from JWKS")
            break

    if not matching_key:
        raise ValueError("No matching key found in JWKS")

    # Verify and decode
    payload = pyjwt.decode(
        token,
        matching_key,
        algorithms=["RS256"],
        issuer=settings.CLERK_ISSUER,
        options={"verify_aud": False},
    )
    return payload


async def get_current_user(request: Request) -> dict | None:
    """
    Extract and verify Clerk JWT from Authorization header.
    Returns user info dict or None for anonymous requests.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header.split(" ", 1)[1]
    try:
        payload = _verify_clerk_token(token)
        return {
            "user_id": payload["sub"],
            "email": payload.get("email"),
            "name": payload.get("name"),
        }
    except Exception as e:
        print(f"Auth verification failed: {e}")
        return None


async def require_auth(user: dict | None = Depends(get_current_user)) -> dict:
    """Require user authentication to hit certain endpoints."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    services.ensure_user_exists(user["user_id"], user.get("email"), user.get("name"))
    return user

async def sync_auth(user: dict | None = Depends(get_current_user)) -> None:
    """Synchronize user authentication with the database if we can."""
    print("User: ")
    if user is not None:
        print(user["user_id"], user.get("email"), user.get("name"))
        services.ensure_user_exists(user["user_id"], user.get("email"), user.get("name"))