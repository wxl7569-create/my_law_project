"""统一认证接口的前端请求客户端。"""

# ═══════════════════════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import httpx

from core.config.settings import Config


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

BACKEND_API_URL = Config.BACKEND_API_URL
AUTH_API_BASE = f"{BACKEND_API_URL}/api/v1/auth"
_TIMEOUT = httpx.Timeout(10.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Internal Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _error_from_response(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except Exception:
        return resp.text or f"HTTP {resp.status_code}"
    detail = data.get("detail", data)
    if isinstance(detail, list):
        return "; ".join(str(item.get("msg", item)) for item in detail)
    return str(detail)


def _api_post(path: str, json_data: dict) -> dict:
    try:
        resp = httpx.post(
            f"{AUTH_API_BASE}{path}",
            json=json_data,
            timeout=_TIMEOUT,
            trust_env=False,
        )
        if resp.status_code == 200:
            return {"success": True, **resp.json()}
        return {"success": False, "error": _error_from_response(resp)}
    except httpx.ConnectError:
        return {"success": False, "error": f"无法连接后端服务：{BACKEND_API_URL}"}
    except httpx.TimeoutException:
        return {"success": False, "error": "认证服务响应超时"}
    except Exception as e:
        return {"success": False, "error": f"请求异常: {e}"}


def _api_get_with_token(path: str, token: str) -> dict:
    try:
        resp = httpx.get(
            f"{AUTH_API_BASE}{path}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=_TIMEOUT,
            trust_env=False,
        )
        if resp.status_code == 200:
            return {"success": True, **resp.json()}
        return {"success": False, "error": _error_from_response(resp)}
    except httpx.ConnectError:
        return {"success": False, "error": f"无法连接后端服务：{BACKEND_API_URL}"}
    except httpx.TimeoutException:
        return {"success": False, "error": "认证服务响应超时"}
    except Exception as e:
        return {"success": False, "error": f"请求异常: {e}"}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API Functions (Register, Login, Verify Token)
# ═══════════════════════════════════════════════════════════════════════════════

def register(name: str, phone: str, password: str) -> dict:
    return _api_post(
        "/register",
        {"name": name.strip(), "phone": phone.strip(), "password": password},
    )


def login(phone: str, password: str) -> dict:
    return _api_post("/login", {"phone": phone.strip(), "password": password})


def verify_token(token: str) -> dict:
    return _api_get_with_token("/me", token)
