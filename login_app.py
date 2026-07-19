"""旧认证服务兼容入口。

当前主后端入口是 ``python -m core.api.server``。本文件保留
``python login_app.py --serve`` 的旧启动方式，并在 8002 端口挂载相同认证路由。
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════
# 导入标准库
# ══════════════════════════════════════════════════════════════

import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

# ══════════════════════════════════════════════════════════════
# 导入核心模块
# ══════════════════════════════════════════════════════════════

from core.api.auth_routes import get_db, init_auth_db, router
from core.config.settings import Config


# ══════════════════════════════════════════════════════════════
# 应用生命周期
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_auth_db()
    yield


# ══════════════════════════════════════════════════════════════
# FastAPI 应用初始化
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="PBL Auth Service",
    description="Standalone compatibility service for login/register/token verification.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


# ══════════════════════════════════════════════════════════════
# 测试函数
# ══════════════════════════════════════════════════════════════

def run_tests() -> bool:
    init_auth_db()
    client = TestClient(app)
    phone = "13800000001"
    password = "123456"
    name = "测试用户"

    conn = get_db()
    conn.execute("DELETE FROM users WHERE phone = ?", (phone,))
    conn.commit()
    conn.close()

    checks = [client.get("/api/v1/auth/health").status_code == 200]
    resp = client.post(
        "/api/v1/auth/register",
        json={"name": name, "phone": phone, "password": password},
    )
    checks.append(resp.status_code == 200)
    token = resp.json().get("token") if resp.status_code == 200 else ""
    checks.append(
        client.post(
            "/api/v1/auth/login",
            json={"phone": phone, "password": password},
        ).status_code
        == 200
    )
    checks.append(
        client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        ).status_code
        == 200
    )

    conn = get_db()
    conn.execute("DELETE FROM users WHERE phone = ?", (phone,))
    conn.commit()
    conn.close()
    return all(checks)


# ══════════════════════════════════════════════════════════════
# 启动入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--serve"
    if mode == "--test-only":
        ok = run_tests()
        print("auth self-test passed" if ok else "auth self-test failed")
        sys.exit(0 if ok else 1)

    print("Auth compatibility service: http://127.0.0.1:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
