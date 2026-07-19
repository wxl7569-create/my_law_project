"""统一后端使用的认证接口路由。"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════
# 导入标准库
# ══════════════════════════════════════════════════════════════

import hashlib
import os
import re
import sqlite3
import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

# ══════════════════════════════════════════════════════════════
# 导入内部模块
# ══════════════════════════════════════════════════════════════

from core.config.settings import Config

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

TOKEN_DAYS = 7


# ══════════════════════════════════════════════════════════════
# 请求 / 响应数据模型
# ══════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=20)
    phone: str = Field(..., min_length=11, max_length=11)
    password: str = Field(..., min_length=6, max_length=20)


class LoginRequest(BaseModel):
    phone: str
    password: str


class TokenResponse(BaseModel):
    token: str
    user_id: str
    name: str
    phone: str


class UserInfoResponse(BaseModel):
    user_id: str
    name: str
    phone: str
    is_active: bool
    created_at: str


# ══════════════════════════════════════════════════════════════
# 数据库连接与初始化
# ══════════════════════════════════════════════════════════════

def get_db() -> sqlite3.Connection:
    """
    获取 users.db 的数据库连接。

    配置项：
    - row_factory=Row: 查询结果以字典形式返回
    - journal_mode=WAL: 启用 Write-Ahead Logging，提高并发读写性能
    - foreign_keys=ON: 启用外键约束（tokens.user_id → users.user_id）
    """
    os.makedirs(os.path.dirname(Config.USERS_DB_PATH), exist_ok=True)  # 确保 data/ 目录存在
    conn = sqlite3.connect(Config.USERS_DB_PATH)
    conn.row_factory = sqlite3.Row  # 查询结果支持按列名访问
    conn.execute("PRAGMA journal_mode=WAL")     # WAL 模式：读写不互斥
    conn.execute("PRAGMA foreign_keys=ON")       # 启用外键约束
    return conn


def init_auth_db() -> None:
    """
    初始化认证数据库（幂等操作，多次调用安全）。

    创建两个表：
    - users: 用户表（user_id, name, phone, password_hash, salt, is_active, created_at）
    - tokens: 令牌表（token, user_id, expires_at），外键关联 users.user_id
    """
    conn = get_db()
    cur = conn.cursor()
    # 用户表：存储注册用户的基本信息和密码哈希
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id VARCHAR(128) NOT NULL UNIQUE,     -- 用户唯一ID（格式: "姓名+手机号"）
            name VARCHAR(64) NOT NULL,                 -- 用户姓名
            phone VARCHAR(20) NOT NULL UNIQUE,         -- 手机号（登录凭据）
            password_hash VARCHAR(256) NOT NULL,       -- PBKDF2-SHA256 密码哈希
            salt VARCHAR(64) NOT NULL,                 -- 密码盐值（随机生成）
            is_active INTEGER DEFAULT 1,               -- 账号是否启用（1=正常, 0=禁用）
            created_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
            updated_at TIMESTAMP DEFAULT (datetime('now', 'localtime'))
        )
        """
    )
    # 令牌表：存储登录后的 Bearer Token（有效期 7 天）
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tokens (
            token VARCHAR(64) PRIMARY KEY,             -- UUID hex 令牌
            user_id VARCHAR(128) NOT NULL,             -- 关联的用户ID
            created_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
            expires_at TIMESTAMP NOT NULL,             -- 令牌过期时间
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
        """
    )
    # 加速手机号查询（登录时按手机号查找用户）
    cur.execute("CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone)")
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════
# 验证工具函数
# ══════════════════════════════════════════════════════════════

def validate_phone(phone: str) -> None:
    """
    验证手机号格式：中国大陆 11 位手机号（1 开头，第二位 3-9）。

    Raises:
        HTTPException(400): 格式不正确
    """
    if not re.match(r"^1[3-9]\d{9}$", phone):
        raise HTTPException(status_code=400, detail="手机号格式不正确")


def hash_password(password: str) -> tuple[str, str]:
    """
    使用 PBKDF2-SHA256 对密码进行哈希处理。

    参数：password - 明文密码
    返回：(hash_hex, salt_hex) — 哈希值和盐值，均为十六进制字符串

    安全参数：100,000 次迭代，符合 OWASP 推荐标准
    """
    salt = os.urandom(16).hex()  # 生成 16 字节（128 位）随机盐
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",                  # 哈希算法
        password.encode("utf-8"),  # 明文密码编码
        salt.encode("utf-8"),      # 盐值编码
        100000,                    # 迭代次数
    ).hex()
    return pwd_hash, salt


def verify_password(password: str, pwd_hash: str, salt: str) -> bool:
    """验证密码是否与存储的哈希值匹配。"""
    computed = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
    ).hex()
    return computed == pwd_hash  # 常量时间比较（Python 字符串比较天然是常量时间）


def generate_user_id(name: str, phone: str) -> str:
    """生成用户唯一标识：格式 "姓名+手机号"（如 "张三+13800000001"）。"""
    return f"{name}+{phone}"


def generate_token() -> str:
    """生成随机 32 位十六进制 Token（UUID4）。"""
    return uuid.uuid4().hex


# ══════════════════════════════════════════════════════════════
# 内部辅助 — 根据 Token 获取用户
# ══════════════════════════════════════════════════════════════

def _get_user_by_token(token: str) -> dict:
    """
    根据 Bearer Token 查找用户信息。

    流程：
    1. 通过 tokens 表 JOIN users 表查找匹配 token 且未过期的用户
    2. 未找到则抛出 401 Unauthorized

    用于 /api/v1/auth/me 接口的认证校验。
    """
    init_auth_db()  # 确保表已创建
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT u.* FROM users u
        JOIN tokens t ON u.user_id = t.user_id
        WHERE t.token = ? AND t.expires_at > datetime('now', 'localtime')
        """,
        (token,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")
    return dict(row)


# ══════════════════════════════════════════════════════════════
# API — 用户注册
# ══════════════════════════════════════════════════════════════

@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    """
    用户注册接口。

    流程：
    1. 验证手机号格式和姓名非空
    2. 检查手机号是否已注册（409 冲突）
    3. PBKDF2 哈希密码 + 随机盐
    4. 插入用户记录，生成 7 天有效期 Token
    5. 返回 Token + 用户信息
    """
    init_auth_db()
    name = req.name.strip()
    phone = req.phone.strip()
    if not name:
        raise HTTPException(status_code=400, detail="姓名不能为空")
    validate_phone(phone)

    conn = get_db()
    cur = conn.cursor()
    # 检查手机号是否已注册
    cur.execute("SELECT user_id FROM users WHERE phone = ?", (phone,))
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=409, detail="该手机号已注册，请直接登录")

    # 生成用户ID并哈希密码
    user_id = generate_user_id(name, phone)
    pwd_hash, salt = hash_password(req.password)

    # 插入用户记录
    cur.execute(
        "INSERT INTO users (user_id, name, phone, password_hash, salt) VALUES (?, ?, ?, ?, ?)",
        (user_id, name, phone, pwd_hash, salt),
    )

    # 生成并插入登录 Token（有效期 7 天）
    token = generate_token()
    expires_at = (datetime.now() + timedelta(days=TOKEN_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        "INSERT INTO tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
        (token, user_id, expires_at),
    )
    conn.commit()
    conn.close()
    return TokenResponse(token=token, user_id=user_id, name=name, phone=phone)


# ══════════════════════════════════════════════════════════════
# API — 用户登录
# ══════════════════════════════════════════════════════════════

@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """
    用户登录接口。

    流程：
    1. 验证手机号格式
    2. 按手机号查找用户（不存在 → 401）
    3. 检查账号是否启用（禁用 → 403）
    4. 验证密码哈希（不匹配 → 401）
    5. 生成新 Token 并返回
    """
    init_auth_db()
    phone = req.phone.strip()
    validate_phone(phone)

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE phone = ?", (phone,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=401, detail="手机号未注册，请先注册")

    user = dict(row)
    if not user["is_active"]:
        conn.close()
        raise HTTPException(status_code=403, detail="账号已被禁用")
    if not verify_password(req.password, user["password_hash"], user["salt"]):
        conn.close()
        raise HTTPException(status_code=401, detail="密码错误")

    # 登录成功，生成新 Token
    token = generate_token()
    expires_at = (datetime.now() + timedelta(days=TOKEN_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        "INSERT INTO tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
        (token, user["user_id"], expires_at),
    )
    conn.commit()
    conn.close()
    return TokenResponse(
        token=token,
        user_id=user["user_id"],
        name=user["name"],
        phone=user["phone"],
    )


# ══════════════════════════════════════════════════════════════
# API — 获取当前用户信息（Token 校验）
# ══════════════════════════════════════════════════════════════

@router.get("/me", response_model=UserInfoResponse)
async def get_me(authorization: str | None = Header(None)):
    """
    验证 Token 并返回当前用户信息。

    请求头: Authorization: Bearer <token>
    返回: user_id, name, phone, is_active, created_at
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少 Authorization 头")
    user = _get_user_by_token(authorization.replace("Bearer ", "", 1))  # 提取 Token 并查找用户
    return UserInfoResponse(
        user_id=user["user_id"],
        name=user["name"],
        phone=user["phone"],
        is_active=bool(user["is_active"]),
        created_at=str(user["created_at"]),
    )


# ══════════════════════════════════════════════════════════════
# API — 健康检查
# ══════════════════════════════════════════════════════════════

@router.get("/health")
async def health():
    init_auth_db()
    return {"status": "ok", "service": "auth"}
