"""Unified FastAPI backend for auth, sessions, and legal QA.

【架构角色】
本模块是后端服务的唯一入口，通过 lifespan 管理启动/关闭生命周期，
负责：认证数据库初始化、模型预热（Agent/RAG/ASR/TTS）、
会话检查点刷入、数据库连接关闭等。
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════
# 导入标准库与第三方库
# ══════════════════════════════════════════════════════════════

import asyncio              # 异步任务管理（预热、后台刷入）
from contextlib import asynccontextmanager  # FastAPI lifespan 上下文管理器

import uvicorn              # ASGI 服务器
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # 跨域资源共享中间件

# ══════════════════════════════════════════════════════════════
# 导入内部模块 — 路由、配置、日志
# ══════════════════════════════════════════════════════════════

from core.api.agent_routes import router as agent_router    # 文本问答路由（/chat, /chat/stream, /health）
from core.api.auth_routes import init_auth_db, router as auth_router  # 认证路由（/register, /login, /me）
from core.api.voice_routes import router as voice_router    # 语音路由（/asr, /tts, /chat/stream）
from core.config.settings import Config                     # 统一配置
from core.utils.logger import LoggerManager

logger = LoggerManager.get_logger()

_startup_tasks: set[asyncio.Task] = set()  # 跟踪启动时的后台预热任务，供关闭时取消


# ══════════════════════════════════════════════════════════════
# 后台任务管理
# ══════════════════════════════════════════════════════════════

def _track_task(task: asyncio.Task) -> None:
    """跟踪后台任务，任务完成后自动从集合中移除。"""
    _startup_tasks.add(task)
    task.add_done_callback(_startup_tasks.discard)


async def _run_prewarm_task(name: str, fn, *args) -> None:
    """
    执行单个预热任务，带错误捕获。

    参数 fn 可以是同步函数或协程函数，内部自动判断调用方式。
    """
    try:
        logger.info(f"[startup] 开始后台预热: {name}")
        result = fn(*args)
        if asyncio.iscoroutine(result):  # 协程函数返回 coroutine 对象
            await result
        logger.info(f"[startup] 后台预热完成: {name}")
    except Exception as e:
        logger.warning(f"[startup] 后台预热失败: {name}: {type(e).__name__}: {e}", exc_info=True)


# ══════════════════════════════════════════════════════════════
# 预热函数 — Agent / RAG / 语音（减少首次请求延迟）
# ══════════════════════════════════════════════════════════════

def _prewarm_rag() -> None:
    """预热民法典 RAG 向量库（加载 Chroma 索引 + 嵌入模型）。"""
    from core.utils.rag_law_civil import get_vectorstore
    get_vectorstore()  # 同步加载向量库和嵌入模型


async def _prewarm_voice() -> None:
    """预热语音 ASR 模型和 TTS 提供器（加载 SenseVoiceSmall 等）。"""
    from core.voice.asr import prewarm_asr
    from core.voice.tts import get_tts_provider
    await prewarm_asr()        # 加载本地 ASR 模型
    await get_tts_provider()   # 初始化 TTS 提供器（pyttsx3 或 edge-tts）


async def _prewarm_agent() -> None:
    """预热 Agent 实例（加载 LLM + 工具 + 检查点）。"""
    from core.agent_engine import get_agent
    await get_agent()          # 惰性初始化 Agent 单例


# ══════════════════════════════════════════════════════════════
# 应用生命周期管理（FastAPI lifespan）
#
# 启动流程：
#   1. 初始化认证数据库（users, tokens 表）
#   2. 清理过期语音缓存文件
#   3. 启动 3 个后台预热任务（Agent / RAG / Voice）
#
# 关闭流程：
#   1. 取消所有未完成的预热任务
#   2. 刷入所有会话检查点到 SQLite
#   3. 关闭所有会话
#   4. 关闭 SQLite 数据库连接
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Agent API service starting")

    # ── 启动阶段 ──
    init_auth_db()  # 确保 users.db 中有 users / tokens 表
    from core.utils.voice_cache import cleanup_voice_cache
    await asyncio.to_thread(cleanup_voice_cache)  # 清理过期的语音缓存文件

    # 依次启动三个预热任务
    await _run_prewarm_task("Agent", _prewarm_agent)  # Agent 是同步等待的，先完成
    _track_task(asyncio.create_task(_run_prewarm_task("RAG vectorstore", asyncio.to_thread, _prewarm_rag)))
    _track_task(asyncio.create_task(_run_prewarm_task("Voice ASR/TTS", _prewarm_voice)))

    try:
        yield  # 应用运行期间在此处挂起
    finally:
        # ── 关闭阶段 ──
        # 1. 取消仍在运行的预热任务
        for task in list(_startup_tasks):
            if not task.done():
                task.cancel()
        if _startup_tasks:
            await asyncio.gather(*list(_startup_tasks), return_exceptions=True)

        # 2. 刷入检查点 + 清理会话 + 关闭数据库连接
        from core.chat_session import close_all_sessions, flush_all_sessions
        from core.utils.memory_sqlite import close_sqlite_saver

        await flush_all_sessions()   # 将所有缓冲的检查点写入 SQLite
        close_all_sessions()         # 清空会话注册表
        await close_sqlite_saver()   # 关闭 SQLite 数据库连接
        logger.info("Agent API service stopped")


# ══════════════════════════════════════════════════════════════
# FastAPI 应用初始化与路由注册
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="PBL Legal QA Backend",
    description="FastAPI backend for user auth, session management, and LangGraph legal QA.",
    version="1.0.0",
    lifespan=lifespan,  # 绑定生命周期管理
)

# ── CORS 中间件：允许前端跨域请求（Gradio 前端默认运行在 7860 端口） ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,  # 默认 ["http://127.0.0.1:7860", "http://localhost:7860"]
    allow_credentials=True,             # 允许携带认证凭证
    allow_methods=["*"],                # 允许所有 HTTP 方法
    allow_headers=["*"],                # 允许所有请求头
)

# ── 注册三个子路由 ──
app.include_router(auth_router)   # /api/v1/auth/*  认证（登录/注册/Token验证）
app.include_router(agent_router)  # /api/v1/chat*   文本问答（非流式/流式）
app.include_router(voice_router)  # /api/v1/voice/* 语音（ASR/TTS/语音对话）


# ══════════════════════════════════════════════════════════════
# 根路由
# ══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "service": "PBL Legal QA Backend",
        "version": "1.0.0",
        "docs": "/docs",
        "auth": "/api/v1/auth",
        "chat": "/api/v1/chat",
    }


# ══════════════════════════════════════════════════════════════
# 启动入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"PBL Legal QA Backend: http://127.0.0.1:{Config.API_PORT}")
    print(f"API docs: http://127.0.0.1:{Config.API_PORT}/docs")
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT, log_level="info")
