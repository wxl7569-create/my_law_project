"""法律问答智能体的后端接口路由。

【路由清单】
POST /api/v1/chat        — 非流式文本问答（一次性返回完整回答）
POST /api/v1/chat/stream — 流式文本问答（SSE-like，逐行 JSON 事件）
GET  /api/v1/health      — 智能体健康检查
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════
# 导入标准库与第三方库
# ══════════════════════════════════════════════════════════════

import asyncio              # create_task 用于后台检查点刷入
import json                 # 流式响应 JSON 序列化
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse  # 流式响应类型（application/x-ndjson）
from pydantic import BaseModel, Field             # 请求/响应数据校验

# ══════════════════════════════════════════════════════════════
# 导入内部模块
# ══════════════════════════════════════════════════════════════

from core.chat_session import ChatSession, ChatSessionError, get_chat_session
from core.utils.logger import LoggerManager
from core.utils.quick_reply import get_quick_reply  # 简单问候/感谢的即时回复

logger = LoggerManager.get_logger()

router = APIRouter(prefix="/api/v1", tags=["agent"])  # 路由前缀 /api/v1，Swagger 标签 "agent"


# ══════════════════════════════════════════════════════════════
# 请求 / 响应数据模型
# ══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """文本问答请求体"""
    message: str = Field(..., min_length=1, max_length=20000)  # 用户输入消息，最大 20000 字符
    thread_id: str = Field("default", min_length=1)             # 会话线程ID（LangGraph checkpoint key）
    user_id: str = Field("anonymous", min_length=1)             # 用户标识


class ChatResponse(BaseModel):
    """非流式问答响应体"""
    answer: str                                                # Agent 完整回答文本
    session_id: str                                            # 会话ID（格式: thread_id:user_id）


class HealthResponse(BaseModel):
    """健康检查响应体"""
    status: str = "ok"
    service: str = "agent"
    version: str = "1.0.0"


# ══════════════════════════════════════════════════════════════
# 辅助函数 — 后台检查点刷入
#
# 每轮问答结束后，通过后台异步任务调用 session.flush()，
# 将缓冲的检查点写入 SQLite，不阻塞 HTTP 响应返回。
# ══════════════════════════════════════════════════════════════

def _schedule_flush(session: ChatSession) -> None:
    """
    创建后台任务刷入会话检查点。

    使用 asyncio.create_task 而非 await，确保 HTTP 响应立即返回，
    检查点写入在后台异步完成。
    """
    async def _flush() -> None:
        try:
            await session.flush()  # 将缓冲区中的检查点批量写入 SQLite
        except Exception as e:
            logger.warning(f"Background checkpoint flush failed: {e}", exc_info=True)

    asyncio.create_task(_flush())


# ══════════════════════════════════════════════════════════════
# API — 非流式文本问答
# ══════════════════════════════════════════════════════════════

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    保留用于兼容的非流式聊天接口。

    流程：
    1. 先检查是否匹配快捷回复（问候/感谢/告别）
    2. 非快捷则调用 Agent 生成回答
    3. 后台刷入检查点
    """
    # 步骤 1：尝试匹配快捷回复，避免不必要的 LLM 调用
    quick_reply = get_quick_reply(request.message)
    if quick_reply is not None:
        logger.info(f"Quick reply matched for chat: user={request.user_id}, thread={request.thread_id}")
        return ChatResponse(
            answer=quick_reply,
            session_id=f"{request.thread_id}:{request.user_id}",
        )

    # 步骤 2：获取会话并调用 Agent（非流式，一次性返回完整回答）
    session = get_chat_session(request.thread_id, request.user_id)
    try:
        answer = await session.invoke([{"role": "user", "content": request.message}])
        return ChatResponse(
            answer=answer or "抱歉，未能获取有效回答。",
            session_id=f"{request.thread_id}:{request.user_id}",
        )
    except ChatSessionError as e:
        logger.error(f"Agent call failed: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务暂时无法完成回答，请稍后重试") from e
    finally:
        _schedule_flush(session)  # 后台刷入检查点


# ══════════════════════════════════════════════════════════════
# API — 流式文本问答
# ══════════════════════════════════════════════════════════════

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    供前端增量展示使用的流式聊天接口。

    返回格式：application/x-ndjson（每行一个 JSON 事件）
    事件类型：
      {"type": "chunk", "content": "..."}  — 增量文本片段
      {"type": "done"}                     — 对话结束
      {"type": "error", "content": "..."}  — 错误消息
    """
    # 步骤 1：快捷回复也走流式格式，保持前端处理逻辑统一
    quick_reply = get_quick_reply(request.message)
    if quick_reply is not None:
        logger.info(f"Quick reply matched for stream: user={request.user_id}, thread={request.thread_id}")

        async def quick_event_generator():
            # 一次性发送完整回复，然后发送 done 信号
            yield json.dumps(
                {"type": "chunk", "content": quick_reply},
                ensure_ascii=False,  # 保持中文字符不转义为 \uXXXX
            ) + "\n"
            yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"

        return StreamingResponse(
            quick_event_generator(),
            media_type="application/x-ndjson",  # 换行分隔 JSON 流
        )

    session = get_chat_session(request.thread_id, request.user_id)

    async def event_generator():
        """
        内部异步生成器：
        - 调用 session.stream_response 获取增量文本
        - 每次增量包装为 {"type": "chunk", "content": "..."} 事件
        - 结束时发送 {"type": "done"} 事件
        - 异常时发送 {"type": "error"} 事件
        """
        try:
            async for chunk in session.stream_response(
                [{"role": "user", "content": request.message}]
            ):
                yield json.dumps(
                    {"type": "chunk", "content": chunk},
                    ensure_ascii=False,
                ) + "\n"

            yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.error(f"Agent stream failed: {type(e).__name__}: {e}", exc_info=True)
            yield json.dumps(
                {"type": "error", "content": "服务暂时无法完成回答，请稍后重试"},
                ensure_ascii=False,
            ) + "\n"
        finally:
            _schedule_flush(session)  # 后台刷入检查点

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
    )


# ══════════════════════════════════════════════════════════════
# API — 健康检查
# ══════════════════════════════════════════════════════════════

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse()
