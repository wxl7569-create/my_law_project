"""会话持久化键生成与解析。"""

from __future__ import annotations

from urllib.parse import quote, unquote

CHECKPOINT_THREAD_PREFIX = "user="
CHECKPOINT_THREAD_SEPARATOR = "|thread="


def build_checkpoint_thread_id(user_id: str, thread_id: str) -> str:
    """生成 LangGraph checkpoint 使用的复合 thread_id。"""
    safe_user = quote(user_id or "anonymous", safe="")
    safe_thread = quote(thread_id or "default", safe="")
    return f"{CHECKPOINT_THREAD_PREFIX}{safe_user}{CHECKPOINT_THREAD_SEPARATOR}{safe_thread}"


def parse_checkpoint_thread_id(value: str) -> tuple[str | None, str]:
    """解析复合 checkpoint thread_id，旧格式原样返回。"""
    if not value.startswith(CHECKPOINT_THREAD_PREFIX) or CHECKPOINT_THREAD_SEPARATOR not in value:
        return None, value

    raw_user, raw_thread = value[len(CHECKPOINT_THREAD_PREFIX):].split(
        CHECKPOINT_THREAD_SEPARATOR,
        1,
    )
    return unquote(raw_user), unquote(raw_thread)
