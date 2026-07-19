"""文字聊天的前端包装函数。"""

# ══════════════════════════════════════════════════════════════
# 导入标准库
# ══════════════════════════════════════════════════════════════
from __future__ import annotations

# ══════════════════════════════════════════════════════════════
# 导入内部模块
# ══════════════════════════════════════════════════════════════
from core.app_handlers import handle_text


# ══════════════════════════════════════════════════════════════
# 核心前端包装函数
# ══════════════════════════════════════════════════════════════

async def wrap_handle_text(message, files, history, thread_id, login_state):
    state = login_state or {}
    if not state.get("logged_in"):
        yield history or [], "请先登录后再提问。", None
        return

    user_id = state.get("user_id") or "anonymous"
    thread = thread_id or user_id
    async for result in handle_text(
        message,
        history,
        files=files,
        thread_id=thread,
        user_id=user_id,
    ):
        yield result
