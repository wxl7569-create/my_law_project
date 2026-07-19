"""通过后端语音接口调用的前端语音包装函数。"""

# ══════════════════════════════════════════════════════════════
# 导入标准库
# ══════════════════════════════════════════════════════════════
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

# ══════════════════════════════════════════════════════════════
# 导入第三方库
# ══════════════════════════════════════════════════════════════
import gradio as gr
import httpx

# ══════════════════════════════════════════════════════════════
# 导入内部模块
# ══════════════════════════════════════════════════════════════
from core.app_handlers import AGENT_API_URL
from core.utils.logger import LoggerManager
from core.utils.session_manager import DEFAULT_SESSION_NAME, SessionManager

# ══════════════════════════════════════════════════════════════
# 全局变量
# ══════════════════════════════════════════════════════════════
logger = LoggerManager.get_logger()


# ══════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════

def _gr_msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _save_session_index(user_id: str, thread_id: str) -> None:
    session_name = (
        DEFAULT_SESSION_NAME
        if thread_id == user_id
        else SessionManager()._extract_session_name(thread_id, user_id)
    )
    SessionManager().save_session(user_id, thread_id, session_name)


def _track_background_task(task: asyncio.Task) -> None:
    def _done(done_task: asyncio.Task) -> None:
        try:
            done_task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"语音后台任务执行失败: {e}", exc_info=True)

    task.add_done_callback(_done)


# ══════════════════════════════════════════════════════════════
# 内部 API 调用函数
# ══════════════════════════════════════════════════════════════

async def _api_voice_chat_stream(
    audio_path: str,
    thread_id: str,
    user_id: str,
) -> AsyncIterator[dict]:
    url = f"{AGENT_API_URL}/api/v1/voice/chat/stream"
    data = {"thread_id": thread_id, "user_id": user_id}

    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        with open(audio_path, "rb") as audio_file:
            files = {"file": (audio_path, audio_file, "application/octet-stream")}
            async with client.stream("POST", url, data=data, files=files) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"语音接口返回了无法解析的流式数据: {line!r}")


# ══════════════════════════════════════════════════════════════
# 核心前端包装函数
# ══════════════════════════════════════════════════════════════

async def wrap_handle_voice(audio_path, history, thread_id, login_state):
    """处理 Gradio 语音发送按钮。"""
    state = login_state or {}
    if not state.get("logged_in"):
        yield history or [], gr.update(), gr.update(visible=False)
        return

    if not audio_path:
        updated = (history or []) + [_gr_msg("assistant", "请先录制语音后再发送。")]
        yield updated, gr.update(), gr.update(visible=False)
        return

    user_id = state.get("user_id") or "anonymous"
    thread = thread_id or user_id
    history = history or []
    visible_history = history + [_gr_msg("assistant", "正在识别语音，请稍候。")]
    yield visible_history, gr.update(value=None), gr.update(visible=False)

    user_added = False
    answer_started = False

    try:
        async for event in _api_voice_chat_stream(str(audio_path), thread, user_id):
            event_type = event.get("type")

            if event_type == "asr":
                user_text = (event.get("text") or "").strip()
                if not user_text:
                    continue
                history = history + [_gr_msg("user", user_text)]
                visible_history = history + [_gr_msg("assistant", "正在思考，请稍候。")]
                user_added = True
                _track_background_task(
                    asyncio.create_task(asyncio.to_thread(_save_session_index, user_id, thread))
                )
                yield visible_history, gr.update(value=None), gr.update(visible=False)

            elif event_type == "text":
                content = event.get("content") or ""
                if not user_added:
                    history = history + [_gr_msg("user", "语音输入")]
                    user_added = True
                visible_history = history + [_gr_msg("assistant", content or "正在思考，请稍候。")]
                answer_started = True
                yield visible_history, gr.update(value=None), gr.skip()

            elif event_type == "audio":
                audio_reply = event.get("path")
                if audio_reply:
                    yield visible_history, gr.update(value=None), gr.update(
                        value=audio_reply,
                        visible=True,
                    )

            elif event_type == "audio_error":
                logger.warning(f"语音合成片段失败: {event.get('content')}")

            elif event_type == "error":
                content = event.get("content") or "语音处理失败。"
                if user_added and answer_started:
                    visible_history = visible_history[:-1] + [_gr_msg("assistant", content)]
                else:
                    visible_history = history + [_gr_msg("assistant", content)]
                yield visible_history, gr.update(value=None), gr.update(visible=False)
                return

            elif event_type == "done":
                return

    except Exception as e:
        logger.error(f"[voice wrapper error] {type(e).__name__}: {e}", exc_info=True)
        visible_history = history + [_gr_msg("assistant", f"连接语音后端服务失败：{type(e).__name__}: {e}")]
        yield visible_history, gr.update(value=None), gr.update(visible=False)
