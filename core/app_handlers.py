"""前端事件辅助函数。

网页前端只负责界面状态和用户交互，实际问答工作统一通过后端接口完成。
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator

import gradio as gr
import httpx

from core.config.settings import Config
from core.utils.logger import LoggerManager
from core.utils.session_manager import DEFAULT_SESSION_NAME, SessionManager
from core.voice.splitter import TtsSentenceSplitter

logger = LoggerManager.get_logger()

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

AGENT_API_URL = Config.BACKEND_API_URL

GUIDE_EXAMPLES = [
    "民法典中的基本原则规定有哪些？",
    "什么是自然人？请结合民法典说明",
    "民法典中规定什么叫宣告失踪？",
    "民法典中关于监护的规定有哪些？",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Internal Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

def _gr_msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _extract_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def _build_file_context(files: list[str]) -> str:
    parts = ["用户上传了以下文档，请使用 word_reader 工具读取并分析："]
    for fp in files:
        parts.append(f"- [{os.path.basename(fp)}] {fp}")
    return "\n".join(parts)


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
            logger.warning(f"后台任务执行失败: {e}", exc_info=True)

    task.add_done_callback(_done)


# ═══════════════════════════════════════════════════════════════════════════════
# API Communication (Streaming & TTS)
# ═══════════════════════════════════════════════════════════════════════════════

async def _api_ask_stream(
    message: str,
    thread_id: str,
    user_id: str,
) -> AsyncIterator[str]:
    payload = {"message": message, "thread_id": thread_id, "user_id": user_id}
    url = f"{AGENT_API_URL}/api/v1/chat/stream"

    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"后端返回了无法解析的流式数据: {line!r}")
                    continue

                event_type = event.get("type")
                if event_type == "chunk":
                    yield event.get("content") or ""
                elif event_type == "error":
                    yield event.get("content") or "后端流式响应失败。"
                    return
                elif event_type == "done":
                    return


async def _api_tts(text: str) -> str | None:
    url = f"{AGENT_API_URL}/api/v1/voice/tts"
    payload = {"text": text}

    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("audio_path")


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Reply Streaming
# ═══════════════════════════════════════════════════════════════════════════════

async def stream_agent_reply(history: list, thread_id: str, user_id: str):
    messages = [{"role": m["role"], "content": _extract_text(m)} for m in history]
    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if not last_user_msg:
        yield history + [_gr_msg("assistant", "未检测到用户消息。")]
        return

    has_chunk = False
    try:
        async for full_text in _api_ask_stream(last_user_msg, thread_id, user_id):
            has_chunk = True
            yield history + [_gr_msg("assistant", full_text)]

        if not has_chunk:
            yield history + [_gr_msg("assistant", "抱歉，未能获取有效回答。")]
    except Exception as e:
        logger.error(f"[API stream mode error] {type(e).__name__}: {e}", exc_info=True)
        yield history + [_gr_msg("assistant", f"连接后端服务失败：{type(e).__name__}: {e}")]


# ═══════════════════════════════════════════════════════════════════════════════
# Text Chat Handler
# ═══════════════════════════════════════════════════════════════════════════════

async def handle_text(
    message: str,
    history: list,
    files: list | None = None,
    thread_id: str = "default",
    user_id: str = "anonymous",
):
    file_context = ""
    if files:
        file_list = files if isinstance(files, list) else [files]
        file_list = [str(f) for f in file_list if f]
        if file_list:
            file_context = _build_file_context(file_list)

    full_message = message.strip() if message else ""
    if file_context:
        full_message = f"{file_context}\n\n用户问题：{full_message}" if full_message else file_context

    if not full_message:
        yield history, gr.update(), gr.update(visible=False)
        return

    history = history or []
    shown_text = message.strip() if message else "(文件上传)"
    history.append(_gr_msg("user", shown_text))
    _track_background_task(
        asyncio.create_task(asyncio.to_thread(_save_session_index, user_id, thread_id))
    )

    yield history, gr.update(value=""), gr.update(visible=False)
    yield history + [_gr_msg("assistant", "正在思考，请稍候...")], gr.skip(), gr.update(visible=False)

    api_history = history[:-1] + [_gr_msg("user", full_message)]
    if not Config.TEXT_CHAT_ENABLE_TTS:
        async for updated in stream_agent_reply(api_history, thread_id=thread_id, user_id=user_id):
            display_history = history + [updated[-1]]
            yield display_history, gr.skip(), gr.update(visible=False)
        return

    splitter = TtsSentenceSplitter()
    tts_queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()
    tts_tasks: set[asyncio.Task] = set()
    last_answer = ""
    tts_disabled = False

    async def schedule_tts(segment: str) -> None:
        nonlocal tts_disabled
        if tts_disabled:
            return

        try:
            audio_path = await _api_tts(segment)
            if audio_path:
                await tts_queue.put((audio_path, None))
        except Exception as e:
            tts_disabled = True
            logger.warning(f"[text tts] 文字回复语音合成失败: {e}", exc_info=True)
            await tts_queue.put(("", str(e)))

    def add_tts_segments(segments: list[str]) -> None:
        if tts_disabled:
            return
        for segment in segments:
            task = asyncio.create_task(schedule_tts(segment))
            tts_tasks.add(task)
            task.add_done_callback(tts_tasks.discard)

    async def drain_tts_queue(display_history: list):
        while True:
            try:
                audio_path, error = tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if error:
                logger.warning(f"[text tts] 文字回复语音合成失败: {error}")
                continue

            yield display_history, gr.skip(), gr.update(
                value=audio_path,
                visible=True,
            )

    display_history = history
    async for updated in stream_agent_reply(api_history, thread_id=thread_id, user_id=user_id):
        display_history = history + [updated[-1]]
        current_answer = _extract_text(updated[-1])
        delta = (
            current_answer[len(last_answer) :]
            if current_answer.startswith(last_answer)
            else current_answer
        )
        last_answer = current_answer
        if delta:
            add_tts_segments(splitter.feed(delta))

        async for tts_update in drain_tts_queue(display_history):
            yield tts_update

        yield display_history, gr.skip(), gr.skip()

    add_tts_segments(splitter.flush())
    while tts_tasks:
        done, _pending = await asyncio.wait(
            set(tts_tasks),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            tts_tasks.discard(task)
            try:
                await task
            except Exception:
                pass
        async for tts_update in drain_tts_queue(display_history):
            yield tts_update

    async for tts_update in drain_tts_queue(display_history):
        yield tts_update
