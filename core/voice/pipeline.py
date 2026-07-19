"""语音对话编排。

【核心流程】
ASR 识别 → Agent 流式生成 → 文本增量切分 → 并行 TTS 合成 → NDJSON 事件流输出

【设计要点】
- 文本流和音频流并行：Agent 每产生增量文本，立即触发 TTS 合成
- 短句优先播放：使用 TtsSentenceSplitter 将流式文本切为 ~45 字短句
- 完整语音兜底：回答结束后合成完整版本（VOICE_TTS_FULL_FINAL=1）
- 会话隔离：通过 ChatSession 管理 (user_id, thread_id) 隔离
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════
# 导入标准库与第三方库
# ══════════════════════════════════════════════════════════════

import asyncio              # 并发任务管理（TTS 调度、信号量、队列）
import json                 # JSON 事件序列化
from collections.abc import AsyncIterator
from pathlib import Path

from core.chat_session import get_chat_session  # 会话管理
from core.config.settings import Config
from core.utils.logger import LoggerManager
from core.utils.quick_reply import get_quick_reply      # 快捷回复（问候/感谢）
from core.utils.voice_cache import delete_voice_file     # 临时文件清理
from core.voice.asr import transcribe_audio             # 语音识别
from core.voice.splitter import TtsSentenceSplitter     # 短句切分器
from core.voice.tts import synthesize_text              # 语音合成
from core.voice.vad import maybe_trim_silence           # 静音裁剪

logger = LoggerManager.get_logger()

# ── 并发控制信号量（与 voice_routes.py 独立，pipeine 内也需要控制） ──
_asr_semaphore = asyncio.Semaphore(Config.VOICE_ASR_MAX_CONCURRENCY)  # ASR 并发限制
_tts_semaphore = asyncio.Semaphore(Config.VOICE_TTS_MAX_CONCURRENCY)  # TTS 并发限制


# ══════════════════════════════════════════════════════════════
# 常量 — 语音模式提示词
#
# 注入到 Agent 的 system prompt 中，指导 Agent 生成适合语音朗读的回复：
# - 保持法律回答完整性
# - 使用自然口语，避免 Markdown/表格/代码块
# ══════════════════════════════════════════════════════════════

VOICE_MODE_PROMPT = (
    "当前是语音回复模式。请保持法律回答的完整性，不要因为需要朗读而省略关键条件、"
    "法条依据、法律效果或必要提醒。回答应使用自然口语，不要使用表情符号、Markdown 标题、"
    "表格、代码块或复杂分层编号。可以用短段落表达，但内容必须完整、连贯、适合直接朗读。"
)


# ══════════════════════════════════════════════════════════════
# 辅助函数 — 事件格式化与后台刷入
# ══════════════════════════════════════════════════════════════

def _event(event_type: str, **payload) -> str:
    """
    将事件格式化为 NDJSON 行。

    生成格式：{"type": "asr", "text": "..."}\n
    ensure_ascii=False 保持中文原样，不转义为 \\uXXXX。
    """
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


def _schedule_session_flush(session) -> None:
    """
    后台刷入会话检查点。

    pipeline 中每次语音对话结束后调用，不阻塞事件流返回。
    """
    async def _flush() -> None:
        try:
            await session.flush()  # 将缓冲检查点写入 SQLite
        except Exception as e:
            logger.warning(f"[voice] 后台刷入检查点失败: {e}", exc_info=True)

    asyncio.create_task(_flush())


# ══════════════════════════════════════════════════════════════
# 核心 — ASR → Agent → TTS 语音对话流
#
# 整体架构：
#   ┌──────────┐     ┌──────────┐     ┌──────────┐
#   │   ASR    │ ──→ │  Agent   │ ──→ │   TTS    │
#   │ 语音→文本 │     │ 流式生成 │     │ 文本→语音 │
#   └──────────┘     └──────────┘     └──────────┘
#                          ↓
#                    TtsSentenceSplitter
#                    (增量→短句切分)
#                          ↓
#                ┌──────────────────┐
#                │ 并行 TTS 合成     │
#                │ (每短句一个 Task) │
#                └──────────────────┘
#                          ↓
#                NDJSON 事件流输出
# ══════════════════════════════════════════════════════════════

async def stream_voice_chat(
    audio_path: str | Path,
    thread_id: str,
    user_id: str,
) -> AsyncIterator[str]:
    """
    ASR → Agent → TTS 的语音对话事件流。

    返回：AsyncIterator[str]，每个元素是一行 NDJSON 格式的事件。

    执行阶段：
    阶段 1 — ASR 识别：音频 → 静音裁剪 → 语音识别 → 文本
    阶段 2 — 快捷回复：如果是问候/感谢等简单消息，直接 TTS 合成返回
    阶段 3 — Agent 流式生成 + TTS 并行合成：
      - Agent 每产生增量文本 → 切分短句 → 创建 TTS 任务
      - TTS 任务并行执行，完成后放入队列
      - 文本流和音频流交替输出
    阶段 4 — 收尾：等待剩余 TTS 完成 → 可选完整语音 → done
    阶段 5 — 清理：取消未完成任务、刷入检查点、删除临时文件
    """
    session = None                       # ChatSession 实例（延迟创建）
    tts_tasks: set[asyncio.Task] = set()  # 跟踪所有 TTS 异步任务
    trimmed_audio = None                  # 静音裁剪后的音频路径

    # ═══════════════════════════════════════════════════════
    # 阶段 1 — ASR 识别
    # ═══════════════════════════════════════════════════════
    try:
        async with _asr_semaphore:
            trimmed_audio = await maybe_trim_silence(audio_path)   # 可选静音裁剪
            user_text = await transcribe_audio(trimmed_audio)      # 语音→文本
        if not user_text:
            yield _event("error", content="未识别到有效语音，请重新录音。")
            delete_voice_file(trimmed_audio)
            if trimmed_audio != audio_path:
                delete_voice_file(audio_path)
            return

        yield _event("asr", text=user_text)  # 通知前端识别结果
        logger.info(f"[用户提问] {user_text}")

    except Exception as e:
        logger.error(f"[voice] ASR 阶段失败: {type(e).__name__}: {e}", exc_info=True)
        yield _event("error", content="语音识别失败，请稍后重试。")
        delete_voice_file(trimmed_audio)
        if trimmed_audio != audio_path:
            delete_voice_file(audio_path)
        return

    # ═══════════════════════════════════════════════════════
    # 阶段 2 — 快捷回复（问候/感谢/告别，跳过 Agent）
    # ═══════════════════════════════════════════════════════
    quick_reply = get_quick_reply(user_text)
    if quick_reply is not None:
        yield _event("text", content=quick_reply)
        # 一次性合成完整语音
        async with _tts_semaphore:
            audio = await synthesize_text(quick_reply)
        if audio is not None:
            yield _event("audio", path=str(audio), index=0)
        yield _event("done")
        # 清理临时文件
        delete_voice_file(trimmed_audio)
        if trimmed_audio != audio_path:
            delete_voice_file(audio_path)
        return

    # ═══════════════════════════════════════════════════════
    # 阶段 3 — Agent 流式生成 + TTS 并行合成
    # ═══════════════════════════════════════════════════════
    # 内部状态变量
    session = get_chat_session(thread_id, user_id)  # 获取/创建会话
    splitter = TtsSentenceSplitter()                # 短句切分器（~45字/句）
    audio_queue: asyncio.Queue = asyncio.Queue()     # TTS 完成队列
    segment_index = 0            # 短句序号（用于保证前端播放顺序）
    last_text = ""               # 上一轮的累积文本（用于计算增量 delta）
    tts_disabled = False         # TTS 是否因失败而禁用

    # ── 内部函数：单个短句 TTS 合成 ──
    async def schedule_tts(segment: str, index: int) -> None:
        """
        异步调度一个短句的 TTS 合成。

        每个短句独立创建一个协程任务，通过信号量控制并发数。
        合成完成后将结果放入 audio_queue，前端可按 index 顺序播放。
        """
        nonlocal tts_disabled
        if tts_disabled:
            return  # TTS 已因失败而禁用，跳过后续合成

        try:
            async with _tts_semaphore:               # 信号量控制并发
                path = await synthesize_text(segment)  # 文本清洗 + 合成
            if path is not None:
                await audio_queue.put((index, str(path)))  # 成功：放入队列
        except Exception as e:
            tts_disabled = True  # 失败则禁用后续合成，避免连续错误
            logger.warning(f"[voice] TTS 片段合成失败: {e}", exc_info=True)
            await audio_queue.put((index, "", "语音合成失败，请稍后重试。"))  # 失败通知

    # ── 内部函数：将短句列表批量提交 TTS 合成 ──
    def add_tts_segments(segments: list[str]) -> None:
        nonlocal segment_index
        if tts_disabled:
            return
        for segment in segments:
            task = asyncio.create_task(schedule_tts(segment, segment_index))
            tts_tasks.add(task)
            segment_index += 1  # 每个短句分配递增序号

    # ── 内部函数：从队列中取出已完成的 TTS 结果并输出事件 ──
    async def drain_audio_queue() -> AsyncIterator[str]:
        """非阻塞地清空 audio_queue，生成 audio/audio_error 事件。"""
        while True:
            try:
                item = audio_queue.get_nowait()  # 非阻塞取
            except asyncio.QueueEmpty:
                break
            if len(item) == 2:
                index, path = item
                yield _event("audio", path=path, index=index)  # 成功音频
            else:
                index, _, error = item
                yield _event("audio_error", index=index, content=error)  # 失败通知

    # ═══════════════════════════════════════════════════════
    # 阶段 3 主体 — Agent 流式生成循环
    # ═══════════════════════════════════════════════════════
    try:
        # 构造语音模式消息（注入 VOICE_MODE_PROMPT 作为 system prompt）
        messages = [
            {"role": "system", "content": VOICE_MODE_PROMPT},
            {"role": "user", "content": user_text},
        ]
        async for full_text in session.stream_response(messages):
            # 计算增量文本
            delta = full_text[len(last_text):] if full_text.startswith(last_text) else full_text
            last_text = full_text
            if delta:
                # 将增量文本喂入分句器，得到完整短句列表
                add_tts_segments(splitter.feed(delta))

            # 输出当前累积的完整文本到前端
            yield _event("text", content=full_text)

            # 非阻塞取出已完成的 TTS 结果
            async for audio_event in drain_audio_queue():
                yield audio_event

        # 回答结束，强制 flush 分句器中的剩余文本
        add_tts_segments(splitter.flush())

    except Exception as e:
        logger.error(f"[voice] 语音对话失败: {type(e).__name__}: {e}", exc_info=True)
        yield _event("error", content="语音对话失败，请稍后重试。")
        # 异常情况下仍需清理
        for task in list(tts_tasks):
            if not task.done():
                task.cancel()
        if session is not None:
            _schedule_session_flush(session)
        delete_voice_file(trimmed_audio)
        if trimmed_audio != audio_path:
            delete_voice_file(audio_path)
        return

    # ═══════════════════════════════════════════════════════
    # 阶段 4 — 收尾：等待 TTS 完成 + 可选完整语音
    # ═══════════════════════════════════════════════════════
    # 等待所有 TTS 任务完成后输出剩余音频
    while tts_tasks:
        done, _pending = await asyncio.wait(
            set(tts_tasks),
            return_when=asyncio.FIRST_COMPLETED,  # 任一完成即处理
        )
        for task in done:
            tts_tasks.discard(task)
            try:
                await task  # 获取结果/传播异常
            except Exception:
                pass
        async for audio_event in drain_audio_queue():
            yield audio_event

    # 最后一次清空队列
    async for audio_event in drain_audio_queue():
        yield audio_event

    # 完整语音兜底：回答全部结束后合成一份完整版本
    if Config.VOICE_TTS_FULL_FINAL and last_text.strip():
        try:
            async with _tts_semaphore:
                final_audio = await synthesize_text(last_text)
            if final_audio is not None:
                # 标记 final=True 供前端特殊处理（如替换之前的片段音频）
                yield _event("audio", path=str(final_audio), index="final", final=True)
        except Exception as e:
            logger.warning(f"[voice] 完整语音合成失败: {e}", exc_info=True)

    yield _event("done")
    logger.info(f"[Agent回答] {last_text[:100]}")

    # ═══════════════════════════════════════════════════════
    # 阶段 5 — 清理
    # ═══════════════════════════════════════════════════════
    # 取消所有未完成的 TTS 任务
    for task in list(tts_tasks):
        if not task.done():
            task.cancel()
    # 后台刷入检查点
    if session is not None:
        _schedule_session_flush(session)
    # 删除临时音频文件
    delete_voice_file(trimmed_audio)
    if trimmed_audio != audio_path:
        delete_voice_file(audio_path)
