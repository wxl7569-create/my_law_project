"""语音识别、语音合成和语音对话接口路由。

【路由清单】
POST /api/v1/voice/asr         — 语音→文本（上传音频文件，返回识别文本）
POST /api/v1/voice/tts         — 文本→语音（传入文本，返回合成音频路径）
POST /api/v1/voice/chat/stream — 语音对话流（ASR → Agent → TTS 完整流水线）

【安全机制】
- 文件扩展名白名单过滤
- Content-Type 校验 + 文件魔术字检测
- 文件大小上限检查（默认 10MB）
- 按用户/IP 的请求频率限制
- ASR/TTS 并发数限制（Semaphore）
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════
# 导入标准库与第三方库
# ══════════════════════════════════════════════════════════════

import time                     # 时间戳（缓存文件名、限流窗口）
import uuid                     # 唯一标识（缓存文件名）
import asyncio                  # 异步锁、信号量、队列
from collections import defaultdict, deque  # 限流计数器数据结构
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse  # 流式响应（NDJSON 事件流）
from pydantic import BaseModel, Field

# ══════════════════════════════════════════════════════════════
# 导入内部模块
# ══════════════════════════════════════════════════════════════

from core.config.settings import Config
from core.utils.logger import LoggerManager
from core.utils.voice_cache import cleanup_voice_cache, delete_voice_file
from core.voice.asr import transcribe_audio          # 语音识别核心函数
from core.voice.pipeline import stream_voice_chat    # 语音对话编排
from core.voice.tts import synthesize_text           # 语音合成核心函数
from core.voice.vad import maybe_trim_silence        # 静音裁剪（可选）

logger = LoggerManager.get_logger()

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])

# ── 常量 ──
UPLOAD_CHUNK_SIZE = 64 * 1024              # 上传分块大小：64KB
RATE_LIMIT_WINDOW_SECONDS = 60             # 限流窗口：60 秒
# ── 全局状态 ──
_rate_limit_hits: dict[str, deque[float]] = defaultdict(deque)  # 按用户/IP 记录请求时间戳
_rate_limit_lock = asyncio.Lock()           # 限流计数器的异步锁
_asr_semaphore = asyncio.Semaphore(Config.VOICE_ASR_MAX_CONCURRENCY)  # ASR 并发限制（默认 2）
_tts_semaphore = asyncio.Semaphore(Config.VOICE_TTS_MAX_CONCURRENCY)  # TTS 并发限制（默认 2）


# ══════════════════════════════════════════════════════════════
# 请求 / 响应数据模型
# ══════════════════════════════════════════════════════════════

class AsrResponse(BaseModel):
    text: str


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


class TtsResponse(BaseModel):
    audio_path: str


# ══════════════════════════════════════════════════════════════
# 辅助函数 — 文件上传验证（多层安全过滤）
# ══════════════════════════════════════════════════════════════

def _safe_suffix(filename: str | None) -> str:
    """
    校验文件扩展名是否在白名单中。

    白名单（VOICE_ALLOWED_SUFFIXES）：.wav, .mp3, .m4a, .ogg, .webm, .flac
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix in Config.VOICE_ALLOWED_SUFFIXES:
        return suffix
    raise HTTPException(status_code=415, detail="不支持的音频文件扩展名")


def _client_key(request: Request, scope: str, user_id: str | None = None) -> str:
    """
    生成用于限流的客户端标识键。

    优先级：已登录用户用 user_id，未登录用客户端 IP。
    scope 区分不同类型的请求（"asr" / "tts" / "voice_chat"）。
    """
    client_host = request.client.host if request.client else "unknown"
    identity = user_id or client_host  # 优先使用用户ID，回退到IP
    return f"{scope}:{identity}"


async def _check_rate_limit(key: str) -> None:
    """
    滑动窗口限流检查。

    在 60 秒窗口内，同一客户端 key 的请求数不得超过 VOICE_RATE_LIMIT_PER_MINUTE（默认 30）。
    超过限制则抛出 429 Too Many Requests。
    """
    limit = Config.VOICE_RATE_LIMIT_PER_MINUTE
    if limit <= 0:
        return  # 限流关闭

    now = time.monotonic()  # 使用单调时钟，不受系统时间调整影响
    async with _rate_limit_lock:
        hits = _rate_limit_hits[key]
        # 清理窗口外的旧记录
        while hits and now - hits[0] >= RATE_LIMIT_WINDOW_SECONDS:
            hits.popleft()
        # 检查是否超限
        if len(hits) >= limit:
            raise HTTPException(status_code=429, detail="语音请求过于频繁，请稍后再试")
        hits.append(now)


def _validate_content_type(file: UploadFile) -> None:
    """
    校验上传文件的 Content-Type 头是否在白名单中。

    白名单包含：audio/wav, audio/mpeg, audio/mp4, audio/ogg, audio/webm, audio/flac 等。
    未声明 Content-Type 时放行（回退到文件魔术字检测）。
    """
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if not content_type:
        return  # 未声明 Content-Type，回退到魔术字检测
    if content_type not in Config.VOICE_ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="不支持的音频 Content-Type")


def _looks_like_audio(header: bytes) -> bool:
    """
    通过文件头魔术字判断是否为有效音频格式。

    支持的格式检测：
    - WAV: RIFF....WAVE
    - MP3: ID3... 或 0xFF 0xE? 同步头
    - OGG: OggS
    - FLAC: fLaC
    - WebM/EBML: 0x1A 0x45 0xDF 0xA3
    - MP4/M4A: ....ftyp
    """
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return True  # WAV 格式
    if header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0):
        return True  # MP3 格式（ID3v2 标签 或 MPEG 同步头）
    if header.startswith(b"OggS") or header.startswith(b"fLaC"):
        return True  # OGG 或 FLAC 格式
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return True  # WebM/EBML 格式
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return True  # MP4/M4A 格式
    return False


async def _save_upload(file: UploadFile) -> Path:
    """
    安全保存上传的音频文件。

    多层验证流程：
    1. Content-Type 白名单校验
    2. 确保缓存目录存在 + 触发旧缓存清理
    3. 扩展名白名单校验
    4. 按 64KB 分块流式写入磁盘
    5. 首块数据做魔术字检测（防止 Content-Type 伪造）
    6. 累计大小检查（防止磁盘撑爆）

    失败时自动删除已写入的部分文件。

    返回：保存后的文件路径
    """
    path: Path | None = None
    total = 0            # 已写入字节数
    header = b""         # 首块数据（用于魔术字检测）

    try:
        _validate_content_type(file)           # 步骤 1：Content-Type 校验
        Config.VOICE_AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)  # 步骤 2a：确保目录存在
        await asyncio.to_thread(cleanup_voice_cache)  # 步骤 2b：清理过期缓存
        suffix = _safe_suffix(file.filename)   # 步骤 3：扩展名校验
        # 生成唯一文件名：asr_{时间戳}_{随机ID}{扩展名}
        path = Config.VOICE_AUDIO_TEMP_DIR / f"asr_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}{suffix}"

        with path.open("wb") as out:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)  # 步骤 4：64KB 分块读取
                if not chunk:
                    break  # 读取完毕
                if not header:
                    header = chunk[:32]        # 步骤 5：取前 32 字节做魔术字检测
                    if not _looks_like_audio(header):
                        raise HTTPException(status_code=415, detail="上传内容不是有效音频文件")
                total += len(chunk)
                if total > Config.VOICE_UPLOAD_MAX_BYTES:  # 步骤 6：文件大小检查（默认 10MB）
                    raise HTTPException(status_code=413, detail="音频文件过大")
                out.write(chunk)

        if total == 0:
            raise HTTPException(status_code=400, detail="音频文件为空")
    finally:
        await file.close()  # 确保上传文件句柄关闭
        # 如果保存失败，删除已写入的部分文件
        if path is not None and (
            total == 0 or total > Config.VOICE_UPLOAD_MAX_BYTES or not header or not _looks_like_audio(header)
        ):
            delete_voice_file(path)
    if path is None:
        raise HTTPException(status_code=400, detail="音频上传保存失败")
    return path


# ══════════════════════════════════════════════════════════════
# API — 语音识别 (ASR)
# ══════════════════════════════════════════════════════════════

@router.post("/asr", response_model=AsrResponse)
async def asr(request: Request, file: UploadFile = File(...)):
    """
    语音→文本识别接口。

    流程：限流 → 信号量排队 → 保存上传 → 静音裁剪 → ASR 识别 → 清理临时文件
    """
    await _check_rate_limit(_client_key(request, "asr"))
    audio_path = None
    trimmed = None
    try:
        async with _asr_semaphore:           # 信号量控制并发（默认同时最多 2 个 ASR 任务）
            audio_path = await _save_upload(file)      # 保存上传音频
            trimmed = await maybe_trim_silence(audio_path)  # 可选静音裁剪（默认关闭）
            return AsrResponse(text=await transcribe_audio(trimmed))  # 执行 ASR 识别
    except HTTPException:
        raise  # 已知 HTTP 异常直接透传
    except Exception as e:
        logger.error(f"[voice] ASR 接口失败: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="ASR 识别失败，请稍后重试") from e
    finally:
        # 清理临时文件（裁剪后的 + 原始录音）
        delete_voice_file(trimmed)
        if trimmed != audio_path:
            delete_voice_file(audio_path)


# ══════════════════════════════════════════════════════════════
# API — 语音合成 (TTS)
# ══════════════════════════════════════════════════════════════

@router.post("/tts", response_model=TtsResponse)
async def tts(request: Request, body: TtsRequest):
    """
    文本→语音合成接口。

    流程：限流 → 信号量排队 → 文本清洗 → 语音合成 → 返回音频文件路径
    """
    await _check_rate_limit(_client_key(request, "tts"))
    try:
        async with _tts_semaphore:           # 信号量控制并发（默认同时最多 2 个 TTS 任务）
            audio_path = await synthesize_text(body.text)
            if audio_path is None:
                raise ValueError("文本清洗后为空，无法合成语音")
        return TtsResponse(audio_path=str(audio_path))
    except Exception as e:
        logger.error(f"[voice] TTS 接口失败: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="TTS 合成失败，请稍后重试") from e


# ══════════════════════════════════════════════════════════════
# API — 语音对话流式接口 (ASR → Agent → TTS 完整流水线)
# ══════════════════════════════════════════════════════════════

@router.post("/chat/stream")
async def voice_chat_stream(
    request: Request,
    file: UploadFile = File(...),
    thread_id: str = Form("default"),  # 会话线程ID（表单字段）
    user_id: str = Form("anonymous"),  # 用户ID（表单字段）
):
    """
    语音对话流式接口。

    完整流水线：音频上传 → ASR → Agent 流式回答 → TTS 语音合成 → NDJSON 事件流返回

    流式事件类型：
      {"type": "asr", "text": "..."}       — ASR 识别结果
      {"type": "text", "content": "..."}   — Agent 增量回答文本
      {"type": "audio", "path": "...", "index": N}  — TTS 合成音频片段
      {"type": "audio", "path": "...", "index": "final", "final": true}  — 完整语音
      {"type": "done"}                     — 对话结束
      {"type": "audio_error", ...}         — 语音合成片段失败
      {"type": "error", "content": "..."}  — 整体失败
    """
    await _check_rate_limit(_client_key(request, "voice_chat", user_id))
    try:
        audio_path = await _save_upload(file)  # 保存上传音频
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[voice] 音频上传保存失败: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="音频上传保存失败") from e

    return StreamingResponse(
        stream_voice_chat(audio_path=audio_path, thread_id=thread_id, user_id=user_id),
        media_type="application/x-ndjson",  # 换行分隔 JSON 流
    )
