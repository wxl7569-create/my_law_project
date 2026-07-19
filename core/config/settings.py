"""项目统一配置。"""

# ═══════════════════════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Config Class (Runtime Settings)
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """命令行、后端接口和前端共用的运行时配置。"""

    # ── 项目路径 ──
    # 项目根目录，相对于当前脚本向上两个目录
    BASE_DIR = Path(__file__).resolve().parents[2]

    LLM_TYPE = os.getenv("LLM_TYPE", "deepseek")

    DATA_DIR = BASE_DIR / "data"
    MEMORY_DB_PATH = str(DATA_DIR / "memory.db")
    USERS_DB_PATH = str(DATA_DIR / "users.db")

    # ── 日志配置 ──

    LOG_DIR = str(BASE_DIR / "logfile")
    LOG_FILE = str(Path(LOG_DIR) / "app.log")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    MAX_BYTES = 10 * 1024 * 1024
    BACKUP_COUNT = 5

    # ── 服务器配置 ──

    API_HOST = os.getenv("HOST", "0.0.0.0")             #后端API绑定主机地址
    API_PORT = int(os.getenv("PORT", "8001"))             #后端API绑定端口
    FRONTEND_HOST = os.getenv("FRONTEND_HOST", "127.0.0.1") #前端绑定主机地址
    FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "7860")) #前端绑定端口
    BACKEND_API_URL = os.getenv(
        "BACKEND_API_URL",
        f"http://127.0.0.1:{API_PORT}",
    ).rstrip("/")                                            #后端API URL

    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://127.0.0.1:7860,http://localhost:7860",
        ).split(",")
        if origin.strip()
        # 允许的CORS来源
    ]

    # ── 模型和语音路径 ──

    MODELS_DIR = BASE_DIR / "models"
    VOICE_ASR_MODEL_DIR = Path(
        os.getenv("VOICE_ASR_MODEL_DIR", str(MODELS_DIR / "SenseVoiceSmall"))
    )
    VOICE_VAD_MODEL_DIR = Path(
        os.getenv("VOICE_VAD_MODEL_DIR", str(MODELS_DIR / "snakers4_silero-vad"))
    )
    VOICE_AUDIO_TEMP_DIR = Path(
        os.getenv("VOICE_AUDIO_TEMP_DIR", str(DATA_DIR / "voice_cache"))
    )
    VOICE_UPLOAD_MAX_BYTES = int(os.getenv("VOICE_UPLOAD_MAX_BYTES", str(10 * 1024 * 1024)))
    VOICE_CACHE_TTL_SECONDS = int(os.getenv("VOICE_CACHE_TTL_SECONDS", str(60 * 60)))
    VOICE_CACHE_MAX_BYTES = int(os.getenv("VOICE_CACHE_MAX_BYTES", str(200 * 1024 * 1024)))
    VOICE_RATE_LIMIT_PER_MINUTE = int(os.getenv("VOICE_RATE_LIMIT_PER_MINUTE", "30"))
    VOICE_ASR_MAX_CONCURRENCY = int(os.getenv("VOICE_ASR_MAX_CONCURRENCY", "2"))
    VOICE_TTS_MAX_CONCURRENCY = int(os.getenv("VOICE_TTS_MAX_CONCURRENCY", "2"))
    VOICE_ALLOWED_SUFFIXES = {
        item.strip().lower()
        for item in os.getenv("VOICE_ALLOWED_SUFFIXES", ".wav,.mp3,.m4a,.ogg,.webm,.flac").split(",")
        if item.strip()
    }
    VOICE_ALLOWED_MIME_TYPES = {
        item.strip().lower()
        for item in os.getenv(
            "VOICE_ALLOWED_MIME_TYPES",
            "audio/wav,audio/wave,audio/x-wav,audio/vnd.wave,audio/mpeg,audio/mp4,audio/x-m4a,"
            "audio/ogg,application/ogg,audio/webm,video/webm,audio/flac,application/octet-stream",
        ).split(",")
        if item.strip()
    }

    # ── 语音特征标志 ──

    VOICE_ENABLE_VAD = os.getenv("VOICE_ENABLE_VAD", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    VOICE_TTS_PROVIDER = os.getenv("VOICE_TTS_PROVIDER", "pyttsx3").strip().lower()
    VOICE_TTS_VOICE = os.getenv("VOICE_TTS_VOICE", "")
    VOICE_TTS_RATE = int(os.getenv("VOICE_TTS_RATE", "180"))
    VOICE_TTS_VOLUME = float(os.getenv("VOICE_TTS_VOLUME", "1.0"))
    VOICE_TTS_EDGE_VOICE = os.getenv("VOICE_TTS_EDGE_VOICE", "zh-CN-XiaoxiaoNeural")
    VOICE_TTS_EDGE_RATE = os.getenv("VOICE_TTS_EDGE_RATE", "+0%")
    VOICE_TTS_EDGE_VOLUME = os.getenv("VOICE_TTS_EDGE_VOLUME", "+0%")
    TEXT_CHAT_ENABLE_TTS = os.getenv("TEXT_CHAT_ENABLE_TTS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    VOICE_TTS_FULL_FINAL = os.getenv("VOICE_TTS_FULL_FINAL", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
