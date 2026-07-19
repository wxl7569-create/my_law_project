"""本地 ASR 识别（基于 FunASR SenseVoiceSmall）。

【模型加载策略】
- 惰性初始化 + 异步锁保护的单例模式
- 首次调用 get_asr_model() 时加载，后续复用
- 支持后台预热（prewarm_asr），减少首次请求延迟
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════
# 导入标准库与第三方库
# ══════════════════════════════════════════════════════════════

import asyncio              # 异步锁、run_in_executor 避免阻塞事件循环
from pathlib import Path
from typing import Any

from core.config.settings import Config
from core.utils.logger import LoggerManager

logger = LoggerManager.get_logger()

# ══════════════════════════════════════════════════════════════
# 模块级状态（异步锁 + 模型缓存）
# ══════════════════════════════════════════════════════════════

_asr_model: Any | None = None  # FunASR AutoModel 实例（单例）
_asr_lock = asyncio.Lock()      # 异步锁，防止并发初始化


# ══════════════════════════════════════════════════════════════
# 模型生命周期（预热、获取、加载）
# ══════════════════════════════════════════════════════════════

async def prewarm_asr() -> None:
    """后台预热 ASR 模型——在服务启动时提前加载，减少首次请求延迟。"""
    await get_asr_model()


async def get_asr_model():
    """
    获取 FunASR AutoModel 实例（惰性单例，线程安全）。

    双重检查锁模式：
    1. 先无锁检查缓存（快速路径）
    2. 未命中则加锁，再次检查（防止并发重复加载）
    3. 通过 asyncio.to_thread 将阻塞的模型加载移到线程池
    """
    global _asr_model
    if _asr_model is not None:
        return _asr_model  # 快速路径：缓存命中

    async with _asr_lock:
        if _asr_model is not None:
            return _asr_model  # 双重检查：防止竞态
        # 在线程池中加载模型（避免阻塞事件循环）
        _asr_model = await asyncio.to_thread(_load_asr_model)
        return _asr_model


def _load_asr_model():
    """
    加载 SenseVoiceSmall 模型（同步函数，由 asyncio.to_thread 调用）。

    模型路径：Config.VOICE_ASR_MODEL_DIR（默认 models/SenseVoiceSmall）

    配置参数：
    - trust_remote_code=True: 允许执行模型仓库中的自定义代码
    - disable_update=True: 禁止自动更新
    """
    model_dir = Path(Config.VOICE_ASR_MODEL_DIR)
    if not model_dir.exists():
        raise FileNotFoundError(f"ASR 模型目录不存在: {model_dir}")

    try:
        from funasr import AutoModel
    except ModuleNotFoundError as e:
        if e.name == "torch":
            raise RuntimeError(
                "ASR 依赖缺失：funasr 需要 torch。请在当前 PBL 环境中安装 torch 和 torchaudio。"
            ) from e
        raise

    logger.info(f"[voice] 开始加载 ASR 模型: {model_dir}")
    model = AutoModel(
        model=str(model_dir),
        trust_remote_code=True,
        disable_update=True,
    )
    logger.info("[voice] ASR 模型加载完成")
    return model


# ══════════════════════════════════════════════════════════════
# ASR 转录（公共 API 与内部工作器）
# ══════════════════════════════════════════════════════════════

async def transcribe_audio(audio_path: str | Path) -> str:
    """
    识别音频文件并返回文本（异步公共接口）。

    参数：audio_path — 音频文件路径（支持 WAV/MP3/OGG 等）
    返回：识别出的中文文本

    内部通过 asyncio.to_thread 将模型推理移到线程池执行，
    避免阻塞 FastAPI 的事件循环。
    """
    model = await get_asr_model()
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"音频文件不存在: {path}")

    text = await asyncio.to_thread(_generate_text, model, str(path))
    logger.debug(f"[voice] ASR 识别完成: chars={len(text)}")
    return text


def _generate_text(model, audio_path: str) -> str:
    """
    执行实际 ASR 推理（同步函数，运行在线程池中）。

    FunASR 配置：
    - language="auto": 自动检测语言
    - use_itn=True: 启用逆文本标准化（ITN），将数字/日期等转为可读形式
    - batch_size_s=60: 按 60 秒为单位批处理长音频
    - rich_transcription_postprocess: 富文本后处理（标点、分段优化）
    """
    result = model.generate(
        input=audio_path,
        cache={},            # 清空缓存，避免跨请求状态污染
        language="auto",     # 自动语言检测
        use_itn=True,        # 逆文本标准化
        batch_size_s=60,     # 批处理大小
    )
    if not result:
        return ""

    # 提取识别文本
    text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
    # FunASR 后处理：优化标点、分段
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        text = rich_transcription_postprocess(text)
    except Exception:
        pass  # 后处理失败不影响主流程，使用原始文本

    return (text or "").strip()
