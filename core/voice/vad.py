"""可选 VAD 支持。

Gradio 当前传入的是完整录音文件，第一阶段不强制启用 VAD。
如果依赖或模型不可用，直接返回原音频路径，避免影响主流程。
"""

# ══════════════════════════════════════════════════════════════
# 导入标准库与第三方库
# ══════════════════════════════════════════════════════════════

from __future__ import annotations

from pathlib import Path

from core.config.settings import Config
from core.utils.logger import LoggerManager

logger = LoggerManager.get_logger()


# ══════════════════════════════════════════════════════════════
# 静音裁剪（公共 API）
# ══════════════════════════════════════════════════════════════

async def maybe_trim_silence(audio_path: str | Path) -> Path:
    """预留静音裁剪入口；当前默认不阻塞主流程。"""
    path = Path(audio_path)
    if not Config.VOICE_ENABLE_VAD:
        return path

    try:
        model_dir = Path(Config.VOICE_VAD_MODEL_DIR)
        if not model_dir.exists():
            logger.warning(f"[voice] VAD 模型目录不存在，跳过静音裁剪: {model_dir}")
            return path
        logger.info("[voice] VAD 已启用，当前版本保留原始音频进入 ASR")
        return path
    except Exception as e:
        logger.warning(f"[voice] VAD 处理失败，使用原始音频: {e}", exc_info=True)
        return path
