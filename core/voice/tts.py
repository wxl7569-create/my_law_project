"""本地 TTS 合成，支持 pyttsx3（离线）和 edge-tts（在线）。

【设计模式】
- 策略模式：BaseTTSProvider 抽象基类定义 synthesize 接口
- Pyttsx3TTSProvider: Windows SAPI 本地离线合成（.wav）
- EdgeTTSProvider: 微软 Edge 在线语音合成（.mp3）
- 惰性单例工厂：get_tts_provider() 按 VOICE_TTS_PROVIDER 配置选择策略

【线程安全】
- pyttsx3 是非线程安全的，通过专用线程池 _tts_executor (max_workers=1) 串行化
- edge-tts 是异步库，直接在事件循环中 await
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════
# 导入标准库与第三方库
# ══════════════════════════════════════════════════════════════

import asyncio              # 异步锁、run_in_executor
import time                 # 时间戳（音频文件名）
import uuid                 # 唯一标识（音频文件名）
from abc import ABC, abstractmethod  # 抽象基类
from concurrent.futures import ThreadPoolExecutor  # 专用线程池（pyttsx3 串行化）
from pathlib import Path

from core.config.settings import Config
from core.utils.logger import LoggerManager
from core.utils.voice_cache import cleanup_voice_cache
from core.voice.text_normalizer import normalize_for_tts  # 文本清洗（去除 Markdown/表情等）

logger = LoggerManager.get_logger()

# ══════════════════════════════════════════════════════════════
# 模块级状态 — 线程池与 TTS 提供器缓存
# ══════════════════════════════════════════════════════════════

# pyttsx3 非线程安全，使用 max_workers=1 的专用线程池串行化调用
_tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-tts")
_provider = None               # TTS 提供器单例（Pyttsx3TTSProvider 或 EdgeTTSProvider）
_provider_lock = asyncio.Lock()  # 异步锁，防止并发初始化


# ══════════════════════════════════════════════════════════════
# 抽象基类 — TTS 提供器接口
# ══════════════════════════════════════════════════════════════

class BaseTTSProvider(ABC):
    """语音合成提供器基类——所有 TTS 实现必须继承此类。"""

    @abstractmethod
    async def synthesize(self, text: str, output_path: Path) -> Path:
        """
        合成音频并保存到 output_path，返回文件路径。

        参数：
        - text: 已清洗的待朗读文本
        - output_path: 输出音频文件路径

        返回：合成后的音频文件路径
        """


# ══════════════════════════════════════════════════════════════
# Pyttsx3 TTS 实现 — Windows SAPI 离线合成
#
# 使用 Windows 系统内置的 SAPI 引擎（如 Microsoft Huihui Desktop）
# 优点：完全离线、无网络延迟、免费
# 缺点：音质不如云端方案、非线程安全
# ══════════════════════════════════════════════════════════════

class Pyttsx3TTSProvider(BaseTTSProvider):
    """Windows 本地 SAPI 离线 TTS。"""

    async def synthesize(self, text: str, output_path: Path) -> Path:
        """
        异步合成入口——将同步 pyttsx3 调用委托给专用线程池。
        使用 run_in_executor 避免阻塞事件循环。
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _tts_executor,
            self._synthesize_sync,
            text,
            output_path,
        )

    def _synthesize_sync(self, text: str, output_path: Path) -> Path:
        """
        同步合成实现（在专用线程池中执行）。

        配置项（来自 .env）：
        - VOICE_TTS_VOICE: 指定 SAPI 语音名（如 "Microsoft Huihui Desktop"）
        - VOICE_TTS_RATE: 语速（默认 180，正常≈200）
        - VOICE_TTS_VOLUME: 音量 0.0~1.0
        """
        try:
            import pyttsx3
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "TTS 依赖缺失：当前使用 pyttsx3，请在 PBL 环境中执行 "
                "`pip install pyttsx3`，或设置 VOICE_TTS_PROVIDER=edge 并安装 edge-tts。"
            ) from e

        engine = pyttsx3.init()  # 初始化 SAPI 引擎
        if Config.VOICE_TTS_VOICE:
            engine.setProperty("voice", Config.VOICE_TTS_VOICE)  # 设置语音角色
        engine.setProperty("rate", Config.VOICE_TTS_RATE)         # 设置语速
        engine.setProperty("volume", Config.VOICE_TTS_VOLUME)     # 设置音量
        engine.save_to_file(text, str(output_path))  # 合成并保存为 .wav
        engine.runAndWait()  # 阻塞等待合成完成
        engine.stop()        # 释放引擎资源
        return output_path


# ══════════════════════════════════════════════════════════════
# Edge TTS 实现 — 微软在线语音合成
#
# 使用 edge-tts 库调用微软 Edge 浏览器的免费 TTS 服务
# 优点：音质好、自然度高、支持多种语音角色
# 缺点：需要网络连接
# ══════════════════════════════════════════════════════════════

class EdgeTTSProvider(BaseTTSProvider):
    """可通过配置选择的 edge-tts 语音合成实现。"""

    async def synthesize(self, text: str, output_path: Path) -> Path:
        """
        使用 edge-tts 异步合成语音。

        配置项：
        - VOICE_TTS_EDGE_VOICE: 语音角色（默认 zh-CN-XiaoxiaoNeural）
        - VOICE_TTS_EDGE_RATE: 语速调整（如 "+10%",  "+0%"）
        - VOICE_TTS_EDGE_VOLUME: 音量调整（如 "+0%"）

        输出格式：.mp3
        """
        try:
            import edge_tts
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "TTS 依赖缺失：当前使用 edge-tts，请在 PBL 环境中执行 "
                "`pip install edge-tts`，或设置 VOICE_TTS_PROVIDER=pyttsx3 并安装 pyttsx3。"
            ) from e

        # Communicate 对象管理整个合成会话
        communicate = edge_tts.Communicate(
            text=text,
            voice=Config.VOICE_TTS_EDGE_VOICE,      # 语音角色
            rate=Config.VOICE_TTS_EDGE_RATE,        # 语速调整
            volume=Config.VOICE_TTS_EDGE_VOLUME,    # 音量调整
        )
        await communicate.save(str(output_path))  # 异步下载合成音频
        return output_path


# ══════════════════════════════════════════════════════════════
# 提供器工厂函数（惰性单例 + 双重检查锁）
# ══════════════════════════════════════════════════════════════

async def get_tts_provider() -> BaseTTSProvider:
    """
    获取 TTS 提供器实例（惰性单例）。

    根据 VOICE_TTS_PROVIDER 环境变量选择策略：
    - "edge" → EdgeTTSProvider（在线语音合成）
    - 其他值（默认 "pyttsx3"）→ Pyttsx3TTSProvider（离线语音合成）
    """
    global _provider
    if _provider is not None:
        return _provider  # 快速路径

    async with _provider_lock:
        if _provider is not None:
            return _provider  # 双重检查

        provider_name = Config.VOICE_TTS_PROVIDER
        if provider_name == "edge":
            _provider = EdgeTTSProvider()
        else:
            _provider = Pyttsx3TTSProvider()
        logger.info(f"[voice] TTS Provider 已选择: {provider_name}")
        return _provider


# ══════════════════════════════════════════════════════════════
# 辅助函数 — 生成输出路径
# ══════════════════════════════════════════════════════════════

def _new_audio_path(provider_name: str) -> Path:
    """
    在语音缓存目录中生成唯一的音频输出路径。

    文件名格式：tts_{时间戳}_{8位随机ID}.{扩展名}
    - pyttsx3 → .wav
    - edge → .mp3
    """
    Config.VOICE_AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    cleanup_voice_cache()  # 触发缓存清理（超过 TTL 的文件会被删除）
    suffix = ".mp3" if provider_name == "edge" else ".wav"
    name = f"tts_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}{suffix}"
    return Config.VOICE_AUDIO_TEMP_DIR / name


# ══════════════════════════════════════════════════════════════
# 公开 API — 文本转语音（完整的清洗→合成→返回路径流程）
# ══════════════════════════════════════════════════════════════

async def synthesize_text(text: str) -> Path | None:
    """
    清洗并合成一段语音文本。

    流程：
    1. normalize_for_tts: 去除 Markdown/代码块/表情/URL 等不可朗读内容
    2. 获取 TTS 提供器（pyttsx3 或 edge-tts）
    3. 生成唯一输出路径
    4. 执行合成
    5. 返回音频文件路径

    返回：合成后的音频路径，清洗后文本为空则返回 None
    """
    clean_text = normalize_for_tts(text)  # 步骤 1：文本清洗
    if not clean_text:
        return None  # 清洗后为空，跳过合成

    provider = await get_tts_provider()   # 步骤 2：获取提供器
    output_path = _new_audio_path(Config.VOICE_TTS_PROVIDER)  # 步骤 3：生成路径
    logger.debug(f"[voice] 开始 TTS 合成: chars={len(clean_text)}")
    path = await provider.synthesize(clean_text, output_path)  # 步骤 4：合成
    logger.debug(f"[voice] TTS 合成完成: {path}")
    return path
