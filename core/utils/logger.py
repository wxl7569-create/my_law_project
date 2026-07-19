"""
日志管理器

【设计说明】
提供统一日志接口，两个输出通道：
1. 文件输出（app.log）：DEBUG 级别，全量日志
2. 控制台输出：INFO 级别，仅关键节点信息

"""

# ══════════════════════════════════════════════════════════════
# 导入标准库
# ══════════════════════════════════════════════════════════════
import logging
from logging.handlers import RotatingFileHandler
#导入日志模块
import sys
import io
#输入输出模块

# ══════════════════════════════════════════════════════════════
# 导入第三方库
# ══════════════════════════════════════════════════════════════
try:
    from concurrent_log_handler import ConcurrentRotatingFileHandler
except ModuleNotFoundError:
    ConcurrentRotatingFileHandler = RotatingFileHandler
#导入并发日志文件处理模块

# ══════════════════════════════════════════════════════════════
# 导入内部模块
# ══════════════════════════════════════════════════════════════
from core.config.settings import Config
#导入配置模块


# ══════════════════════════════════════════════════════════════
# 控制台日志过滤器
# ══════════════════════════════════════════════════════════════

class _ConsoleFilter(logging.Filter):
    """
    控制台日志过滤器

    只允许以下内容通过到终端：
    - INFO 级别且包含特定标记的消息（[用户提问]、[Agent回答]）
    - WARNING 及以上级别（错误必须显示）
    """
    # 允许显示在终端的关键标记前缀
    _KEY_PREFIXES = (
        "[用户提问", "[Agent回答]",
    )

    def filter(self, record: logging.LogRecord) -> bool:            #record为日志记录对象，包含日志级别、消息等信息
        # 警告、错误和严重错误全部显示。
        if record.levelno >= logging.WARNING:
            return True
        # 普通信息级别只显示含关键标记的消息。
        if record.levelno == logging.INFO:
            msg = record.getMessage()
            return any(msg.startswith(p) for p in self._KEY_PREFIXES)
        # 调试级别和未设置级别不显示。
        return False


# ══════════════════════════════════════════════════════════════
# 核心类定义：日志管理器（单例模式）
# ══════════════════════════════════════════════════════════════

class LoggerManager:
    """日志管理器（单例模式）"""

    _instance = None        # 单例实例，确保全局只有一个LoggerManager实例
    _logger = None          # 日志记录器实例

    # 单例模式，确保全局只有一个LoggerManager实例
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # 初始化日志记录器
    def __init__(self):
        if self._logger is None:
            self._set_logger()
            self._suppress_noisy_loggers()

    # ── 第三方库日志压制 ──
    def _suppress_noisy_loggers(self):
        """压制第三方库的噪音日志"""
        # 压制根日志器（funasr 的 WARNING:root:trust_remote_code 等）
        logging.getLogger().setLevel(logging.ERROR)
        # 压制 funasr / langgraph / 网络库的详细日志
        for name in [
            "funasr", "langgraph", "langchain", "uvicorn",
            "httpx", "httpcore", "urllib3", "asyncio",
        ]:
            logging.getLogger(name).setLevel(logging.WARNING)

    # ── 日志记录器配置 ──
    def _set_logger(self):
        """配置日志记录器"""
        self._logger = logging.getLogger("pbl_legal_agent")
        file_level = getattr(logging, Config.LOG_LEVEL, logging.INFO)
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers = []
        # 禁止传播到 root logger，由本模块统一管理,防止重复日志
        self._logger.propagate = False

        # ── 确保控制台 stdout 使用 UTF-8 编码（解决 Windows 中文乱码） ──
        if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.upper() != "UTF-8":
            try:
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            except Exception:
                pass

        # ── 1. 文件处理器（全量日志，DEBUG 级别，UTF-8 编码） ──
        file_handler = ConcurrentRotatingFileHandler(
            Config.LOG_FILE,
            maxBytes=Config.MAX_BYTES,
            backupCount=Config.BACKUP_COUNT,
            encoding="utf-8",  # ✅ 显式指定 UTF-8，避免 Windows 默认编码导致乱码
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        self._logger.addHandler(file_handler)

        # ── 2. 控制台处理器（仅关键节点，INFO 级别） ──
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            "%(message)s"
        ))
        console_handler.addFilter(_ConsoleFilter())
        self._logger.addHandler(console_handler)

    # ── 属性与类方法 ──
    @property
    def logger(self):
        return self._logger

    @classmethod
    def get_logger(cls):
        instance = cls()
        return instance.logger
