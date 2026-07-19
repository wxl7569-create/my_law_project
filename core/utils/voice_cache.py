"""语音缓存文件清理工具。"""

from __future__ import annotations

import time
from pathlib import Path

from core.config.settings import Config
from core.utils.logger import LoggerManager

logger = LoggerManager.get_logger()


def _cache_dir() -> Path:
    return Path(Config.VOICE_AUDIO_TEMP_DIR).resolve()


def _is_cache_file(path: Path) -> bool:
    try:
        path.resolve().relative_to(_cache_dir())
        return path.is_file()
    except ValueError:
        return False


def delete_voice_file(path: str | Path | None) -> None:
    """删除单个语音缓存文件，只允许删除语音缓存目录内的文件。"""
    if not path:
        return

    target = Path(path)
    if not _is_cache_file(target):
        return

    try:
        target.unlink(missing_ok=True)
    except OSError as e:
        logger.warning(f"[voice] 删除语音缓存失败: {target}: {e}")


def cleanup_voice_cache() -> None:
    """按 TTL 和目录总大小清理语音缓存。"""
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    now = time.time()
    files: list[tuple[float, int, Path]] = []
    for path in cache_dir.iterdir():
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue

        age = now - stat.st_mtime
        if age > Config.VOICE_CACHE_TTL_SECONDS:
            delete_voice_file(path)
            continue
        files.append((stat.st_mtime, stat.st_size, path))

    total_size = sum(size for _, size, _ in files)
    if total_size <= Config.VOICE_CACHE_MAX_BYTES:
        return

    for _, size, path in sorted(files):
        delete_voice_file(path)
        total_size -= size
        if total_size <= Config.VOICE_CACHE_MAX_BYTES:
            break
