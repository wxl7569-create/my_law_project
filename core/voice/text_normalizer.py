"""面向 TTS 的文本清洗。

文字界面仍显示 Agent 原始回答；进入 TTS 前只使用这里的清洗结果，
避免朗读 Markdown、表情、链接、代码块等与语义无关的内容。
"""

# ══════════════════════════════════════════════════════════════
# 导入标准库与第三方库
# ══════════════════════════════════════════════════════════════

from __future__ import annotations

import re

# ══════════════════════════════════════════════════════════════
# 正则表达式模式
# ══════════════════════════════════════════════════════════════

_CODE_BLOCK_RE = re.compile(r"```.*?```", re.S)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u27bf"
    "]+",
    flags=re.UNICODE,
)
_MD_SYMBOL_RE = re.compile(r"[*_>#|~]+")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_LIST_PREFIX_RE = re.compile(r"(?m)^\s*(?:[-+*]|\d+[.)、]|[一二三四五六七八九十]+[、.])\s*")
_SPACES_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n+")
_NOISE_ONLY_RE = re.compile(r"^[，。！？；：、,.!?;:\-\s]+$")


# ══════════════════════════════════════════════════════════════
# 公共 API
# ══════════════════════════════════════════════════════════════

def normalize_for_tts(text: str) -> str:
    """将 Agent 文本转换为适合朗读的自然文本。"""
    if not text:
        return ""

    cleaned = _CODE_BLOCK_RE.sub("。", text)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _INLINE_CODE_RE.sub(r"\1", cleaned)
    cleaned = _URL_RE.sub("", cleaned)
    cleaned = _EMOJI_RE.sub("", cleaned)
    cleaned = _HEADING_RE.sub("", cleaned)
    cleaned = _LIST_PREFIX_RE.sub("", cleaned)
    cleaned = _MD_SYMBOL_RE.sub("", cleaned)
    cleaned = cleaned.replace("\\n", "\n")
    cleaned = cleaned.replace("\r", "\n")
    cleaned = cleaned.replace("：", "。")
    cleaned = cleaned.replace(":", "。")
    cleaned = _SPACES_RE.sub(" ", cleaned)
    cleaned = _BLANK_LINES_RE.sub("。", cleaned)

    parts = []
    for part in re.split(r"(?<=[。！？；!?;])", cleaned):
        item = part.strip(" \n\t，,、")
        if not item or _NOISE_ONLY_RE.match(item):
            continue
        parts.append(item)

    result = "".join(parts).strip()
    result = re.sub(r"\s+([，。！？；、,.!?;])", r"\1", result)
    result = re.sub(r"。{2,}", "。", result)
    return result
