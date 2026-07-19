"""增量文本切分，用于伪流式 TTS。"""

# ══════════════════════════════════════════════════════════════
# 导入标准库与第三方库
# ══════════════════════════════════════════════════════════════

from __future__ import annotations

from core.voice.text_normalizer import normalize_for_tts


# ══════════════════════════════════════════════════════════════
# 句子分割器类
# ══════════════════════════════════════════════════════════════

class TtsSentenceSplitter:
    """把流式增量文本切成适合快速合成的短句。"""

    # ── Initialization ──

    def __init__(self, max_chars: int = 45):
        self.max_chars = max_chars
        self._buffer = ""

    # ── Public API (Feed & Flush) ──

    def feed(self, delta: str) -> list[str]:
        if not delta:
            return []

        self._buffer += delta
        return self._pop_ready_segments(force=False)

    def flush(self) -> list[str]:
        return self._pop_ready_segments(force=True)

    # ── Segment Extraction Logic ──

    def _pop_ready_segments(self, force: bool) -> list[str]:
        segments: list[str] = []
        while self._buffer:
            cut_at = self._find_sentence_end(self._buffer)
            if cut_at < 0 and len(self._buffer) >= self.max_chars:
                cut_at = self._find_soft_cut(self._buffer)
            if cut_at < 0:
                break

            raw = self._buffer[:cut_at].strip()
            self._buffer = self._buffer[cut_at:].lstrip()
            normalized = normalize_for_tts(raw)
            if normalized:
                segments.append(normalized)

        if force and self._buffer.strip():
            normalized = normalize_for_tts(self._buffer)
            self._buffer = ""
            if normalized:
                segments.append(normalized)

        return segments

    # ── Sentence Boundary Detection ──

    @staticmethod
    def _find_sentence_end(text: str) -> int:
        ends = [text.find(ch) for ch in "。！？；!?;" if text.find(ch) >= 0]
        if not ends:
            return -1
        return min(ends) + 1

    def _find_soft_cut(self, text: str) -> int:
        window = text[: self.max_chars]
        for ch in "，,、 ":
            pos = window.rfind(ch)
            if pos >= 12:
                return pos + 1
        return min(len(text), self.max_chars)
