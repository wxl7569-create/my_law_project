"""基于数据库和检查点的会话列表与历史恢复辅助函数。"""

# ══════════════════════════════════════════════════════════════
# 导入标准库
# ══════════════════════════════════════════════════════════════
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, List

# ══════════════════════════════════════════════════════════════
# 导入第三方库
# ══════════════════════════════════════════════════════════════
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# ══════════════════════════════════════════════════════════════
# 导入内部模块
# ══════════════════════════════════════════════════════════════
from core.config.settings import Config
from core.utils.logger import LoggerManager
from core.utils.session_keys import build_checkpoint_thread_id, parse_checkpoint_thread_id

logger = LoggerManager.get_logger()

# ══════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════

SESSION_MAIN = "default"
DEFAULT_SESSION_NAME = "默认对话"


# ══════════════════════════════════════════════════════════════
# 核心类定义：会话管理器
#
# 负责：
# 1. 列出用户的所有会话（从 chat_sessions 表或 checkpoints 表中恢复）
# 2. 保存/更新会话信息（thread_id -> name 映射）
# 3. 从检查点恢复聊天历史（反序列化 LangGraph checkpoint → Gradio 格式）
# ══════════════════════════════════════════════════════════════

class SessionManager:
    """读取已有对话会话并恢复前端展示历史。"""

    # ── 初始化 ──
    def __init__(self):
        self.db_path = Config.MEMORY_DB_PATH         # 检查点数据库路径（memory.db）
        self.session_db_path = Config.USERS_DB_PATH  # 用户数据库路径（users.db，含 chat_sessions 表）
        self.serde = JsonPlusSerializer()            # LangGraph 检查点序列化器

    # ── 异步上下文管理器（兼容 async with 语法） ──
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    # ── 公开接口：列出用户的会话列表 ──
    def list_sessions(self, user_id: str) -> List[dict]:
        """
        获取用户的所有会话列表。

        优先级：
        1. 从 chat_sessions 表读取（最新的会话管理方式）
        2. 若为空，回退到从 checkpoints 表恢复（兼容旧数据迁移）

        返回格式：[{"id": thread_id, "name": 会话名, "updated": 更新时间}, ...]
        """
        if not user_id:
            return []

        # 步骤 1：尝试从 chat_sessions 表读取
        saved_sessions = self._list_saved_sessions(user_id)
        if saved_sessions:
            return self._sorted_sessions({item["id"]: item for item in saved_sessions})

        # 步骤 2：回退——从检查点自行恢复
        legacy_sessions = self._list_checkpoint_sessions(user_id)
        # 将恢复的会话回写到 chat_sessions 表，后续直接读取
        for item in legacy_sessions:
            self.save_session(user_id, item["id"], item["name"])
        return legacy_sessions

    def ensure_default_session(self, user_id: str) -> dict:
        """
        确保用户至少有一个"默认对话"会话。

        新用户首次登录时无任何会话，需要创建一个默认会话。
        """
        session = {"id": user_id, "name": DEFAULT_SESSION_NAME, "updated": ""}
        self.save_session(user_id, session["id"], session["name"])
        return session

    # ── 公开接口：保存会话 ──
    def save_session(self, user_id: str, thread_id: str, name: str) -> None:
        """
        保存或更新会话名称到 chat_sessions 表。

        使用 INSERT ... ON CONFLICT DO UPDATE 实现 upsert：
        - 新会话：插入新记录
        - 已有会话：更新名称和 updated_at 时间
        """
        if not user_id or not thread_id:
            return

        try:
            self._init_session_db()  # 确保 chat_sessions 表和索引存在
            with sqlite3.connect(self.session_db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO chat_sessions (user_id, thread_id, name, updated_at)
                    VALUES (?, ?, ?, datetime('now', 'localtime'))
                    ON CONFLICT(user_id, thread_id) DO UPDATE SET
                        name = excluded.name,
                        updated_at = datetime('now', 'localtime')
                    """,
                    (user_id, thread_id, name or self._extract_session_name(thread_id, user_id)),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.warning(f"[session] failed to save session index: {e}")

    # ── 公开接口：从检查点恢复聊天历史 ──
    def load_history(self, thread_id: str, user_id: str) -> list[dict]:
        """
        从检查点数据库恢复会话的聊天历史。

        流程：
        1. 查询最新的 checkpoint（按 checkpoint_id DESC LIMIT 1）
        2. 反序列化 checkpoint → 提取 messages
        3. 将 LangGraph 消息转换为 Gradio chatbot 格式

        兼容性：支持旧的未编码 thread_id（legacy 方式）和新编码格式
        """
        if not thread_id or not user_id or not os.path.exists(self.db_path):
            return []

        try:
            with sqlite3.connect(self.db_path) as conn:
                # 尝试新格式编码的 thread_id
                row = self._fetch_checkpoint_row(
                    conn,
                    build_checkpoint_thread_id(user_id, thread_id),
                )
                if row is None:
                    # 兼容旧格式（未编码的 thread_id）
                    legacy_row = self._fetch_checkpoint_row(conn, thread_id)
                    if legacy_row is not None and self._metadata_user_id(legacy_row["metadata"]) == user_id:
                        row = legacy_row
        except sqlite3.Error as e:
            logger.warning(f"[session] failed to load checkpoint history: {e}")
            return []

        if not row:
            return []

        # 反序列化检查点（type + checkpoint 二进制的 typed JSON 格式）
        try:
            checkpoint = self.serde.loads_typed((row["type"], row["checkpoint"]))
        except Exception as e:
            logger.warning(f"[session] failed to deserialize checkpoint: {e}")
            return []

        # 提取消息并转换为 Gradio 格式
        messages = checkpoint.get("channel_values", {}).get("messages", [])
        return self._to_gradio_history(messages)

    @staticmethod
    def _fetch_checkpoint_row(conn: sqlite3.Connection, thread_id: str):
        """
        从 checkpoints 表查询最新的检查点。

        SQL: SELECT type, checkpoint, metadata FROM checkpoints
             WHERE thread_id = ? ORDER BY checkpoint_id DESC LIMIT 1
        """
        conn.row_factory = sqlite3.Row  # 结果以字典形式返回
        return conn.execute(
            """
            SELECT type, checkpoint, metadata
            FROM checkpoints
            WHERE thread_id = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
            """,
            (thread_id,),
        ).fetchone()

    # ══════════════════════════════════════════════════════════
    # 内部方法
    # ══════════════════════════════════════════════════════════

    # ── 数据库初始化：创建 chat_sessions 表和索引 ──
    def _init_session_db(self) -> None:
        """
        初始化会话管理表（幂等操作）。

        chat_sessions 表结构：
        - id: 自增主键
        - user_id + thread_id: 复合唯一索引（一个用户不能有重复 thread_id）
        - name: 会话显示名称
        - created_at / updated_at: 时间戳

        索引 idx_chat_sessions_user_updated 加速按用户+时间查询。
        """
        os.makedirs(os.path.dirname(self.session_db_path), exist_ok=True)
        with sqlite3.connect(self.session_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
                    updated_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
                    UNIQUE(user_id, thread_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
                ON chat_sessions(user_id, updated_at)
                """
            )
            conn.commit()

    # ── 读取已保存的会话 ──
    def _list_saved_sessions(self, user_id: str) -> list[dict]:
        try:
            self._init_session_db()
            with sqlite3.connect(self.session_db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT thread_id, name, updated_at
                    FROM chat_sessions
                    WHERE user_id = ?
                    ORDER BY updated_at DESC
                    """,
                    (user_id,),
                ).fetchall()
        except sqlite3.Error as e:
            logger.warning(f"[session] failed to list saved sessions: {e}")
            return []

        return [
            {
                "id": row["thread_id"],
                "name": row["name"],
                "updated": str(row["updated_at"] or ""),
            }
            for row in rows
        ]

    # ── 检查点恢复会话列表（兼容旧数据） ──
    def _list_checkpoint_sessions(self, user_id: str) -> list[dict]:
        """
        从 checkpoints 表中恢复会话列表（兼容旧数据迁移）。

        遍历所有 checkpoints 行，按 (thread_id, namespace) 去重，
        筛选属于当前用户的会话（通过复合 thread_id 编码或 metadata.user_id 判断）。
        """
        if not os.path.exists(self.db_path):
            return []

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT thread_id, checkpoint_ns, checkpoint_id, metadata
                    FROM checkpoints
                    ORDER BY checkpoint_id DESC
                    """
                ).fetchall()
        except sqlite3.Error as e:
            logger.warning(f"[session] failed to list checkpoints: {e}")
            return []

        sessions: dict[str, dict] = {}
        seen: set[tuple[str, str]] = set()  # (thread_id, namespace) 去重
        for row in rows:
            checkpoint_thread_id = row["thread_id"]
            # 解析复合 thread_id（新格式）或直接使用（旧格式）
            owner_user_id, thread_id = parse_checkpoint_thread_id(checkpoint_thread_id)
            namespace = row["checkpoint_ns"]

            # 按用户隔离：新格式按 user_id 过滤，旧格式按 metadata 过滤
            metadata_user_id = self._metadata_user_id(row["metadata"])
            if owner_user_id is not None and owner_user_id != user_id:
                continue  # 不属于当前用户，跳过
            if owner_user_id is None and metadata_user_id != user_id:
                continue  # 旧格式且 metadata 不匹配，跳过

            key = (thread_id, namespace)
            if key in seen:
                continue  # 去重
            seen.add(key)

            sessions[thread_id] = {
                "id": thread_id,
                "name": self._extract_session_name(thread_id, user_id),
                "updated": str(row["checkpoint_id"] or "")[:16],
            }

        return self._sorted_sessions(sessions)

    # ── 元数据解析：从 metadata 字段提取 user_id ──
    @staticmethod
    def _metadata_user_id(raw) -> str | None:
        """
        从检查点 metadata 列中解析 user_id。

        metadata 可能是 bytes（需解码）或 JSON 字符串。
        """
        if raw is None:
            return None
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            meta = json.loads(raw)  # JSON 解析
            if isinstance(meta, dict):
                return meta.get("user_id")  # 返回 user_id 字段
        except Exception:
            return None
        return None

    # ── 会话按更新时间倒序排列 ──
    @staticmethod
    def _sorted_sessions(sessions: dict[str, dict]) -> list[dict]:
        """按 updated 字段降序排列会话列表。"""
        result = list(sessions.values())
        result.sort(key=lambda item: item.get("updated") or "", reverse=True)
        return result

    # ── 检查点消息转 Gradio 聊天历史格式 ──
    @staticmethod
    def _to_gradio_history(messages: list[Any]) -> list[dict]:
        """
        将 LangGraph 消息列表转换为 Gradio chatbot 格式。

        LangGraph 消息类型映射：
        - human → {"role": "user", "content": "..."}
        - ai → {"role": "assistant", "content": "..."}
        - tool (responseformat) → 从 tool_call 的 args 中提取 answer

        最后对相邻重复项去重（agent 回答可能在人机之间产生重复）。
        """
        history: list[dict] = []
        for message in messages:
            msg_type = getattr(message, "type", "")         # 消息类型
            content = getattr(message, "content", "") or ""  # 消息内容

            if msg_type == "human" and content:
                history.append(_gr_msg("user", str(content)))
                continue

            if msg_type == "ai":
                answer = SessionManager._extract_ai_answer(message)
                if answer:
                    history.append(_gr_msg("assistant", answer))
                continue

            # tool 类型 + 名为 responseformat → 从中提取答案文本
            if msg_type == "tool" and getattr(message, "name", "") == "responseformat":
                answer = SessionManager._extract_responseformat_answer(str(content))
                if answer:
                    history.append(_gr_msg("assistant", answer))

        return SessionManager._dedupe_adjacent(history)

    # ── 从 AI 消息提取回答文本 ──
    @staticmethod
    def _extract_ai_answer(message: Any) -> str:
        """
        从 LangGraph AIMessage 中提取回答文本。

        优先从 content 字段获取；若 content 为空，
        则从 tool_calls 中查找 responseformat 并提取其 answer 参数。
        """
        content = getattr(message, "content", "") or ""
        if content:
            return str(content)

        # 回退：从 tool_calls 中提取结构化回答
        for tool_call in getattr(message, "tool_calls", []) or []:
            if tool_call.get("name") != "responseformat":
                continue
            args = tool_call.get("args") or {}
            answer = args.get("answer")
            if answer:
                return str(answer)
        return ""

    # ── 从 responseformat 工具调用提取回答 ──
    @staticmethod
    def _extract_responseformat_answer(content: str) -> str:
        marker = "responseformat(answer="
        start = content.find(marker)
        if start < 0:
            return ""

        value_start = start + len(marker)
        if value_start >= len(content) or content[value_start] not in {"'", '"'}:
            return ""

        quote = content[value_start]
        pos = value_start + 1
        chars: list[str] = []
        escaped = False
        while pos < len(content):
            char = content[pos]
            if escaped:
                chars.append("\\" + char)
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                break
            else:
                chars.append(char)
            pos += 1

        raw = "".join(chars)
        try:
            return json.loads(f'"{raw}"')
        except json.JSONDecodeError:
            return (
                raw.replace(r"\n", "\n")
                .replace(r"\t", "\t")
                .replace(r"\'", "'")
                .replace(r"\"", '"')
                .replace(r"\\", "\\")
            )

    # ── 相邻重复历史去重 ──
    @staticmethod
    def _dedupe_adjacent(history: list[dict]) -> list[dict]:
        result: list[dict] = []
        for item in history:
            if result and result[-1] == item:
                continue
            result.append(item)
        return result

    # ── 从 thread_id 提取会话名称 ──
    @staticmethod
    def _extract_session_name(thread_id: str, user_id: str) -> str:
        if thread_id == user_id or thread_id == f"{user_id}|{SESSION_MAIN}":
            return DEFAULT_SESSION_NAME
        prefix = f"{user_id}|"
        if thread_id.startswith(prefix):
            return thread_id[len(prefix):] or "未命名对话"
        return thread_id

    # ── 构造 thread_id ──
    @staticmethod
    def build_thread_id(user_id: str, session_name: str = SESSION_MAIN) -> str:
        if not session_name or session_name in {SESSION_MAIN, DEFAULT_SESSION_NAME}:
            return user_id
        return f"{user_id}|{session_name}"

    # ── 解析 thread_id ──
    @staticmethod
    def parse_thread_id(thread_id: str) -> tuple[str, str]:
        if "|" in thread_id:
            return tuple(thread_id.split("|", 1))
        return thread_id, SESSION_MAIN


# ══════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════

def _gr_msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}
