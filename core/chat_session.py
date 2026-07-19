"""
═══════════════════════════════════════════════════════════════════════════════
ChatSession — 对话会话核心类

【架构定位】
对标 xiaozhi-esp32-server 的 ConnectionHandler。
每个 ChatSession 管理一次完整的对话生命周期：
- LangGraph Agent 实例（惰性初始化）
- 对话配置（thread_id, user_id）
- 自动刷入检查点

【与 xiaozhi ConnectionHandler 的差异】
xiaozhi 原版：
  - 面向 WebSocket 长连接（一个 ConnectionHandler = 一个设备连接）
  - ~1660 行

本版本：
  - 面向 REST / Gradio 风格的短连接请求
  - 管理 Agent 调用、工具执行、检查点刷入
  - 每个 session 可被多次调用（多轮对话）
  - 约 200 行

【用法】
    session = ChatSession(thread_id="user_001", user_id="张三")
    agent = await session.get_agent()
    async for event in session.stream_response(messages):
        ...
    await session.flush()
═══════════════════════════════════════════════════════════════════════════════
"""

# ══════════════════════════════════════════════════════════════
# 导入标准库与内部模块
# ══════════════════════════════════════════════════════════════

import asyncio
import json
from core.utils.logger import LoggerManager
from core.utils.models import context
from core.utils.session_keys import build_checkpoint_thread_id

logger = LoggerManager.get_logger()


class ChatSessionError(RuntimeError):
    """对话生成失败，面向路由层的脱敏异常。"""


# ══════════════════════════════════════════════════════════════
# 辅助函数 — 部分 JSON 解析与流式回放
# ══════════════════════════════════════════════════════════════

def _read_partial_json_string(raw: str, key: str) -> str:
    """从流式工具参数中读取可能尚未完整输出的 JSON 字符串值。"""
    marker = f'"{key}"'
    key_pos = raw.find(marker)
    if key_pos < 0:
        return ""

    colon_pos = raw.find(":", key_pos + len(marker))
    if colon_pos < 0:
        return ""

    pos = colon_pos + 1
    while pos < len(raw) and raw[pos].isspace():
        pos += 1
    if pos >= len(raw) or raw[pos] != '"':
        return ""

    pos += 1
    value_chars = []
    escaped = False
    while pos < len(raw):
        char = raw[pos]
        if escaped:
            value_chars.append("\\" + char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            break
        else:
            value_chars.append(char)
        pos += 1

    value = "".join(value_chars)
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return (
            value.replace(r"\n", "\n")
            .replace(r"\t", "\t")
            .replace(r"\"", '"')
            .replace(r"\\", "\\")
        )


async def _replay_text_stream(text: str, step: int = 12):
    """当模型服务不支持真实流式输出时，按小段回放累计文本。"""
    if not text:
        return
    for end in range(step, len(text), step):
        yield text[:end]
        await asyncio.sleep(0.02)
    yield text


# ══════════════════════════════════════════════════════════════
# ChatSession 类定义 — 对话会话
# ══════════════════════════════════════════════════════════════

class ChatSession:
    """
    对话会话。

    每个 ChatSession 对应一个 (thread_id, user_id) 的独立对话。
    管理 Agent 实例的惰性创建和检查点的自动刷入。

    Attributes:
        thread_id: 对话标识（LangGraph 检查点 key）
        user_id:   用户标识
    """

    # ── 构造函数 ──

    def __init__(self, thread_id: str, user_id: str = "anonymous"):
        self.thread_id = thread_id
        self.user_id = user_id
        self.checkpoint_thread_id = build_checkpoint_thread_id(user_id, thread_id)
        self._agent = None  # 惰性初始化
        self._config = None
        logger.debug(f"[会话] 创建 ChatSession: thread={thread_id}, user={user_id}")

    # ── 属性 ──

    @property
    def agent(self):
        """获取 Agent 实例（可能为 None，需先调用 get_agent()）"""
        return self._agent

    @property
    def config(self) -> dict:
        """获取 Agent 调用配置"""
        if self._config is None:
            self._config = {
                "configurable": {
                    "thread_id": self.checkpoint_thread_id,
                    "user_id": self.user_id,
                    "logical_thread_id": self.thread_id,
                    "checkpoint_ns": "",
                }
            }
            self._config["configurable"]["context"] = context(
                user_id=self.user_id, session_id=self.thread_id
            )
        return self._config

    # ── Agent 管理 ──

    async def get_agent(self):
        """
        获取（或创建）当前会话的 LangGraph Agent。

        Agent 按 thread_id 隔离，不同会话共享同一代理实例，
        但通过 config 中的 thread_id 区分检查点。
        """
        if self._agent is not None:
            return self._agent

        from core.agent_engine import get_agent
        self._agent = await get_agent()
        return self._agent

    async def rebuild_agent(self):
        """
        强制重建 Agent（用于配置变更后）。
        先刷入检查点，再清空缓存。
        """
        await self.flush()
        self._agent = None
        return await self.get_agent()

    # ── 核心对话：流式响应 ──

    async def stream_response(self, messages: list):
        """
        流式生成 Agent 回复。

        参数 messages 是 Gradio 格式的对话历史：
        [{"role": "user", "content": [...]}, {"role": "assistant", "content": [...]}]

        Yields:
            每次 yield 当前累积的完整回复文本
        """
        agent = await self.get_agent()
        full_text = ""
        tool_args_by_index: dict[int, str] = {}
        last_tool_answer = ""
        from core.utils.tools import begin_tool_call_round, end_tool_call_round, get_tool_call_round

        tool_round_token = begin_tool_call_round()

        try:
            async for event in agent.astream_events(
                {"messages": messages},
                config=self.config,
                version="v2",
            ):
                if event.get("event") == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        full_text += chunk.content
                        yield full_text
                        continue

                    for tool_chunk in getattr(chunk, "tool_call_chunks", []) or []:
                        index = tool_chunk.get("index", 0)
                        args_delta = tool_chunk.get("args") or ""
                        if not args_delta:
                            continue
                        tool_args_by_index[index] = (
                            tool_args_by_index.get(index, "") + args_delta
                        )
                        answer = _read_partial_json_string(
                            tool_args_by_index[index],
                            "answer",
                        )
                        if answer and len(answer) > len(last_tool_answer):
                            full_text = answer
                            last_tool_answer = answer
                            yield full_text

            tool_state = get_tool_call_round()
            if full_text and tool_state is not None and tool_state.rag_success:
                from core.utils.rag_law_civil import build_reference_section

                reference_section = build_reference_section(tool_state.rag_result)
                if reference_section and reference_section not in full_text:
                    full_text = f"{full_text.rstrip()}{reference_section}"
                    yield full_text

            # 回退：一次调用
            if not full_text:
                response = await agent.ainvoke(
                    {"messages": messages}, config=self.config
                )
                structured = response.get("structured_response")
                if structured and getattr(structured, "answer", None):
                    full_text = structured.answer
                elif response.get("messages"):
                    last = response["messages"][-1]
                    full_text = getattr(last, "content", "") or "抱歉，未能获取有效回答。"
                else:
                    full_text = "抱歉，未能获取有效回答。"
                async for replayed in _replay_text_stream(full_text):
                    yield replayed

                tool_state = get_tool_call_round()
                if tool_state is not None and tool_state.rag_success:
                    from core.utils.rag_law_civil import build_reference_section

                    reference_section = build_reference_section(tool_state.rag_result)
                    if reference_section and reference_section not in full_text:
                        full_text = f"{full_text.rstrip()}{reference_section}"
                        yield full_text

        except Exception as e:
            logger.error(
                f"[会话] Agent 调用失败 thread={self.thread_id}, user={self.user_id}: "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )
            raise ChatSessionError("Agent 调用失败") from e
        finally:
            end_tool_call_round(tool_round_token)

    # ── 核心对话：非流式调用 ──

    async def invoke(self, messages: list) -> str:
        """
        一次性调用 Agent（非流式）。

        Returns:
            完整回复文本
        """
        full_text = ""
        async for chunk in self.stream_response(messages):
            full_text = chunk
        return full_text

    # ── 检查点管理 ──

    async def flush(self):
        """刷入当前会话的缓冲检查点"""
        if self._agent is None:
            return
        try:
            cp = getattr(self._agent, "checkpointer", None)
            if cp is not None and hasattr(cp, "flush"):
                await cp.flush()
        except Exception as e:
            logger.warning(f"[会话] 刷入检查点失败: {e}")

    # ── 会话管理 ──

    def close(self):
        """清理会话资源"""
        self._agent = None
        self._config = None
        logger.debug(f"[会话] 关闭 ChatSession: thread={self.thread_id}")


# ══════════════════════════════════════════════════════════════
# 会话管理器（Session Registry）
# ══════════════════════════════════════════════════════════════

_sessions: dict = {}  # (user_id, thread_id) -> ChatSession


def get_chat_session(thread_id: str, user_id: str = "anonymous") -> ChatSession:
    """
    获取或创建 ChatSession。

    使用 (user_id, thread_id) 复合键隔离不同用户的会话，
    避免不同用户使用相同 thread_id 时发生串会话。

    Args:
        thread_id: 对话标识
        user_id: 用户标识

    Returns:
        ChatSession 实例
    """
    global _sessions
    key = (user_id, thread_id)
    if key not in _sessions:
        _sessions[key] = ChatSession(thread_id, user_id)
    return _sessions[key]


async def flush_all_sessions():
    """刷入所有活跃会话的检查点"""
    global _sessions
    for key, session in list(_sessions.items()):
        try:
            await session.flush()
        except Exception as e:
            logger.warning(f"[会话] 刷入失败 key={key}: {e}")


def close_all_sessions():
    """关闭所有活跃会话"""
    global _sessions
    _sessions.clear()
    logger.info("[会话] 所有会话已关闭")
