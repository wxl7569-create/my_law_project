"""智能体工具注册。"""

# ══════════════════════════════════════════════════════════════
# 导入标准库
# ══════════════════════════════════════════════════════════════
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass

# ══════════════════════════════════════════════════════════════
# 导入第三方库
# ══════════════════════════════════════════════════════════════
from langchain.tools import tool

# ══════════════════════════════════════════════════════════════
# 导入内部模块
# ══════════════════════════════════════════════════════════════
from .logger import LoggerManager
from .rag_law_civil import law_civil_query
from .word_reader import parse_file_and_inject

logger = LoggerManager.get_logger()


# ══════════════════════════════════════════════════════════════
# 数据模型：工具调用轮次状态
# ══════════════════════════════════════════════════════════════

@dataclass
class ToolCallRound:
    rag_query: str = ""
    rag_result: str = ""
    rag_success: bool = False


# ══════════════════════════════════════════════════════════════
# 全局上下文变量与状态管理
# ══════════════════════════════════════════════════════════════

_tool_call_round: ContextVar[ToolCallRound | None] = ContextVar(
    "tool_call_round",
    default=None,
)


def begin_tool_call_round():
    """创建单轮工具调用状态，并返回用于重置的令牌。"""
    return _tool_call_round.set(ToolCallRound())


def get_tool_call_round() -> ToolCallRound | None:
    """返回当前对话轮次的工具调用状态，用于最终回答展示检索依据。"""
    return _tool_call_round.get()


def end_tool_call_round(token) -> None:
    _tool_call_round.reset(token)


# ══════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════

def _is_successful_rag_result(result: str) -> bool:
    if not result:
        return False
    failure_markers = ("没有找到相关民法典文档", "民法典检索失败", "检索问题为空")
    return not any(marker in result for marker in failure_markers)


# ══════════════════════════════════════════════════════════════
# 工具注册接口
# ══════════════════════════════════════════════════════════════

def get_tools():
    """返回法律问答智能体可用的全部工具。"""

    @tool("rag_law_civil", description="优先使用民法典本地向量库查询相关法律条文和解释。每轮对话通常只需调用一次。")
    async def rag_law_civil(query: str) -> str:
        state = _tool_call_round.get()
        if state is not None and state.rag_success:
            logger.debug(
                "[Tool] 本轮已完成 rag_law_civil 检索，复用首次结果: "
                f"first_query={state.rag_query!r}, skipped_query={query!r}"
            )
            return (
                "本轮已完成一次民法典检索，以下复用首次检索结果，"
                "请基于该结果直接组织回答，避免继续重复检索。\n\n"
                f"{state.rag_result}"
            )

        logger.info(f"[Tool] 调用 rag_law_civil: {query}")
        result = await asyncio.to_thread(law_civil_query, query)
        logger.debug(f"[Tool] rag_law_civil 返回长度: {len(result)}")

        if state is not None:
            state.rag_query = query
            state.rag_result = result
            state.rag_success = _is_successful_rag_result(result)
            if not state.rag_success:
                logger.debug("[Tool] rag_law_civil 首次检索未成功，本轮允许后续再次检索")

        return result

    @tool("tavily_search", description="当民法典本地检索无结果或需要补充实时信息时，使用 Tavily 网络搜索。")
    async def tavily_search(query: str) -> str:
        logger.info(f"[Tool] 调用 tavily_search: {query}")
        from .tavily_search import tavily_search as tavily_search_func

        result = await asyncio.to_thread(tavily_search_func, query)
        logger.debug(f"[Tool] tavily_search 返回长度: {len(result)}")
        return result

    @tool("word_reader", description="读取并解析用户上传的 .txt、.docx、.doc、.md 文件内容。传入文件绝对路径。")
    async def word_reader(file_path: str) -> str:
        logger.info(f"[Tool] 调用 word_reader: {file_path}")
        result = await asyncio.to_thread(parse_file_and_inject, file_path)
        logger.debug(f"[Tool] word_reader 返回长度: {len(result)}")
        return result

    tools = [rag_law_civil, tavily_search, word_reader]
    logger.info(f"工具函数加载完成，共 {len(tools)} 个工具")
    return tools
