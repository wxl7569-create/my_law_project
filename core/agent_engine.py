"""
LangGraph 法律咨询 Agent 引擎

【设计说明】
本模块提供基于 LangGraph 的法律咨询 Agent 的工厂函数。
不管理会话状态——会话管理由 core/chat_session.py 的 ChatSession 负责。

效仿 xiaozhi-esp32-server 的架构模式：
- agent.py 只作为薄 CLI 入口
- 核心 Agent 工厂在此模块（类似 xiaozhi 的 initialize_modules）
- 会话生命周期在 chat_session.py（类似 xiaozhi 的 ConnectionHandler）

【用法】
    from core.agent_engine import get_agent
    agent = await get_agent()
    # 智能体是共享实例。
    # 通过 config 中的 thread_id 区分不同会话
"""

import sys
import asyncio

# ══════════════════════════════════════════════════════════════
# 运行环境初始化
# ══════════════════════════════════════════════════════════════

from core.bootstrap import setup_runtime_env
setup_runtime_env()

from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════
# 导入核心模块
# ══════════════════════════════════════════════════════════════

from core.utils.logger import LoggerManager
from core.utils.models import context, responseformat

logger = LoggerManager.get_logger()


# ══════════════════════════════════════════════════════════════
# 共享 Agent 实例（单例）
# ══════════════════════════════════════════════════════════════

# 主问答智能体本身是线程安全的（不持有会话状态），
# 多个 ChatSession 可以共享同一 agent 实例，通过 config 隔离检查点。
_agent_instance = None


# ══════════════════════════════════════════════════════════════
# 系统提示词
# ══════════════════════════════════════════════════════════════

def get_system_prompt() -> str:
    """
    获取 Agent 系统提示词。

    提示词采用结构化设计，包含角色定位、工具使用规则、输出格式要求等，
    确保 LLM 能够准确理解任务并生成符合预期的响应。
    """
    return """
你是一名专业的法律咨询助手。你的核心职责是依据中国法律法规，为用户提供准确、简洁、易懂的法律解答。

【角色定位】
- 身份：专业法律顾问，熟悉《民法典》及相关法律法规
- 语言风格：回答需简明扼要，避免冗长，仅使用中文逗号和句号
- 回复原则：直接针对问题作答，无需寒暄，除非用户明确要求详细解释

【工具使用规则】

规则1: 民法典检索 (rag_law_civil)
  - 优先级最高：任何法律相关问题（合同、物权、婚姻家庭、继承、侵权等）必须优先调用
  - 必须引用：检索到相关法条时，务必在回答中引用具体条款内容
  - 兜底回答：即使检索结果为空，也需基于专业知识给出合理回答

规则2: 网络搜索 (tavily_search)
  - 补充检索：民法典检索无结果时使用
  - 实时信息：需要最新政策、案例或非法律类信息时使用
  - 辅助验证：用于验证法律条文的最新有效性

规则3: 文件解析 (word_reader)
  - 触发条件：当用户上传文件内容或基于文件内容提问时必须调用
  - 解析优先：先解析文件内容，再根据文件内容回答用户问题
  - 结果整合：将文件解析结果与法律知识结合，给出综合回答

【输出格式要求】
请严格按照以下 JSON 格式输出，确保可被程序解析：
{
    "answer": "你的回答内容，中文，简洁明了",
    "tool_used": "使用的工具名称，未调用则为空字符串",
    "law_civil": "引用的民法典具体条款内容，未调用 rag_law_civil 则为空字符串",
    "search_results": "网络搜索结果摘要，未调用 tavily_search 则为空字符串"
}

【注意事项】
- 必须严格遵循 JSON 格式，不得使用 Markdown 或其他格式
- 法律条文引用需准确，注明条款编号
- 如无法回答，应明确说明并建议咨询专业律师
- 回答内容仅限中文，避免使用英文术语
"""


# ══════════════════════════════════════════════════════════════
# Agent 工厂函数（惰性单例）
# ══════════════════════════════════════════════════════════════

async def get_agent():
    """
    获取共享的 LangGraph Agent 实例（惰性初始化，单例）。

    所有 LangChain/LangGraph 重型导入在此函数内部进行，
    外部调用 get_agent() 时才触发。

    Returns:
        LangChain Agent 实例（线程安全，可跨会话共享）
    """
    global _agent_instance
    if _agent_instance is not None:
        return _agent_instance

    logger.info("开始初始化 Agent ...")

    try:
        from langchain.agents import create_agent
        from langchain.agents.structured_output import ToolStrategy
        from core.config.settings import Config
        from core.utils.tools import get_tools
        from core.utils.llms import get_chat_model
        from core.utils.memory_sqlite import get_batched_sqlite_saver
    except ImportError as e:
        logger.error(f"Agent 依赖未安装: {e}")
        raise ImportError(
            f"Agent 初始化失败: {e}\n"
            f"请安装依赖: pip install langchain langgraph langchain-chroma langchain-community"
        )

    checkpointer = await get_batched_sqlite_saver(batch_size=5)
    llm_chat = get_chat_model(Config.LLM_TYPE)
    tools = get_tools()

    _agent_instance = create_agent(
        model=llm_chat,
        system_prompt=get_system_prompt(),
        tools=tools,
        context_schema=context,
        checkpointer=checkpointer,
        response_format=ToolStrategy(responseformat),
    )

    logger.info("Agent 初始化完成")
    return _agent_instance


# ══════════════════════════════════════════════════════════════
# CLI 交互式对话循环
# ══════════════════════════════════════════════════════════════

async def run_conversation(thread_id: str = "1", user_id: str = "1"):
    """
    CLI 交互式对话循环。

    使用 ChatSession 管理会话状态，而非直接操作 agent。
    这样 run_conversation 聚焦于 CLI 交互逻辑，
    而 Agent 调用和检查点管理交给了 ChatSession。
    """
    from core.chat_session import get_chat_session, flush_all_sessions

    session = get_chat_session(thread_id, user_id)
    agent = await session.get_agent()

    print(f"开始运行会话: 线程ID {thread_id}, 用户ID {user_id}")
    print("输入 '退出' 或 'exit' 结束会话")

    while True:
        user_input = await asyncio.to_thread(input, "用户: ")
        user_input = user_input.strip()
        if user_input.lower() in ["退出", "exit"]:
            # 退出前刷入检查点
            await flush_all_sessions()
            print("会话结束")
            break

        # 通过 ChatSession 的流式接口调用
        messages = [{"role": "user", "content": user_input}]
        full_response = ""
        async for chunk in session.stream_response([{"role": "system", "content": m}
                                                      if isinstance(m, str) else m
                                                      for m in messages]):
            full_response = chunk

        # 简化的输出
        print(f"助手: {full_response[:200]}{'...' if len(full_response) > 200 else ''}")
        logger.info(f"对话记录: 用户输入: {user_input}, agent回复: {full_response[:100]}")


# ══════════════════════════════════════════════════════════════
# 启动入口（兼容 agent.py 调用）
# ══════════════════════════════════════════════════════════════

async def main():
    """CLI 入口（兼容 agent.py 的调用目标）"""
    try:
        thread_id = sys.argv[1] if len(sys.argv) > 1 else "1"
        user_id = sys.argv[2] if len(sys.argv) > 2 else "1"
        await run_conversation(thread_id, user_id)
    finally:
        from core.chat_session import flush_all_sessions, close_all_sessions
        from core.utils.memory_sqlite import close_sqlite_saver
        await flush_all_sessions()
        close_all_sessions()
        await close_sqlite_saver()  # 确保 SQLite 连接在进程退出前关闭
        logger.info("资源已释放，程序退出")


if __name__ == "__main__":
    asyncio.run(main())
