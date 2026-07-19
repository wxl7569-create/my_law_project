"""
LangGraph 法律咨询 Agent — CLI 入口

【设计说明】
薄 CLI 入口，核心逻辑委托给 core/agent_engine.py。
效仿 xiaozhi-esp32-server 的 app.py → core/ 设计模式。

【用法】
    python agent.py [thread_id] [user_id]

【环境变量】
API 密钥统一在 PBL_pro/.env 中管理。
"""

# ══════════════════════════════════════════════════════════════
# 导入标准库与运行环境初始化
# ══════════════════════════════════════════════════════════════

import asyncio
import sys

from core.bootstrap import setup_runtime_env
setup_runtime_env()

from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════
# 导入核心模块
# ══════════════════════════════════════════════════════════════

from core.agent_engine import run_conversation
from core.chat_session import flush_all_sessions, close_all_sessions
from core.utils.logger import LoggerManager

logger = LoggerManager.get_logger()


# ══════════════════════════════════════════════════════════════
# 主异步入口
# ══════════════════════════════════════════════════════════════

async def main():
    """命令行入口"""
    try:
        # 从命令行参数获取 thread_id 和 user_id
        thread_id = sys.argv[1] if len(sys.argv) > 1 else "1"
        user_id = sys.argv[2] if len(sys.argv) > 2 else "1"
        # 异步运行对话引擎
        await run_conversation(thread_id, user_id)
    finally:
        # 刷入检查点 + 关闭会话 + 关闭数据库连接
        from core.utils.memory_sqlite import close_sqlite_saver
        await flush_all_sessions()
        close_all_sessions()
        await close_sqlite_saver()  # 确保 SQLite 连接在进程退出前关闭
        logger.info("资源已释放，程序退出")


# ══════════════════════════════════════════════════════════════
# 启动点
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    asyncio.run(main())
