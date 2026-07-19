# ══════════════════════════════════════════════════════════════
# 网络检索工具
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# 导入标准库
# ══════════════════════════════════════════════════════════════
import os
from typing import Optional

# ══════════════════════════════════════════════════════════════
# 导入第三方库
# ══════════════════════════════════════════════════════════════
from tavily import TavilyClient

# ══════════════════════════════════════════════════════════════
# 导入内部模块
# ══════════════════════════════════════════════════════════════
from .logger import LoggerManager

logger = LoggerManager.get_logger()

# ══════════════════════════════════════════════════════════════
# 环境变量与全局变量
# ══════════════════════════════════════════════════════════════

import dotenv
dotenv.load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    logger.warning("Tavily API Key 未配置")

_client : Optional[TavilyClient] = None


# ══════════════════════════════════════════════════════════════
# Tavily 客户端获取
# ══════════════════════════════════════════════════════════════

def get_tavily_client()->TavilyClient:
    """获取Tavily客户端实例"""
    global _client
    if _client is None:
        if not TAVILY_API_KEY:
            raise ValueError("Tavily API Key 未配置")
        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client


# ══════════════════════════════════════════════════════════════
# 核心检索函数
# ══════════════════════════════════════════════════════════════

def tavily_search(query: str,max_results: int = 5)->str:
    """
    使用tavily进行检索，返回格式化后的结果

    args:
        query: 搜索查询
        max_results: 最大返回结果数，默认5

    return:
        格式化后的检索结果

    """
    try:
        client = get_tavily_client()
        logger.debug(f"开始使用Tavily检索: {query}")
        response = client.search(query, max_results=max_results)

        results = response.get("results", [])
        if not results:
            return "没有找到相关结果"
        
        formatted_results = []
        for i,res in enumerate(results,1):
            title = res.get("title", "无标题")
            url = res.get("url", "无URL")
            content = res.get("content", "无内容").strip()
            # 限制内容长度
            if len(content) > 300:
                content = content[:300] + " ..."
            formatted_results.append(f"{i}. {title}\n ({url})\n {content}\n")

        result_text = "\n\n".join(formatted_results)
        logger.debug(f"检索完成，共找到 {len(results)} 条结果")
        return result_text
    except Exception as e:
        error_msg = f"检索失败: {e}"
        logger.error(error_msg)
        return error_msg
    

# ══════════════════════════════════════════════════════════════
# 测试入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "民法典包含几章"
    print(f"测试搜索: {query}")
    print(tavily_search(query,max_results=3))
