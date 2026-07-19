"""
Chroma 向量数据库构建脚本。

用于独立重建民法典 RAG 向量库。
执行前需确保 data/Civil Code.docx 存在，且 .env 中配置了 DASHSCOPE_API_KEY。

用法：
    python scripts/build_chroma.py
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Imports & Bootstrap
# ═══════════════════════════════════════════════════════════════════════════════

import sys
import os

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bootstrap import setup_runtime_env
setup_runtime_env()

from dotenv import load_dotenv
load_dotenv()

from core.utils.logger import LoggerManager
from core.utils.rag_law_civil import get_vectorstore

logger = LoggerManager.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  民法典 Chroma 向量库构建工具")
    print("=" * 60)

    # 检查 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("[错误] 未设置 DASHSCOPE_API_KEY，请在 .env 中配置后重试。")
        sys.exit(1)

    try:
        vectorstore = get_vectorstore()
        count = vectorstore._collection.count()
        print(f"\n[完成] 向量库构建成功！共 {count} 个文档块。")
    except FileNotFoundError as e:
        print(f"\n[错误] 民法典文件缺失: {e}")
        print("请确保 data/Civil Code.docx 存在后重试。")
        sys.exit(1)
    except Exception as e:
        print(f"\n[错误] 构建失败: {e}")
        raise


if __name__ == "__main__":
    main()
