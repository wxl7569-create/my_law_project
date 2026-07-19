"""民法典本地检索工具。"""

# ══════════════════════════════════════════════════════════════
# 导入标准库
# ══════════════════════════════════════════════════════════════
from __future__ import annotations

import os
import threading
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

# ══════════════════════════════════════════════════════════════
# 导入第三方库
# ══════════════════════════════════════════════════════════════
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ══════════════════════════════════════════════════════════════
# 导入内部模块
# ══════════════════════════════════════════════════════════════
from .llms import get_embedding_model
from .logger import LoggerManager

logger = LoggerManager.get_logger()

# ══════════════════════════════════════════════════════════════
# 路径与常量定义
# ══════════════════════════════════════════════════════════════

_BASE = Path(__file__).resolve().parents[2]
PAPER_PATH = _BASE / "data" / "Civil Code.docx"
CHROMA_DB_PATH = _BASE / "chroma_law_civil"
COLLECTION_NAME = "law_civil_collection"

# ══════════════════════════════════════════════════════════════
# 全局变量
# ══════════════════════════════════════════════════════════════

_vector_store: Optional[Chroma] = None
_vector_store_lock = threading.RLock()


# ══════════════════════════════════════════════════════════════
# 嵌入模型获取
# ══════════════════════════════════════════════════════════════

def get_embeddings():
    """通过兼容接口返回通义千问嵌入模型。"""
    logger.info("[RAG] 初始化民法典 embedding 模型: qwen/text-embedding-v4")
    return get_embedding_model("qwen")


# ══════════════════════════════════════════════════════════════
# 文档加载与分块
# ══════════════════════════════════════════════════════════════

def load_documents() -> List[Document]:
    """加载民法典源文档。"""
    logger.info(f"[RAG] 开始加载民法典文档: {PAPER_PATH}")
    if not PAPER_PATH.exists():
        logger.error(f"[RAG] 民法典文档不存在: {PAPER_PATH}")
        raise FileNotFoundError(f"民法典文档不存在: {PAPER_PATH}")

    file_ext = os.path.splitext(str(PAPER_PATH))[1].lower()
    supported_exts = [".txt", ".pdf", ".docx", ".doc", ".md"]
    if file_ext not in supported_exts:
        logger.error(f"[RAG] 不支持的文档类型: {file_ext}")
        raise ValueError(f"不支持的文档类型: {file_ext}")

    try:
        from langchain_unstructured import UnstructuredLoader

        loader = UnstructuredLoader(str(PAPER_PATH), mode="single", strategy="fast")
        docs = loader.load()
    except Exception as e:
        logger.error(f"[RAG] 加载民法典文档失败: {e}", exc_info=True)
        raise

    logger.info(f"[RAG] 民法典文档加载成功，共 {len(docs)} 个 Document")
    return docs


def split_documents() -> List[Document]:
    """将民法典源文档切分为可检索文本块。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "；", "，", "、", " "],
    )
    docs_split = splitter.split_documents(docs)
    logger.info(f"[RAG] 民法典文档拆分成功，共 {len(docs_split)} 个文本块")
    return docs_split


# ══════════════════════════════════════════════════════════════
# 向量库管理
# ══════════════════════════════════════════════════════════════

def get_vectorstore() -> Chroma:
    """加载或创建持久化向量库。"""
    global _vector_store
    if _vector_store is not None:
        logger.debug("[RAG] 复用已缓存的民法典向量库实例")
        return _vector_store

    with _vector_store_lock:
        if _vector_store is not None:
            return _vector_store

        embeddings_model = get_embeddings()
        vectorstore = None

        if CHROMA_DB_PATH.exists() and any(CHROMA_DB_PATH.iterdir()):
            logger.info(f"[RAG] 检测到已有民法典向量库，开始加载: {CHROMA_DB_PATH}")
            vectorstore = Chroma(
                persist_directory=str(CHROMA_DB_PATH),
                embedding_function=embeddings_model,
                collection_name=COLLECTION_NAME,
            )
            count = vectorstore._collection.count()
            if count > 0:
                _vector_store = vectorstore
                logger.info(f"[RAG] 民法典向量库加载成功，共 {count} 个向量")
                return vectorstore
            logger.warning("[RAG] 向量库目录存在但 collection 为空，将重新构建")

        logger.info("[RAG] 开始构建民法典向量库")
        docs_split = split_documents()
        vectorstore = Chroma.from_documents(
            documents=docs_split,
            embedding=embeddings_model,
            persist_directory=str(CHROMA_DB_PATH),
            collection_name=COLLECTION_NAME,
        )
        _vector_store = vectorstore
        logger.info(
            f"[RAG] 民法典向量库构建成功，路径={CHROMA_DB_PATH}，向量数={vectorstore._collection.count()}"
        )
        return vectorstore


# ══════════════════════════════════════════════════════════════
# 检索与格式化
# ══════════════════════════════════════════════════════════════

def retrieve_documents(query: str, k: int = 3) -> List[Document]:
    """根据问题检索相关民法典文本块。"""
    logger.debug(f"[RAG] 开始民法典检索: query={query!r}, k={k}")
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    logger.debug(f"[RAG] 民法典检索完成，命中 {len(docs)} 条")
    return docs


def _clean_reference_content(content: str, max_length: int = 260) -> str:
    """清理参考片段，避免前端引用区域过长。"""
    compact = " ".join((content or "").split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[:max_length].rstrip()}..."


def build_reference_section(rag_result: str, max_items: int = 3) -> str:
    """
    将 rag_law_civil 的格式化检索结果转换为可展示的 Markdown 参考依据。

    当前工具接口返回的是纯文本，为避免重构流式协议，这里从已有格式中解析
    “序号 / 来源 / 片段”，追加到最终回答下方，方便演示 RAG 可追溯性。
    """
    if not rag_result or "没有找到相关民法典文档" in rag_result:
        return ""
    if "民法典检索失败" in rag_result or "检索问题为空" in rag_result:
        return ""

    blocks = re.split(r"\n\n(?=\d+\.)", rag_result.strip())
    reference_lines = []

    for fallback_index, block in enumerate(blocks, 1):
        if len(reference_lines) >= max_items:
            break

        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        header = lines[0]
        content = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        match = re.match(r"^(?P<index>\d+)\.\s*(?:来源:\s*(?P<source>.*))?$", header)
        index = match.group("index") if match else str(fallback_index)
        source = (
            (match.group("source") or "民法典本地知识库").strip()
            if match
            else "民法典本地知识库"
        )
        excerpt = _clean_reference_content(content)
        if not excerpt:
            continue

        reference_lines.append(f"{index}. 来源：{source}\n   片段：{excerpt}")

    if not reference_lines:
        return ""

    return (
        "\n\n<details>\n<summary>参考依据（RAG 检索片段）</summary>\n\n"
        + "\n\n".join(reference_lines)
        + "\n\n</details>"
    )


def format_documents(docs: List[Document]) -> str:
    formatted = []
    for index, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        source = doc.metadata.get("source") if isinstance(doc.metadata, dict) else None
        prefix = f"{index}."
        if source:
            prefix += f" 来源: {source}"
        formatted.append(f"{prefix}\n{content}")
    return "\n\n".join(formatted)


# ══════════════════════════════════════════════════════════════
# 缓存检索与对外接口
# ══════════════════════════════════════════════════════════════

@lru_cache(maxsize=128)
def _law_civil_query_cached(query: str) -> str:
    docs = retrieve_documents(query, k=3)
    if not docs:
        logger.warning(f"[RAG] 民法典检索无结果: query={query!r}")
        return "没有找到相关民法典文档。"

    result = format_documents(docs)
    logger.debug(
        f"[RAG] 民法典检索成功: query={query!r}, hits={len(docs)}, result_length={len(result)}"
    )
    return result


def law_civil_query(query: str) -> str:
    """民法典检索工具入口。"""
    try:
        normalized_query = " ".join((query or "").split())
        if not normalized_query:
            return "检索问题为空，无法查询民法典。"
        return _law_civil_query_cached(normalized_query)
    except Exception as e:
        logger.error(f"[RAG] 民法典检索失败: query={query!r}, error={e}", exc_info=True)
        return f"民法典检索失败：{e}"


# ══════════════════════════════════════════════════════════════
# 缓存清理
# ══════════════════════════════════════════════════════════════

def clear_rag_cache() -> None:
    _law_civil_query_cached.cache_clear()


# ══════════════════════════════════════════════════════════════
# 测试入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(law_civil_query("什么叫宣告失踪"))
