# ══════════════════════════════════════════════════════════════
# 各类LLM的初始化配置
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# 导入标准库
# ══════════════════════════════════════════════════════════════
import os

# ══════════════════════════════════════════════════════════════
# 导入第三方库
# ══════════════════════════════════════════════════════════════
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════
# 模型配置
# ══════════════════════════════════════════════════════════════

# 配置不同模型和接口的相关参数，包括基础URL、模型名称和API密钥等。使用过程中可以根据需要选择不同的模型和接口进行调用。
MODEL_CONFIG = {
    "deepseek":{
        # 基础url
        "base_url":"https://api.deepseek.com",
        # 模型名称
        "chat_model":["deepseek-chat","deepseek-v4-flash"],
        # 接口密钥
        "api_key":os.getenv("DEEPSEEK_API_KEY"),
    },
    "qwen":{
        # 基础url
        "base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1",
        # 模型名称
        "chat_model":["qwen3.6-plus"],
        "embedding_model":["text-embedding-v4"],  
        # 接口密钥
        "api_key":os.getenv("DASHSCOPE_API_KEY"),
    }
}

# ══════════════════════════════════════════════════════════════
# 默认配置常量
# ══════════════════════════════════════════════════════════════

# 默认模型
DEFAULT_LLM_TYPE = "deepseek"
# 默认嵌入模型
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
# 默认温度
DEFAULT_TEMPERATURE = 0
# 默认最大上下文长度
DEFAULT_MAX_CONTEXT_LENGTH = 4096

# ══════════════════════════════════════════════════════════════
# 自定义异常类
# ══════════════════════════════════════════════════════════════

# 自定义异常类，大模型初始化失败时抛出
class LLMInitializationError(Exception):
    """大模型初始化失败异常类"""
    pass


# ══════════════════════════════════════════════════════════════
# 导入内部模块与日志初始化
# ══════════════════════════════════════════════════════════════

from .logger import LoggerManager
logger = LoggerManager.get_logger()


# ══════════════════════════════════════════════════════════════
# 大模型初始化函数
# ══════════════════════════════════════════════════════════════

def init_chat_model(llm_type: str = DEFAULT_LLM_TYPE, model_name: str = None) -> "ChatOpenAI":
    from langchain_openai import ChatOpenAI

    logger.info(f"开始初始化聊天模型: {llm_type}")

    try:
        config = MODEL_CONFIG.get(llm_type)
        if not config:
            raise ValueError(f"不支持的聊天模型类型: {llm_type}")
        
        if model_name is None:
            model_name = config["chat_model"][0]    # 取第一个模型或指定模型
        else:
            if model_name not in config["chat_model"]:
                logger.warning(
                    f"模型 {model_name} 不在支持列表 {config['chat_model']} 中，仍将尝试初始化"
                    )
                raise ValueError(f"模型 {model_name} 不在配置的聊天模型列表中: {config['chat_model']}")

        chat_model = ChatOpenAI(
            base_url=config["base_url"],
            model=model_name,  
            api_key=config["api_key"],
            temperature=DEFAULT_TEMPERATURE,
            streaming=True,
            timeout=60,
            max_retries=3,
        )
        logger.info(f"聊天模型初始化成功: {llm_type} - {model_name}")
        return chat_model
    except Exception as e:
        logger.error(f"聊天模型初始化失败: {llm_type}, 错误: {str(e)}", exc_info=True)
        raise LLMInitializationError(f"聊天模型初始化失败: {str(e)}")

def init_embedding_model(embedding_type: str = "qwen") -> "OpenAIEmbeddings":
    from langchain_openai import OpenAIEmbeddings

    logger.info(f"开始初始化嵌入模型: {embedding_type}")
    try:
        config = MODEL_CONFIG.get(embedding_type)
        if not config or "embedding_model" not in config:
            raise ValueError(f"模型 {embedding_type} 未配置嵌入模型")
        
        embedding_model = OpenAIEmbeddings(
            base_url=config["base_url"],
            model=config["embedding_model"][0],
            api_key=config["api_key"],
            timeout=60,
            max_retries=3,
            default_headers={"X-DashScope-Async": "disable"},  # 添加需要的头部信息
            check_embedding_ctx_length=False,  # 禁用检查上下文长度
        )
        logger.info(f"嵌入模型初始化成功: {embedding_type} - {config['embedding_model']}")
        return embedding_model
    except Exception as e:
        logger.error(f"嵌入模型初始化失败: {embedding_type}, 错误: {str(e)}", exc_info=True)
        raise LLMInitializationError(f"嵌入模型初始化失败: {str(e)}")


# ══════════════════════════════════════════════════════════════
# 对外封装函数
# ══════════════════════════════════════════════════════════════

# 获取聊天模型的封装函数，默认使用deepseek-chat模型
def get_chat_model(llm_type: str = DEFAULT_LLM_TYPE, model_name: str = None) -> "ChatOpenAI":
    """
    获取聊天模型的封装函数，默认使用deepseek-chat模型

    args:
        llm_type: 模型类型，默认为deepseek
        model_name: 模型名称，默认为配置中的第一个模型

    returns:
        ChatOpenAI实例
    """
    try:
        return init_chat_model(llm_type, model_name)     # 初始化聊天模型
    except LLMInitializationError as e:
        logger.warning(
            f"聊天模型初始化失败: {str(e)}，将使用默认模型 {model_name}重试", exc_info=True,
            )    # 记录错误日志
        raise e

# 获取嵌入模型的封装函数，默认使用qwen的text-embedding-v4模型
def get_embedding_model(embedding_type: str = "qwen") -> "OpenAIEmbeddings":
    """
    获取嵌入模型的封装函数，默认使用qwen的text-embedding-v4模型

    args:
        embedding_type: 嵌入模型类型，默认为qwen

    returns:
        OpenAIEmbeddings实例
    """
    try:
        return init_embedding_model(embedding_type)     # 初始化嵌入模型
    except LLMInitializationError as e:
        logger.warning(
            f"嵌入模型初始化失败: {str(e)}，将使用默认模型 {DEFAULT_EMBEDDING_MODEL}重试", exc_info=True,
            )    # 记录错误日志
        raise e


# ══════════════════════════════════════════════════════════════
# 测试入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

# -----------测试1: 默认聊天模型deepseek-chat--------------

    try:
        print("\n[测试1] 获取默认聊天模型...")
        chat = get_chat_model()   # 使用默认聊天模型
        chat_model_name1 = getattr(chat, 'model', None) or getattr(chat, 'model_name', 'unknown')
        print(f"聊天模型初始化成功，模型名称: {chat_model_name1}")
        response = chat.invoke("你好！")
        content = response.content if response.content else "(空响应)"
        print(f"响应: {content[:20]}")
    except Exception as e:
        print(f"测试1失败: {e}")

    # ---------- 测试2: 指定其他聊天模型 (deepseek-v4-flash) ----------
    try:
        print("\n[测试2] 获取 deepseek-v4-flash 聊天模型...")
        chat2 = get_chat_model(model_name="deepseek-v4-flash")  # 指定模型名称
        chat_model_name2 = getattr(chat2, 'model', None) or getattr(chat2, 'model_name', 'unknown')
        print(f"聊天模型初始化成功，模型名称: {chat_model_name2}")
        response = chat2.invoke("你是谁？")
        content = response.content if response.content else "(空响应)"
        print(f"响应: {content[:20]}")
    except Exception as e:
        print(f"测试2失败: {e}")

    # ---------- 测试3: 默认嵌入模型 (qwen text-embedding-v4) ----------
    try:
        print("\n[测试3] 获取默认嵌入模型...")
        embed = get_embedding_model()   # 默认 embedding_type="qwen"
        # 获取嵌入模型名称（通常存储在 model 属性）
        embed_model_name = getattr(embed, 'model', None) or getattr(embed, 'model_name', 'unknown')
        test_text = "测试文本"
        print(f"输入类型: {type(test_text)}")  # 应该是 <class 'str'>
        vec = embed.embed_query(test_text)
        print(f"嵌入向量维度: {len(vec)}")
        print(f"嵌入模型初始化成功，模型名称: {embed_model_name}")
    except Exception as e:
        print(f"测试3失败: {e}")
