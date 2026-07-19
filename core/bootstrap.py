"""
统一运行时环境初始化。

在所有入口文件（agent.py、app.py、core/api/server.py）的最顶部调用，
确保 SSL 证书路径、LangGraph msgpack 序列化行为等配置一致。

【来源】
替代原有的全局 monkey patch ssl.create_default_context 和
零散的 os.environ["SSL_CERT_FILE"]="" 等高风险写法。
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════════════════════

import os
import certifi


# ═══════════════════════════════════════════════════════════════════════════════
# Runtime Environment Setup
# ═══════════════════════════════════════════════════════════════════════════════

def setup_runtime_env():
    """设置统一的运行时环境变量，所有入口文件在 import 其他模块前调用。"""
    # 图状态序列化严格模式统一关闭，优先保证兼容性。
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "false")
    # 使用 certifi 提供的 CA 证书包，而非系统证书存储
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
