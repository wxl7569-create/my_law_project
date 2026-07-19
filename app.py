"""
法律咨询系统的网页前端入口。

【架构角色】
基于 Gradio 的 Web 前端，提供用户认证、会话管理、文字/语音问答的统一界面。
前端不直接调用智能体——所有业务逻辑通过后端 API (FastAPI) 完成。

【数据流】
用户输入 → Gradio 事件处理 → 后端 API → 智能体 → 流式响应 → 前端展示
"""
from __future__ import annotations

# ══════════════════════════════════════════════════════════════
# 导入标准库
# ══════════════════════════════════════════════════════════════
import atexit         # 注册进程退出时的清理回调（关闭后端子进程、恢复代理设置）
import os             # 读写环境变量（HTTP 代理清理/恢复）
import socket         # TCP 端口检测（判断后端是否已运行）
import subprocess     # 启动后端子进程（uvicorn 服务）
import sys            # 获取 Python 解释器路径，用于启动子进程
import time           # 轮询等待后端端口就绪时的 sleep

# ══════════════════════════════════════════════════════════════
# 运行环境初始化（必须在 import 其他模块前调用）
# ══════════════════════════════════════════════════════════════
from core.bootstrap import setup_runtime_env

setup_runtime_env()  # 设置 SSL 证书路径、LangGraph 序列化行为等运行时环境

# ══════════════════════════════════════════════════════════════
# 导入 Gradio 与核心模块
# ══════════════════════════════════════════════════════════════
import gradio as gr    # Web UI 框架，提供 Chatbot/Button/Audio 等组件

from core.app_handlers import AGENT_API_URL, GUIDE_EXAMPLES  # 后端地址 + 快捷提问示例
from core.config.settings import Config                      # 统一配置（端口、主机等）
from core.register_handler import handle_login, handle_logout, handle_register  # 认证事件处理
from core.utils.chat_wrappers import wrap_handle_text        # 文字聊天包装（含登录校验）
from core.utils.voice_wrappers import wrap_handle_voice      # 语音聊天包装（含登录校验）
from core.utils.session_handlers import create_new_session, find_thread_id, switch_session  # 会话管理


# ══════════════════════════════════════════════════════════════
# 前端样式（覆盖 Gradio 默认样式）
# ══════════════════════════════════════════════════════════════
custom_css = """
.gradio-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
footer { display: none !important; }
"""
# 以上 CSS 移除 Gradio 默认 padding/margin 实现全宽布局，并隐藏 Gradio 页脚

# ══════════════════════════════════════════════════════════════
# 后端进程管理（前端自动拉起后端）
# ══════════════════════════════════════════════════════════════

_backend_process: subprocess.Popen | None = None  # 后端子进程对象，用于前端退出时终止


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """通过尝试 TCP 连接检测端口是否已被占用。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True  # 连接成功 = 端口已被监听
    except OSError:
        return False   # 连接失败 = 端口空闲


def _ensure_backend_running() -> None:
    """
    当前端单独启动时，自动拉起本地后端 uvicorn 服务。

    流程：
    1. 检查 backend 端口 (8001) 是否已被占用
    2. 如未占用，以子进程方式启动 `python -m uvicorn core.api.server:app`
    3. 轮询最多 6 秒（30次×0.2秒）等待端口就绪
    4. 子进程 stdout/stderr 丢弃，避免污染前端日志
    """
    global _backend_process
    host = "127.0.0.1"
    port = Config.API_PORT       # 默认 8001
    if _is_port_open(host, port):
        return                   # 后端已在运行，无需拉起

    creationflags = 0
    if os.name == "nt":
        # Windows 下隐藏子进程控制台窗口，避免弹出黑框
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    _backend_process = subprocess.Popen(
        [
            sys.executable,              # 当前 Python 解释器
            "-m", "uvicorn",             # 使用 uvicorn 模块启动
            "core.api.server:app",       # FastAPI 应用入口
            "--host", Config.API_HOST,   # 绑定主机（默认 0.0.0.0）
            "--port", str(port),         # 绑定端口（默认 8001）
        ],
        cwd=str(Config.BASE_DIR),       # 工作目录设为项目根目录
        stdout=subprocess.DEVNULL,       # 丢弃标准输出
        stderr=subprocess.DEVNULL,       # 丢弃标准错误
        creationflags=creationflags,     # Windows 下隐藏控制台窗口
    )

    # 轮询等待后端端口就绪（最多 30 次 × 0.2 秒 = 6 秒）
    for _ in range(30):
        if _backend_process.poll() is not None:
            return                   # 子进程已退出（启动失败），直接返回
        if _is_port_open(host, port):
            return                   # 端口就绪，后端已启动
        time.sleep(0.2)              # 等待 200ms 后重试


# ══════════════════════════════════════════════════════════════
# 页面布局 — 认证区域（登录 / 注册）
# ══════════════════════════════════════════════════════════════

with gr.Blocks(title="法律咨询助手", fill_width=True) as demo:
    # ── 全局状态变量 ──
    # login_state: 存储当前用户的认证信息（token, user_id, name, phone, logged_in）
    login_state = gr.State(
        {"logged_in": False, "token": "", "user_id": "", "name": "", "phone": ""}
    )
    # session_list: 当前用户的会话列表，每个元素为 {"id": ..., "name": ..., "updated": ...}
    session_list = gr.State([])
    # current_thread_id: 当前选中会话的线程ID（LangGraph checkpoint key）
    current_thread_id = gr.State("")

    # ── 页面标题头 ──
    gr.Markdown(
        "<div style='text-align:center;background-color:#e0e0e0;padding:10px;border-radius:5px;'>"
        "<h1>法律咨询助手</h1>"
        f"<p>后端服务：{AGENT_API_URL}</p>"  # 显示当前后端地址，便于调试
        "</div>"
    )

    # ── 认证区域（auth_block: 初始可见，登录后隐藏） ──
    with gr.Column(visible=True) as auth_block:
        auth_status = gr.Markdown("")  # 显示登录/注册结果消息（成功/失败提示）
        with gr.Tabs():
            # 登录 Tab
            with gr.Tab("登录"):
                login_phone = gr.Textbox(label="手机号")           # 11位手机号输入
                login_password = gr.Textbox(label="密码", type="password")  # 密码（掩码显示）
                login_btn = gr.Button("登录", variant="primary")
            # 注册 Tab
            with gr.Tab("注册"):
                register_name = gr.Textbox(label="姓名")           # 用户姓名
                register_phone = gr.Textbox(label="手机号")        # 11位手机号
                register_password = gr.Textbox(label="密码", type="password")
                register_confirm = gr.Textbox(label="确认密码", type="password")  # 二次确认
                register_btn = gr.Button("注册并登录", variant="primary")

    # ══════════════════════════════════════════════════════════════
    # 页面布局 — 聊天区域（会话列表 / 文字输入 / 语音输入）
    # chat_block: 登录后可见，包含完整的对话交互界面
    # ══════════════════════════════════════════════════════════════

    with gr.Column(visible=False) as chat_block:  # 初始隐藏，登录成功后切换为可见
        # ── 顶栏：用户信息 + 退出按钮 ──
        with gr.Row():
            with gr.Column(scale=4):
                user_info = gr.Markdown("欢迎用户")  # 登录后动态更新为"用户: 姓名 | 手机号"
            with gr.Column(scale=1):
                logout_btn = gr.Button("退出", variant="stop", size="sm")

        # ── 主体：左侧会话列表 + 右侧聊天区域 ──
        with gr.Row():
            with gr.Column(scale=1):
                new_session_btn = gr.Button("新建对话", size="sm", min_width=100)  # 创建新对话会话
                session_dropdown = gr.Radio(label="会话列表", choices=[], interactive=True)  # 可点击切换的会话列表

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=500, placeholder="等待您的提问...")  # 聊天历史展示组件

                # ── 底部输入栏 ──
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1, min_width=160):
                                file = gr.File(
                                    file_count="multiple",   # 支持多文件上传
                                    type="filepath",         # 返回文件本地路径
                                    label="上传文件",
                                    height=80,
                                )
                            with gr.Column(scale=5):
                                msg = gr.Textbox(
                                    placeholder="在此输入法律问题，支持上传 .txt / .docx / .doc / .md 文件",
                                    label="文字输入",
                                    container=True,
                                    interactive=True,
                                    lines=4,
                                    max_lines=8,
                                )
                            with gr.Column(scale=1, min_width=110):
                                send_btn = gr.Button("发送", variant="primary", size="sm")   # 发送文字消息
                                clear_btn = gr.Button("清空对话", variant="stop", size="sm")  # 清空当前聊天历史

                        gr.Examples(examples=GUIDE_EXAMPLES, inputs=msg, label="快速提问")  # 预设示例问题，点击即填入

                    # ── 右侧语音输入/输出 ──
                    with gr.Column(scale=1):
                        audio_in = gr.Audio(
                            sources=["microphone"],  # 允许浏览器麦克风录音
                            type="filepath",         # 录音后返回本地文件路径
                            label="语音输入",
                        )
                        voice_btn = gr.Button("语音发送", variant="secondary")  # 提交录音进行语音问答
                        tts_audio = gr.Audio(
                            label="语音回复",
                            type="filepath",
                            autoplay=True,           # 收到音频后自动播放
                            visible=False,           # 初始隐藏，有语音回复时才显示
                        )

    # ══════════════════════════════════════════════════════════════
    # 事件绑定 — 认证事件（登录 / 注册 / 退出）
    #
    # 每个 .click() 链式调用格式：
    #   component.click(handler_fn, [inputs], [outputs])
    #   .then(next_fn, [inputs], [outputs])  ← 前一步完成后触发
    # ══════════════════════════════════════════════════════════════

    login_btn.click(
        handle_login,  # 核心处理函数：调用后端登录 API，验证凭据
        [login_phone, login_password, login_state],  # 输入：手机号、密码、当前登录状态
        [
            login_state,      # 更新登录状态（写入 token/用户信息）
            auth_block,       # 隐藏认证区域（登录成功后）
            chat_block,       # 显示聊天区域（登录成功后）
            auth_status,      # 显示登录结果消息
            user_info,        # 更新顶栏用户信息
            chatbot,          # 清空聊天历史
            session_list,     # 更新会话列表
            session_dropdown, # 更新会话下拉菜单
        ],
    ).then(
        # 登录成功后自动选中第一个会话，加载其历史记录
        lambda lst, val, state: find_thread_id(lst, val, state),
        [session_list, session_dropdown, login_state],
        [current_thread_id, chatbot],
    )

    register_btn.click(
        handle_register,  # 核心处理函数：调用后端注册 API，验证并创建账号
        [register_name, register_phone, register_password, register_confirm, login_state],
        [
            login_state, auth_block, chat_block, auth_status,
            user_info, chatbot, session_list, session_dropdown,
        ],
    ).then(
        # 注册成功后同样自动选中首个会话
        lambda lst, val, state: find_thread_id(lst, val, state),
        [session_list, session_dropdown, login_state],
        [current_thread_id, chatbot],
    )

    logout_btn.click(
        handle_logout,  # 清空登录状态，切换回认证界面
        [login_state, chatbot],
        [
            login_state, auth_block, chat_block, chatbot,
            user_info, session_list, session_dropdown, current_thread_id,
        ],
    )

    # ══════════════════════════════════════════════════════════════
    # 事件绑定 — 会话管理事件（新建 / 切换）
    # ══════════════════════════════════════════════════════════════

    new_session_btn.click(
        create_new_session,  # 创建新的对话会话（生成 thread_id 并保存到数据库）
        [login_state, session_list],
        [session_list, session_dropdown],
    ).then(
        # 创建后自动选中新会话
        lambda lst, val, state: find_thread_id(lst, val, state),
        [session_list, session_dropdown, login_state],
        [current_thread_id, chatbot],
    )

    session_dropdown.change(
        switch_session,  # 切换会话：从检查点加载历史消息到 chatbot
        [session_dropdown, session_list, login_state],
        [chatbot, session_list],
    ).then(
        # 更新当前 thread_id
        lambda lst, val, state: find_thread_id(lst, val, state),
        [session_list, session_dropdown, login_state],
        [current_thread_id, chatbot],
    )

    # ══════════════════════════════════════════════════════════════
    # 事件绑定 — 文本对话事件
    # ══════════════════════════════════════════════════════════════

    send_btn.click(
        wrap_handle_text,  # 文字发送包装器（含登录校验 + 流式响应处理）
        [msg, file, chatbot, current_thread_id, login_state],
        [chatbot, msg, tts_audio],
    )
    msg.submit(
        wrap_handle_text,  # 按回车同样触发发送
        [msg, file, chatbot, current_thread_id, login_state],
        [chatbot, msg, tts_audio],
    )
    clear_btn.click(
        # 清空聊天历史、输入框和语音回复
        lambda: ([], gr.update(value=""), gr.update(visible=False)),
        None,
        [chatbot, msg, tts_audio],
    )

    # ══════════════════════════════════════════════════════════════
    # 事件绑定 — 语音对话事件
    # ══════════════════════════════════════════════════════════════

    voice_btn.click(
        wrap_handle_voice,  # 语音发送包装器（上传录音 → ASR → Agent → TTS 流式处理）
        [audio_in, chatbot, current_thread_id, login_state],
        [chatbot, audio_in, tts_audio],
    )


# ══════════════════════════════════════════════════════════════
# 非热重载模式下的代理清理与端口检查
#
# Gradio 热重载模式下（gr.NO_RELOAD=False），每次文件变更都会重启进程，
# 以下清理逻辑只在生产/普通模式执行，避免干扰热重载。
# ══════════════════════════════════════════════════════════════

def _prepare_frontend_runtime() -> None:
    """普通启动模式下准备前端运行环境。"""
    from core.utils.port_utils import ensure_port_available

    # 确保前端端口（默认 7860）可用，如被占用则自动 kill 占用进程
    ensure_port_available(Config.FRONTEND_HOST, Config.FRONTEND_PORT)

    # 保存并清除系统 HTTP 代理环境变量，避免 Gradio 请求走代理导致连接后端失败
    _saved_proxies = {}
    for _key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
        _val = os.environ.pop(_key, None)
        if _val is not None:
            _saved_proxies[_key] = _val
    # 设置 no_proxy 确保本地请求不走代理
    os.environ["no_proxy"] = "127.0.0.1,localhost,::1"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"

    cleaned = False  # 防止重复清理的标记

    def _cleanup():
        """进程退出时的清理函数：关闭前端、终止后端子进程、恢复代理设置。"""
        nonlocal cleaned
        if cleaned:
            return  # 已执行过清理，避免重复
        cleaned = True
        try:
            demo.close()  # 关闭 Gradio 界面
        except Exception:
            pass
        if _backend_process is not None and _backend_process.poll() is None:
            try:
                _backend_process.terminate()  # 终止子进程（uvicorn 后端）
            except Exception:
                pass
        # 恢复原始代理环境变量
        for _k, _v in _saved_proxies.items():
            os.environ[_k] = _v

    atexit.register(_cleanup)  # 注册退出回调，确保进程结束时执行清理


# ══════════════════════════════════════════════════════════════
# 启动入口（main guard）
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if gr.NO_RELOAD:
        _prepare_frontend_runtime()
    _ensure_backend_running()  # 自动拉起后端服务（如未运行）
    demo.queue()               # 启用请求队列，支持并发流式响应
    demo.launch(
        theme="soft",                        # Gradio 内置柔和主题
        server_port=Config.FRONTEND_PORT,    # 前端端口（默认 7860）
        server_name=Config.FRONTEND_HOST,    # 绑定地址（默认 127.0.0.1）
        show_error=True,                     # 界面显示错误详情
        css=custom_css,                      # 自定义页面样式
    )
