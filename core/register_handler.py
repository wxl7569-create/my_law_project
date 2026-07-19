"""前端登录、注册和退出事件处理。"""

# ═══════════════════════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import gradio as gr

from core.auth_client import login as api_login
from core.auth_client import register as api_register
from core.utils.session_handlers import load_sessions


# ═══════════════════════════════════════════════════════════════════════════════
# Internal Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _state(result: dict) -> dict:
    return {
        "logged_in": True,
        "token": result["token"],
        "user_id": result["user_id"],
        "name": result["name"],
        "phone": result["phone"],
    }


def _session_updates(user_state: dict):
    sessions, dropdown = load_sessions(user_state)
    return sessions, dropdown


# ═══════════════════════════════════════════════════════════════════════════════
# Event Handlers (Login, Register, Logout)
# ═══════════════════════════════════════════════════════════════════════════════

def handle_login(phone, password, state):
    if not str(phone or "").strip():
        return state, gr.update(), gr.update(), "请输入手机号", gr.update(), gr.update(), gr.update(), gr.update()
    if not password:
        return state, gr.update(), gr.update(), "请输入密码", gr.update(), gr.update(), gr.update(), gr.update()

    result = api_login(str(phone).strip(), password)
    if not result["success"]:
        return (
            state,
            gr.update(),
            gr.update(),
            f"登录失败: {result['error']}",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    user_state = _state(result)
    sessions, dropdown = _session_updates(user_state)
    return (
        user_state,
        gr.update(visible=False),
        gr.update(visible=True),
        "",
        gr.update(value=f"用户: {result['name']} | {result['phone']}"),
        [],
        sessions,
        dropdown,
    )


def handle_register(name, phone, password, confirm, state):
    if not str(name or "").strip():
        return state, gr.update(), gr.update(), "请输入姓名", gr.update(), gr.update(), gr.update(), gr.update()
    if not str(phone or "").strip():
        return state, gr.update(), gr.update(), "请输入手机号", gr.update(), gr.update(), gr.update(), gr.update()
    if not password:
        return state, gr.update(), gr.update(), "请输入密码", gr.update(), gr.update(), gr.update(), gr.update()
    if password != confirm:
        return state, gr.update(), gr.update(), "两次密码输入不一致", gr.update(), gr.update(), gr.update(), gr.update()

    result = api_register(str(name).strip(), str(phone).strip(), password)
    if not result["success"]:
        return (
            state,
            gr.update(),
            gr.update(),
            f"注册失败: {result['error']}",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    user_state = _state(result)
    sessions, dropdown = _session_updates(user_state)
    return (
        user_state,
        gr.update(visible=False),
        gr.update(visible=True),
        "",
        gr.update(value=f"用户: {result['name']} | {result['phone']}"),
        [],
        sessions,
        dropdown,
    )


def handle_logout(state, chatbot_value):
    _ = state, chatbot_value
    return (
        {"logged_in": False, "token": "", "user_id": "", "name": "", "phone": ""},
        gr.update(visible=True),
        gr.update(visible=False),
        [],
        gr.update(value="欢迎用户"),
        [],
        gr.update(choices=[], value=None),
        "",
    )
