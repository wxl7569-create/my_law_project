"""前端会话列表辅助函数。"""

# ══════════════════════════════════════════════════════════════
# 导入标准库
# ══════════════════════════════════════════════════════════════
from __future__ import annotations

# ══════════════════════════════════════════════════════════════
# 导入第三方库
# ══════════════════════════════════════════════════════════════
import gradio as gr

# ══════════════════════════════════════════════════════════════
# 导入内部模块
# ══════════════════════════════════════════════════════════════
from core.utils.session_manager import SessionManager


# ══════════════════════════════════════════════════════════════
# 会话列表加载
# ══════════════════════════════════════════════════════════════

def load_sessions(login_state):
    user_id = (login_state or {}).get("user_id", "")
    if not user_id:
        return [], gr.update(choices=[], value=None)

    manager = SessionManager()
    sessions = manager.list_sessions(user_id)
    if not sessions:
        sessions = [manager.ensure_default_session(user_id)]

    choices = [s["name"] for s in sessions]
    return sessions, gr.update(choices=choices, value=choices[0] if choices else None)


# ══════════════════════════════════════════════════════════════
# 会话切换
# ══════════════════════════════════════════════════════════════

def switch_session(selected_name, session_list, login_state):
    thread_id, history = find_thread_id(session_list, selected_name, login_state)
    _ = thread_id
    return history, session_list


# ══════════════════════════════════════════════════════════════
# 创建新会话
# ══════════════════════════════════════════════════════════════

def create_new_session(login_state, session_list):
    user_id = (login_state or {}).get("user_id", "")
    if not user_id:
        return session_list or [], gr.update()

    current = session_list or []
    manager = SessionManager()
    existing = {s["name"] for s in current}
    index = len(current) + 1
    name = f"对话{index}"
    while name in existing:
        index += 1
        name = f"对话{index}"

    new_session = {"id": manager.build_thread_id(user_id, name), "name": name, "updated": ""}
    manager.save_session(user_id, new_session["id"], new_session["name"])
    updated = [new_session] + current
    return updated, gr.update(choices=[s["name"] for s in updated], value=name)


# ══════════════════════════════════════════════════════════════
# 辅助函数：查找会话线程 ID
# ══════════════════════════════════════════════════════════════

def find_thread_id(session_list, selected_name, login_state=None):
    if not selected_name or not session_list:
        return "", []

    user_id = (login_state or {}).get("user_id", "")
    for session in session_list:
        if session["name"] == selected_name:
            thread_id = session["id"]
            history = SessionManager().load_history(thread_id, user_id) if user_id else []
            return thread_id, history
    return "", []
