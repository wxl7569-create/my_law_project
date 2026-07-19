"""
端口工具函数 — 自动检测并释放端口

【功能】
查找并结束占用指定端口的进程（仅限 Windows），在服务启动前自动释放端口。
避免因上一次异常退出导致端口残留而无法重新启动。

【用法】
    from core.utils.port_utils import ensure_port_available
    if not ensure_port_available("127.0.0.1", 7860):
        print("端口无法释放，请手动处理")
"""

# ══════════════════════════════════════════════════════════════
# 导入标准库
# ══════════════════════════════════════════════════════════════
import socket
import subprocess
import time
from typing import Optional


# ══════════════════════════════════════════════════════════════
# 辅助函数：查找占用端口的进程
# ══════════════════════════════════════════════════════════════

def find_process_on_port(port: int) -> Optional[str]:
    """
    查找占用指定端口的进程 PID（仅限 Windows）

    Args:
        port: 端口号

    Returns:
        PID 字符串，如果未找到则返回 None
    """
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.split("\n"):
            if f":{port}" in line and (
                "LISTENING" in line or "ESTABLISHED" in line or "TIME_WAIT" in line
            ):
                parts = line.strip().split()
                if parts:
                    return parts[-1]
        return None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
# 辅助函数：结束进程
# ══════════════════════════════════════════════════════════════

def kill_process(pid: str) -> bool:
    """
    结束指定 PID 的进程

    Args:
        pid: 进程 ID

    Returns:
        True 表示成功，False 表示失败
    """
    try:
        subprocess.run(
            ["taskkill", "/F", "/PID", pid],
            capture_output=True,
            timeout=5,
        )
        time.sleep(1)  # 等待进程完全退出
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════
# 核心函数：查找并结束占用端口的进程
# ══════════════════════════════════════════════════════════════

def find_and_kill_process_on_port(port: int) -> bool:
    """
    查找并结束占用指定端口的进程（仅限 Windows）

    流程:
    1. 用 netstat 查找占用端口的 PID
    2. 用 taskkill /F 强行结束进程
    3. 等待 1 秒确保端口释放

    Args:
        port: 端口号

    Returns:
        True 表示成功释放端口，False 表示释放失败或未找到占用进程
    """
    pid = find_process_on_port(port)
    if pid is None:
        print(f"  → 未找到占用端口 {port} 的进程")
        return False

    print(f"  → 找到占用端口 {port} 的进程 (PID={pid})，正在结束...")
    success = kill_process(pid)
    if success:
        print(f"  → 进程已结束")
    else:
        print(f"  → 进程结束失败")
    return success


# ══════════════════════════════════════════════════════════════
# 对外接口：确保端口可用
# ══════════════════════════════════════════════════════════════

def ensure_port_available(host: str, port: int) -> bool:
    """
    确保端口可用 — 如果被占则自动释放

    流程:
    1. 尝试连接端口，检查是否被占用
    2. 如果被占用，自动查找并结束占用进程
    3. 等待 1 秒后再次检查
    4. 如果仍然不可用，返回 False

    Args:
        host: 监听地址（如 "0.0.0.0" 或 "127.0.0.1"）
        port: 端口号

    Returns:
        True 表示端口可用，False 表示无法释放
    """
    # 第1步：检查端口是否可用
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        target_host = host.replace("0.0.0.0", "127.0.0.1")
        result = s.connect_ex((target_host, port))
        s.close()
        if result != 0:
            return True  # 端口空闲
    except Exception:
        return True  # 检查出错时让应用自己处理

    # 第2步：端口被占，自动释放
    print(f"\n⚠️  端口 {port} 已被占用，正在尝试自动释放...")
    success = find_and_kill_process_on_port(port)

    if not success:
        print(f"  ❌ 无法自动释放端口 {port}")
        return False

    # 第3步：等待并重新检查
    time.sleep(1)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        target_host = host.replace("0.0.0.0", "127.0.0.1")
        result = s.connect_ex((target_host, port))
        s.close()
        if result != 0:
            print(f"  ✅ 端口 {port} 已释放\n")
            return True
        else:
            print(f"  ❌ 端口 {port} 释放后仍然被占用\n")
            return False
    except Exception:
        return True
