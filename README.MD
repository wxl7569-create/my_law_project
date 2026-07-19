# PBL 法律咨询系统

本项目是一个法律问答系统，核心能力包括民法典本地检索、网络搜索补充、文件解析、多会话记忆、用户认证、网页前端、统一后端接口和本地语音交互。

## 项目架构

项目采用前后端分离结构：

- `app.py`：网页前端入口，负责登录状态、会话选择、文字输入、文件上传和语音输入。
- `core/api/server.py`：统一后端服务，负责认证、文本问答和语音问答接口。
- `core/agent_engine.py`：智能体工厂，负责创建主问答智能体。
- `core/chat_session.py`：会话生命周期、流式响应和检查点管理。
- `core/voice/`：语音识别、语音合成、文本清洗和语音对话编排。
- `login_app.py`：旧认证服务兼容入口。
- `agent.py`：命令行入口，可直接调用核心问答逻辑。

前端默认请求 `http://127.0.0.1:8001`，可通过 `BACKEND_API_URL` 修改。

## 快速开始

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 配置 `.env`：

```ini
DEEPSEEK_API_KEY=sk-xxx
DASHSCOPE_API_KEY=sk-xxx
TAVILY_API_KEY=tvly-xxx
```

3. 确认本地模型目录：

```text
models/SenseVoiceSmall
models/snakers4_silero-vad
```

4. 构建民法典向量库：

```bash
python scripts/build_chroma.py
```

5. 启动后端：

```bash
python -m core.api.server
```

6. 启动前端：

```bash
python app.py
```

访问 `http://127.0.0.1:7860`。

## 语音配置

默认使用本地 `SenseVoiceSmall` 做语音识别，使用 `pyttsx3` 做离线语音合成。也可以切换到 `edge-tts`：

```ini
VOICE_TTS_PROVIDER=pyttsx3
# 或
VOICE_TTS_PROVIDER=edge
```

可选配置：

```ini
VOICE_ASR_MODEL_DIR=models/SenseVoiceSmall
VOICE_VAD_MODEL_DIR=models/snakers4_silero-vad
VOICE_ENABLE_VAD=0
VOICE_TTS_RATE=180
VOICE_TTS_VOLUME=1.0
VOICE_TTS_EDGE_VOICE=zh-CN-XiaoxiaoNeural
VOICE_TTS_EDGE_RATE=+0%
VOICE_TTS_EDGE_VOLUME=+0%
TEXT_CHAT_ENABLE_TTS=0
VOICE_TTS_FULL_FINAL=1
```

语音提问始终触发语音回复。文字提问默认只返回文字；开发测试时可设置 `TEXT_CHAT_ENABLE_TTS=1`，让文字提问也触发语音回复。

`VOICE_TTS_FULL_FINAL=1` 表示语音提问时先快速播放首批短句，完整回答结束后再生成一份完整语音，避免长回答被多段音频覆盖后听起来过短。开发阶段如果只想测试首段响应速度，可设为 `0`。

语音回复会先清洗智能体文本，移除表情、标记符号、链接、代码块和复杂编号，再按短句送入语音合成队列，避免朗读与逻辑无关的符号内容。

## 日志配置

默认文件日志级别为 `INFO`，可以通过 `.env` 调整：

```ini
LOG_LEVEL=INFO
```

开发排查细节时可设为 `DEBUG`；演示或长期运行时可设为 `WARNING`。语音合成成功、检索命中数、工具返回长度等高频细节会降到 `DEBUG`。

## 检查点持久化

会话记忆由 LangGraph checkpoint 写入 `data/memory.db`。当前策略不是按固定时间间隔刷入，而是：

- `BatchedAsyncSqliteSaver` 保留 `batch_size=5` 的批量写入阈值，减少 SQLite 高频 I/O。
- 每轮文本问答结束后，通过后台异步任务调用 `session.flush()`。
- 每轮语音问答结束、异常或流连接关闭时，也会通过后台异步任务调用 `session.flush()`。
- 服务关闭阶段会执行最终 `flush_all_sessions()`，确保仍在缓冲区中的检查点落盘。

## 接口说明

- `POST /api/v1/auth/register`：注册并登录。
- `POST /api/v1/auth/login`：登录。
- `GET /api/v1/auth/me`：校验令牌并返回用户信息。
- `POST /api/v1/chat`：非流式文本问答。
- `POST /api/v1/chat/stream`：流式文本问答。
- `POST /api/v1/voice/asr`：语音识别。
- `POST /api/v1/voice/tts`：文本转语音。
- `POST /api/v1/voice/chat/stream`：语音输入、文本流式回答和伪流式语音回复。
- `GET /api/v1/health`：智能体健康检查。

## 主流程

```text
用户浏览器
  -> 网页前端
  -> 后端文本接口或语音接口
  -> 会话管理
  -> 主问答智能体
  -> 本地检索、网络搜索、文件解析工具
  -> 检查点数据库
  -> 每轮结束后后台刷入检查点
  -> 后端流式响应
  -> 前端展示
```

## 语音流程

```text
前端麦克风录音
  -> 后端语音对话接口
  -> 本地语音识别
  -> 复用现有智能体流式回答
  -> 原始文本流返回前端
  -> 朗读文本清洗
  -> 短句切分
  -> 后台语音合成
  -> 首段音频快速返回
  -> 完整回答结束后返回完整语音
  -> 后台刷入检查点
```

## 功能

- 用户注册、登录、退出。
- 多用户、多会话隔离。
- 法律文字问答。
- 语音输入和语音回复。
- 文件上传分析，支持 `.txt`、`.docx`、`.doc`、`.md`。
- 民法典本地检索。
- 网络搜索补充。
- 检查点记忆。
- 命令行、后端接口和网页前端多入口。

## 求职演示要点

本项目定位为“民法典资料检索与咨询辅助系统”，用于展示 RAG、Agent、FastAPI、Gradio 和本地向量库的完整应用链路。系统不替代律师给出正式法律意见。

建议演示问题：

- 民法典中的基本原则规定有哪些？
- 什么是自然人？请结合民法典说明。
- 民法典中规定什么叫宣告失踪？
- 民法典中关于监护的规定有哪些？
- 租赁合同纠纷中，承租人逾期支付租金通常涉及哪些规定？

回答展示逻辑：

- Agent 优先调用 `rag_law_civil` 检索民法典本地知识库。
- 回答结束后追加“参考依据（RAG 检索片段）”折叠区。
- 参考依据来自 Chroma 检索命中的原文片段和 `Document.metadata.source`。
- 当未命中、检索失败或问题为空时，不展示参考依据，避免制造伪引用。

当前边界：

- 回答仅作法律资料检索和解释辅助，不构成法律意见。
- 向量库来源范围有限，不能保证覆盖所有法律法规、司法解释和最新案例。
- 系统未接入律师审核流程，也未实现正式法律风险评估。
- 演示环境未按生产级并发、权限和审计要求设计。

## 运行产物

以下文件属于运行产物或本地敏感数据，应保持在 `.gitignore` 中：

- `.env`
- `data/*.db`
- `data/voice_cache/`
- `logfile/*.log`
- `chroma_law_civil/`
- `__pycache__/`
