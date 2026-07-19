# 项目结构

更新时间：2026-06-22

## 总览

```text
PBL_project/
├── app.py                         # 网页前端入口
├── agent.py                       # 命令行入口
├── login_app.py                   # 旧认证服务兼容入口
├── requirements.txt               # 依赖清单
├── README.MD                      # 项目说明
├── STRUCTURE.md                   # 项目结构说明
├── scripts/
│   └── build_chroma.py            # 构建民法典向量库
├── core/
│   ├── api/
│   │   ├── auth_routes.py         # 认证接口路由
│   │   ├── agent_routes.py        # 文本问答接口路由
│   │   ├── voice_routes.py        # 语音识别、语音合成和语音对话接口路由
│   │   └── server.py              # 统一后端服务入口
│   ├── config/
│   │   └── settings.py            # 统一配置
│   ├── voice/
│   │   ├── __init__.py
│   │   ├── asr.py                 # 本地语音识别
│   │   ├── vad.py                 # 可选静音检测入口
│   │   ├── tts.py                 # 语音合成提供器
│   │   ├── splitter.py            # 伪流式语音合成短句切分
│   │   ├── text_normalizer.py     # 朗读文本清洗
│   │   └── pipeline.py            # 语音识别到智能体再到语音合成的编排
│   ├── utils/
│   │   ├── chat_wrappers.py       # 前端文字聊天包装
│   │   ├── voice_wrappers.py      # 前端语音聊天包装
│   │   ├── llms.py                # 大模型和嵌入模型初始化
│   │   ├── logger.py              # 日志配置
│   │   ├── memory_sqlite.py       # 检查点数据库
│   │   ├── models.py              # 智能体上下文和响应模型
│   │   ├── port_utils.py          # 本地端口辅助
│   │   ├── quick_reply.py         # 简单问题快速回复
│   │   ├── rag_law_civil.py       # 民法典本地检索
│   │   ├── session_handlers.py    # 前端会话界面处理
│   │   ├── session_manager.py     # 会话列表和历史恢复
│   │   ├── tavily_search.py       # 网络搜索
│   │   ├── tools.py               # 工具注册
│   │   └── word_reader.py         # 上传文档解析
│   ├── agent_engine.py            # 主问答智能体工厂
│   ├── app_handlers.py            # 前端到后端的文本请求辅助
│   ├── auth_client.py             # 前端认证请求客户端
│   ├── bootstrap.py               # 运行环境初始化
│   ├── chat_session.py            # 会话生命周期
│   └── register_handler.py        # 前端认证事件处理
├── data/
│   ├── Civil Code.docx            # 民法典源文档
│   ├── memory.db                  # 运行时检查点数据库
│   ├── users.db                   # 运行时用户数据库
│   └── voice_cache/               # 运行时语音缓存
├── models/
│   ├── SenseVoiceSmall/           # 本地语音识别模型
│   └── snakers4_silero-vad/       # 本地静音检测模型
├── chroma_law_civil/              # 运行时向量库
└── logfile/
    └── app.log                    # 运行日志
```

## 分层

```text
前端层
  app.py
  core/register_handler.py
  core/app_handlers.py
  core/auth_client.py
  core/utils/session_handlers.py
  core/utils/chat_wrappers.py
  core/utils/voice_wrappers.py

接口层
  core/api/server.py
  core/api/auth_routes.py
  core/api/agent_routes.py
  core/api/voice_routes.py

智能体层
  core/chat_session.py
  core/agent_engine.py
  core/utils/tools.py

语音层
  core/voice/asr.py
  core/voice/vad.py
  core/voice/tts.py
  core/voice/splitter.py
  core/voice/text_normalizer.py
  core/voice/pipeline.py

工具和数据层
  core/utils/rag_law_civil.py
  core/utils/tavily_search.py
  core/utils/word_reader.py
  core/utils/memory_sqlite.py
  core/utils/session_manager.py
```

## 文本运行流

```text
网页前端
  -> 后端文本流式接口
  -> 会话管理
  -> 主问答智能体
  -> 本地检索、网络搜索、文件解析工具
  -> 检查点数据库
  -> 每轮结束后后台刷入检查点
  -> 后端流式响应
  -> 前端展示
```

## 语音运行流

```text
网页前端麦克风
  -> 后端语音对话接口
  -> 本地语音识别
  -> 会话管理和主问答智能体
  -> 文本流返回前端
  -> 朗读文本清洗和短句切分
  -> 后台语音合成
  -> 音频返回前端自动播放
  -> 对话结束后后台刷入检查点
```

## 说明

- 前端不直接调用智能体、语音识别或语音合成，本地模型能力统一通过后端接口提供。
- 语音识别、语音合成等耗时任务通过后台线程或异步任务执行，避免阻塞主文本流。
- 语音合成使用清洗后的朗读文本，文字界面仍保留智能体原始回答。
- 检查点不再使用固定 30 秒周期任务；文本和语音问答会在每轮结束后异步刷入，服务关闭时执行最终刷入。
- `login_app.py` 保留为旧认证服务兼容入口。
- `.env`、数据库、向量库、日志、语音缓存和字节码文件应排除在版本历史之外。
