# Gagent

Gagent 是一个运行在 Windows 终端里的本地 Agent。它提供连续对话、多会话切换、网页搜索、
PDF/Excel 读取、Python 计算，跨会话使用的文件 RAG 长期记忆，以及跨会话回忆（找回以前的讨论）。

## 技术栈

- **Python 3.11**：核心运行环境、CLI 和后台任务。
- **LangGraph + LangChain**：Agent 状态流转、工具调用循环和消息管理。
- **OpenAI-compatible API**：通过 `ChatOpenAI` 接入兼容 OpenAI 协议的模型服务。
- **FastAPI + Uvicorn + HTTPX**：独立文件服务、后台入库任务和 CLI/服务通信。
- **ChromaDB**：持久化保存 PDF 片段、Excel 摘要向量（`file_chunks`）和对话检索片段（`conversation_chunks`）。
- **Sentence Transformers + multilingual-e5-small**：中英文文本 embedding 和语义检索。
- **SQLite**：保存文件版本、入库任务、重试状态和事件记录，以及对话版本、对话索引任务和摘要缓存。
- **pypdf + pandas/openpyxl**：PDF 文本提取、Excel 预览和数据处理。
- **Beautiful Soup + DDGS**：网页解析和网络搜索。
- **unittest**：Agent、CLI 与 RAG 服务的自动化测试。

## 使用

首次启动会询问 API Key、Model 和 Base URL，并保存到 `api.py`（该文件不会提交到 Git）。
如果已经有 `api.py`，首次配置会跳过且不会覆盖它；使用 `/api` 切换后，启动时优先使用新选择。`api.py` 格式为：

```python
API_KEY = "你的 API Key"
BASE_URL = "模型接口地址"
MODEL = "模型名称"
```

在终端输入（首次运行会自动在项目目录创建 `.venv` 并安装 `requirements.txt`）：

```powershell
Gagent
```

进入 CLI 后输入 `/api` 可以重新录入并切换模型接口；新选择保存在本地的 `api_active.json`，下次启动继续使用。

进入 CLI 后可以直接提问，也可以使用以下命令：

```text
/help             查看帮助
/api              切换模型接口
/new [标题]       新建会话
/sessions         查看所有会话
/switch <id>      切换会话
/memory           查看文件长期记忆与对话索引进度
/retry <job_id>   重试失败的文件或对话入库任务
/exit             退出
```

读取 PDF 或 Excel 后，文件会在后台加入长期记忆。入库完成后，即使切换到新的会话，
Gagent 也能检索之前读取过的资料。首次使用 RAG 时会自动下载
`intfloat/multilingual-e5-small` embedding 模型，约 470 MB；模型和向量数据库都只保存在本地，
不会提交到 Git。

## 跨会话回忆

每轮问答保存成功后，这一轮会由本地服务在后台切块并建立对话向量索引（不等待 embedding，
也不影响本轮回答）。因此可以直接问：

```text
❯ 之前那个对话里，我们最后怎么决定 Excel 的入库方式？

● 正在查找历史对话……
● 正在整理相关会话……

之前确定的是：Excel 使用模型根据网页介绍提供的 summary 入库，
PDF 则直接提取正文。这个决定来自会话「RAG 设计」[abc123]。
```

工作方式：

- 索引以**问答轮次**为单位，片段里保留本轮 `user` 与 `final_answer` 的关联，并附带会话标题
  和少量上一轮原文，用来理解"这个""方案 A"之类的指代；上下文只来自真实历史，不做改写。
- 检索先取相关片段，按会话聚合（用该会话最佳片段的相似度排序，最多 3 个候选），再判断
  是找到一个明确会话、多个候选接近（返回候选让用户确认），还是没找到。
- 会话摘要按需生成并缓存，与会话版本绑定；会话内容变化后重新生成，生成期间会话又更新时
  会标明摘要覆盖的是哪个版本。摘要用当前 `/api` 选定的模型接口，凭据不进入向量和缓存。
- 回忆结果同时给出摘要、命中的原文轮次和来源会话；拿到会话 id 后可以用 `/switch <id>` 打开原会话。
- 索引失败会在 `/memory` 里显示原因，可用 `/retry <job_id>` 重试；重复通知和重复启动同步
  不会重复入库。

对话记忆只在模型主动调用回忆工具时使用，不加入每轮默认检索；文件 RAG 的自动检索方式不变。

## 调试与隔离

以下环境变量用于排查问题或跑隔离实例（服务进程会继承它们，CLI 拉起服务时会带上会话目录）：

```text
GAGENT_RAG_DATA        数据根目录，默认项目下的 rag_data/
GAGENT_SESSIONS_DIR    会话目录，默认项目下的 sessions/
GAGENT_RAG_DISABLE=1   完全不接触本地服务（单测/离线排错用）
GAGENT_CONVERSATION_SYNC=0   关闭后台对话索引同步（检索与回忆仍在服务侧可用）
```

服务日志在 `rag_data/logs/service.log`；`python -m rag.service --status` 可以查看当前服务实例。
