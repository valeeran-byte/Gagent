# Gagent

Gagent 是一个运行在 Windows 终端里的本地 Agent。它提供连续对话、多会话切换、网页搜索、
PDF/Excel 读取、Python 计算，以及跨会话使用的文件 RAG 长期记忆。

## 技术栈

- **Python 3.11**：核心运行环境、CLI 和后台任务。
- **LangGraph + LangChain**：Agent 状态流转、工具调用循环和消息管理。
- **OpenAI-compatible API**：通过 `ChatOpenAI` 接入兼容 OpenAI 协议的模型服务。
- **FastAPI + Uvicorn + HTTPX**：独立文件服务、后台入库任务和 CLI/服务通信。
- **ChromaDB**：持久化保存 PDF 片段与 Excel 摘要向量。
- **Sentence Transformers + multilingual-e5-small**：中英文文本 embedding 和语义检索。
- **SQLite**：保存文件版本、入库任务、重试状态和事件记录。
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
/memory           查看文件长期记忆状态
/retry <job_id>   重试失败的文件入库任务
/exit             退出
```

读取 PDF 或 Excel 后，文件会在后台加入长期记忆。入库完成后，即使切换到新的会话，
Gagent 也能检索之前读取过的资料。首次使用 RAG 时会自动下载
`intfloat/multilingual-e5-small` embedding 模型，约 470 MB；模型和向量数据库都只保存在本地，
不会提交到 Git。
