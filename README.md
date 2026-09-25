# Gagent

Gagent 是一个运行在 Windows 终端里的本地 Agent。它提供连续对话、多会话切换、网页搜索、
PDF/Excel 读取、Python 计算，以及跨会话使用的文件 RAG 长期记忆。

## 使用

在 `api.py` 中配置模型接口（该文件不会提交到 Git）：

```python
API_KEY = "你的 API Key"
BASE_URL = "模型接口地址"
MODEL = "模型名称"
```

安装项目依赖后，在终端输入：

```powershell
Gagent
```

进入 CLI 后可以直接提问，也可以使用以下命令：

```text
/help             查看帮助
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
