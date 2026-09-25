"""本地文件与 RAG 服务：state（SQLite 状态）/storage（下载解析）/vectors（embedding 与向量库）
/service（HTTP 与后台执行器）/client（CLI 侧发现与调用）。

数据固定在项目 rag_data/ 下；Chroma 只由 service 进程读写，其他进程一律走 client 的 HTTP 接口。
"""
