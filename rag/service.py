"""Gagent 的文件与 RAG 服务：本机 HTTP 接口 + 一个后台入库执行器。

只在独立进程里运行（CLI 通过 file_service_client 访问），只监听 127.0.0.1。
文件、向量、任务和事件都归本进程管：CLI 退出不影响正在处理的任务。

耗时的下载、解析和 embedding 全部放到工作线程，不占 HTTP 事件循环，
否则后台建库期间所有接口会一起卡住。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag import state as rag_state
from rag import storage as file_storage
from rag import vectors as rag_store

PROTOCOL = rag_state.PROTOCOL
DEFAULT_PORT = 8732
MAX_POLL_SECONDS = 30.0


def service_id() -> str:
    return rag_state.service_id()


# ---------------------------------------------------------------- 日志

_logger = logging.getLogger("gagent.file_service")


def setup_logging() -> Path:
    rag_state.ensure_dirs()
    path = rag_state.logs_dir() / "service.log"
    _rotate_if_large(path)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "chromadb", "httpx"):
        logging.getLogger(name).handlers = [handler]
        logging.getLogger(name).propagate = False
    return path


def _rotate_if_large(path: Path, limit: int = 5 * 1024 * 1024) -> None:
    try:
        if path.exists() and path.stat().st_size > limit:
            os.replace(path, path.with_suffix(".log.1"))
    except OSError:
        pass


# ---------------------------------------------------------------- 后台执行器

class IngestionWorker(threading.Thread):
    """单个顺序执行器：一次只做一件事，天然避免同一版本被并行重建。"""

    def __init__(self) -> None:
        super().__init__(name="rag-ingestion", daemon=True)
        # 不能叫 _stop：Thread 自己有 _stop() 方法，被 Event 顶掉后 join() 会炸
        self._stopping = threading.Event()
        self._wake = threading.Event()

    def wake(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._stopping.set()
        self._wake.set()

    def run(self) -> None:
        while not self._stopping.is_set():
            try:
                job = self._claim()
            except Exception as exc:  # 取任务本身失败（库被占用等）不能带走线程
                _logger.exception("claim failed: %s", exc)
                job = None
            if job is None:
                self._wait()
                continue
            self._process(job)

    def _claim(self) -> dict | None:
        conn = rag_state.connect()
        try:
            return rag_state.claim_job(conn, time.time())
        finally:
            conn.close()

    def _wait(self) -> None:
        conn = rag_state.connect()
        try:
            due = rag_state.next_due_at(conn)
        finally:
            conn.close()
        timeout = MAX_POLL_SECONDS if due is None else max(0.05, min(MAX_POLL_SECONDS, due - time.time()))
        self._wake.wait(timeout)
        self._wake.clear()

    def _process(self, job: dict) -> None:
        conn = rag_state.connect()
        started = time.monotonic()
        try:
            version = rag_state.get_version(conn, job["version_id"])
            if version is None:
                raise rag_state.PermanentError("文件版本记录已不存在")
            if not Path(version["local_path"]).is_file():
                raise rag_state.PermanentError(f"本地文件缺失：{version['local_path']}")
            rag_state.set_job_stage(conn, job["job_id"], "embed")
            count = rag_store.index_version(version)
            rag_state.set_job_stage(conn, job["job_id"], "activate")
            rag_store.set_version_active(version["doc_id"], version["sha256"], True)
            result = rag_state.finish_job(conn, job["job_id"], ok=True)
            superseded = result.get("superseded")
            if superseded:
                old = rag_state.get_version(conn, superseded)
                if old is not None:
                    rag_store.delete_version_vectors(old["doc_id"], old["sha256"])
            _logger.info("indexed job=%s kind=%s file=%s vectors=%d switched=%s %.1fs",
                         job["job_id"], version["kind"], version["file_name"], count,
                         result.get("switched"), time.monotonic() - started)
        except rag_state.PermanentError as exc:
            rag_state.finish_job(conn, job["job_id"], ok=False, error=str(exc), permanent=True)
            _logger.warning("job %s 不可重试：%s", job["job_id"], exc)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            result = rag_state.finish_job(conn, job["job_id"], ok=False, error=detail)
            _logger.warning("job %s 失败（第 %s 次，%s）：%s", job["job_id"], job["attempts"],
                            result.get("status"), detail)
        finally:
            conn.close()


# ---------------------------------------------------------------- 请求模型

class PdfRead(BaseModel):
    url: str = Field(min_length=1)
    max_pages: int = 5
    start_page: int = 1
    start_char: int = 0
    max_chars: int = 20000
    refresh: bool = False


class ExcelRead(BaseModel):
    url: str = Field(min_length=1)
    summary: str
    sheet_name: str | None = None
    max_rows: int = 100
    max_columns: int = 30
    start_row: int = 0
    start_column: int = 0
    preview: bool = True
    preview_rows: int = 5
    refresh: bool = False


class PdfFind(BaseModel):
    url: str = Field(min_length=1)
    query: str = Field(min_length=1)
    start_page: int = 1
    max_matches: int = 5
    context_chars: int = 600
    refresh: bool = False


class Search(BaseModel):
    query: str = Field(min_length=1)
    pdf_candidates: int = rag_store.PDF_CANDIDATES
    excel_candidates: int = rag_store.EXCEL_CANDIDATES
    pdf_min_score: float = rag_store.PDF_MIN_SCORE
    excel_min_score: float = rag_store.EXCEL_MIN_SCORE
    max_results: int = rag_store.MAX_RESULTS
    max_chars: int = rag_store.MAX_CONTEXT_CHARS


worker: IngestionWorker | None = None
started_at = ""


def _error(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": kind, "message": message})


def _tool_error(exc: Exception) -> JSONResponse:
    """把异常翻成一句对模型可执行的话，不回显整段参数。"""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        hint = {401: "认证失败", 403: "拒绝访问", 404: "资源不存在，不要重复请求或拼接附件路径",
                429: "服务限流"}.get(status, "远程服务请求失败")
        return _error(502, "remote_error", f"远端文件获取失败：HTTP {status} {hint}。")
    if isinstance(exc, httpx.RequestError):
        return _error(502, "network_error", "网络连接失败或超时；检查网络/代理，勿连续重复相同请求。")
    if isinstance(exc, ValueError):
        return _error(400, "bad_request", str(exc))
    if isinstance(exc, KeyError):
        return _error(404, "not_found", str(exc))
    return _error(500, "server_error", f"{type(exc).__name__}: {exc}")


async def _run(function, *args, **kwargs):
    try:
        return await run_in_threadpool(function, *args, **kwargs)
    except Exception as exc:
        if isinstance(exc, (httpx.RequestError, httpx.HTTPStatusError)):
            _logger.warning("remote fetch failed: %s", exc)
        else:
            _logger.warning("request failed: %s: %s", type(exc).__name__, exc)
        return _tool_error(exc)


# ---------------------------------------------------------------- 应用

@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker, started_at
    setup_logging()
    rag_state.ensure_dirs()
    conn = rag_state.connect()
    try:
        recovered = rag_state.recover_interrupted(conn, time.time())
        pending = conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE status IN ('queued','retry_wait')"
                               " ").fetchone()["n"]
    finally:
        conn.close()
    fixed = 0
    try:
        orphans = rag_store.drop_orphans()
        if orphans:
            _logger.warning("清理了 %d 个没有版本记录的残留向量", orphans)
    except Exception as exc:
        _logger.warning("残留向量清理失败：%s", exc)
    try:
        fixed = rag_store.reconcile_active()
    except Exception as exc:
        _logger.warning("向量 active 校对失败：%s", exc)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    worker = IngestionWorker()
    worker.wake()
    worker.start()
    threading.Thread(target=_warm_model, name="rag-warm", daemon=True).start()
    _logger.info("服务启动 pid=%s 恢复中断任务=%d 待处理=%d active 修正=%d",
                 os.getpid(), recovered, pending, fixed)
    yield
    if worker is not None:
        worker.stop()
        worker.join(timeout=5)
    _logger.info("服务退出 pid=%s", os.getpid())


def _warm_model() -> None:
    """启动后先把 embedding 模型读进内存，别让第一次提问等 10 秒加载。"""
    started = time.monotonic()
    try:
        rag_store.get_model()
        _logger.info("embedding 模型就绪，用时 %.1fs", time.monotonic() - started)
    except Exception as exc:
        _logger.error("embedding 模型不可用：%s: %s", type(exc).__name__, exc)


app = FastAPI(title="Gagent file service", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    conn = rag_state.connect()
    try:
        pending = conn.execute("SELECT COUNT(*) AS n FROM jobs"
                               " WHERE status IN ('queued','processing','retry_wait')").fetchone()["n"]
        failed = conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE status='failed'").fetchone()["n"]
        documents = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
    finally:
        conn.close()
    try:
        vectors = rag_store.vector_count()
    except Exception as exc:
        vectors = -1
        _logger.warning("向量库不可读：%s", exc)
    return {"ok": True, "protocol": PROTOCOL, "service_id": service_id(), "pid": os.getpid(),
            "started_at": started_at, "documents": documents, "vectors": vectors,
            "jobs_pending": pending, "jobs_failed": failed,
            "model_ready": rag_store.model_ready(), "data_root": str(rag_state.DATA_ROOT)}


@app.post("/v1/read/pdf")
async def read_pdf(request: PdfRead):
    result = await _run(file_storage.read_pdf, request.url, max_pages=request.max_pages,
                        start_page=request.start_page, start_char=request.start_char,
                        max_chars=request.max_chars, refresh=request.refresh)
    _wakeup(result)
    return result


@app.post("/v1/read/excel")
async def read_excel(request: ExcelRead):
    if not request.summary.strip():
        return _error(400, "bad_request", "summary 不能为空白：请描述整个工作簿的主题和用途，供后续检索定位文件")
    result = await _run(file_storage.read_excel, request.url, request.summary,
                        sheet_name=request.sheet_name, max_rows=request.max_rows,
                        max_columns=request.max_columns, start_row=request.start_row,
                        start_column=request.start_column, preview=request.preview,
                        preview_rows=request.preview_rows, refresh=request.refresh)
    _wakeup(result)
    return result


@app.post("/v1/find/pdf")
async def find_in_pdf(request: PdfFind):
    result = await _run(file_storage.find_in_pdf, request.url, request.query,
                        start_page=request.start_page, max_matches=request.max_matches,
                        context_chars=request.context_chars, refresh=request.refresh)
    _wakeup(result)
    return result


def _wakeup(result) -> None:
    """刚提交任务就叫醒执行器，不让排队等到下一轮轮询。"""
    if isinstance(result, dict) and worker is not None:
        worker.wake()


@app.post("/v1/search")
async def search(request: Search):
    return await run_in_threadpool(search_results, request)


def search_results(request: Search) -> dict | JSONResponse:
    """检索只接收本轮用户输入原文；结果只来自已激活的有效版本。"""
    query = (request.query or "").strip()
    if not query:
        return _error(400, "bad_request", "query 不能为空")
    try:
        rows = rag_store.search(query, pdf_candidates=request.pdf_candidates,
                                excel_candidates=request.excel_candidates,
                                pdf_min_score=request.pdf_min_score,
                                excel_min_score=request.excel_min_score,
                                max_results=request.max_results, max_chars=request.max_chars)
    except Exception as exc:
        _logger.warning("检索失败：%s: %s", type(exc).__name__, exc)
        return _error(500, "search_error", f"{type(exc).__name__}: {exc}")
    segments = rag_store.query_segments(query)
    results = []
    for index, row in enumerate(rows, start=1):
        entry = {"source_id": f"R{index}", "kind": row["kind"], "file_name": row["file_name"],
                 "document_id": row["doc_id"], "version": row["version"],
                 "local_path": row["local_path"], "source_url": row["source_url"],
                 "score": round(row["score"], 4), "text": row["text"]}
        if row["kind"] == rag_store.PDF_KIND:
            entry.update(page=row["page"], start_char=row["start_char"], end_char=row["end_char"])
            if row.get("page_count"):
                entry["page_count"] = row["page_count"]
        elif row.get("sheets"):
            entry["sheets"] = row["sheets"]
        if row.get("newer_version_pending"):
            entry.update(newer_version_pending=True,
                         note="该文件存在更新版本尚未完成入库，本条来自当前有效版本")
        results.append(entry)
    # segments 回显实际送进 embedding 的文本（不含前缀），便于核对"只用了本轮输入"
    return {"query": query, "segments": segments, "results": results}


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str):
    return await run_in_threadpool(job_detail, job_id)


def job_detail(job_id: str) -> dict | JSONResponse:
    conn = rag_state.connect()
    try:
        job = rag_state.get_job(conn, job_id)
        if job is None:
            return _error(404, "not_found", f"没有任务 {job_id}")
        version = rag_state.get_version(conn, job["version_id"]) or {}
        return {**job, "file_name": version.get("file_name"), "version": version.get("sha256"),
                "version_status": version.get("status")}
    finally:
        conn.close()


@app.get("/v1/events")
async def list_events(after: int = 0, limit: int = 50):
    return await run_in_threadpool(events_window, after, limit)


def events_window(after: int, limit: int) -> dict:
    limit = max(1, min(int(limit), 200))
    conn = rag_state.connect()
    try:
        rows = rag_state.events_after(conn, max(0, int(after)), limit)
        return {"events": rows, "cursor": (rows[-1]["id"] if rows else max(0, int(after))),
                "latest": rag_state.latest_event_id(conn)}
    finally:
        conn.close()


@app.get("/v1/state")
async def state(limit: int = 20):
    """CLI 的 /memory 用：最近事件 + 未成功任务，含可查看的错误详情。"""
    return await run_in_threadpool(state_snapshot, limit)


def state_snapshot(limit: int) -> dict:
    conn = rag_state.connect()
    try:
        events = rag_state.recent_events(conn, max(1, min(int(limit), 100)))
        jobs = [dict(row) for row in conn.execute(
            "SELECT j.job_id, j.status, j.attempts, j.max_attempts, j.round, j.last_error,"
            " v.file_name, v.kind, v.id AS version_id, v.status AS version_status"
            " FROM jobs j JOIN versions v ON v.id=j.version_id"
            " WHERE j.status IN ('failed','queued','processing','retry_wait')"
            " ORDER BY j.updated_at DESC LIMIT ?", (max(1, min(int(limit), 100)),)).fetchall()]
        return {"events": events, "jobs": jobs, "model_ready": rag_store.model_ready()}
    finally:
        conn.close()


@app.get("/v1/documents")
async def documents(limit: int = 50):
    """每个文件版本的索引情况：状态、任务、向量条数、覆盖页码。/memory 与验收都用它。"""
    return await run_in_threadpool(document_snapshot, limit)


def document_snapshot(limit: int) -> dict:
    limit = max(1, min(int(limit), 200))
    conn = rag_state.connect()
    try:
        docs = [dict(row) for row in conn.execute("SELECT * FROM documents ORDER BY updated_at DESC LIMIT ?",
                                                  (limit,)).fetchall()]
        for doc in docs:
            doc["versions"] = []
            for row in conn.execute("SELECT * FROM versions WHERE doc_id=? ORDER BY id", (doc["doc_id"],)):
                version = dict(row)
                job = conn.execute("SELECT * FROM jobs WHERE version_id=? ORDER BY round DESC LIMIT 1",
                                   (version["id"],)).fetchone()
                index = rag_store.version_index(doc["doc_id"], version["sha256"])
                doc["versions"].append({
                    "version_id": version["id"], "sha256": version["sha256"], "status": version["status"],
                    "active": version["id"] == doc["active_version"], "local_path": version["local_path"],
                    "page_count": version["page_count"], "empty_pages": json.loads(version["empty_pages"] or "[]"),
                    "chunk_count": version["chunk_count"], "max_tokens": version["max_tokens"],
                    "sheet_names": json.loads(version["sheet_names"] or "[]"), "summary": version["summary"],
                    "error": version["error"], "vectors": index["vectors"], "pages": index["pages"],
                    "active_flags": index["active"],
                    "job": None if job is None else {"job_id": job["job_id"], "status": job["status"],
                                                     "attempts": job["attempts"], "round": job["round"],
                                                     "last_error": job["last_error"]}})
        return {"documents": docs}
    finally:
        conn.close()


@app.post("/v1/jobs/{job_id}/retry")
async def retry(job_id: str):
    return await run_in_threadpool(start_retry, job_id)


def start_retry(job_id: str) -> dict | JSONResponse:
    conn = rag_state.connect()
    try:
        job = rag_state.retry_job(conn, job_id)
    except KeyError:
        return _error(404, "not_found", f"没有任务 {job_id}")
    except ValueError as exc:
        return _error(409, "not_retryable", str(exc))
    except Exception as exc:
        _logger.exception("重试提交失败")
        return _error(500, "server_error", f"{type(exc).__name__}: {exc}")
    finally:
        conn.close()
    if worker is not None:
        worker.wake()
    return {"job_id": job["job_id"], "status": job["status"], "round": job["round"],
            "attempts": job["attempts"]}


# ---------------------------------------------------------------- 启动

def pick_port(requested: int | None) -> int:
    """先占一个空闲端口再交给 uvicorn，端口写进 service.json 供 CLI 发现。"""
    for _ in range(20):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", requested or 0))
            return probe.getsockname()[1]
        except OSError:
            requested = None
        finally:
            probe.close()
    raise OSError("找不到可用的本机端口")


def write_service_info(port: int) -> Path:
    rag_state.ensure_dirs()
    path = rag_state.service_info_path()
    payload = {"port": port, "pid": os.getpid(), "protocol": PROTOCOL, "service_id": service_id(),
               "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
               "data_root": str(rag_state.DATA_ROOT)}
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def serve(port: int | None) -> int:
    import uvicorn
    chosen = pick_port(port)
    try:
        write_service_info(chosen)
    except OSError as exc:
        print(f"无法写入服务信息：{exc}", file=sys.stderr)
        return 1
    config = uvicorn.Config(app, host="127.0.0.1", port=chosen, log_level="info", log_config=None,
                            access_log=False, lifespan="on")
    server = uvicorn.Server(config)
    try:
        server.run()
    finally:
        try:
            rag_state.service_info_path().unlink(missing_ok=True)
        except OSError:
            pass
    return 0


def prepare() -> int:
    """安装准备：建目录、建库、下载 embedding 模型。"""
    setup_logging()
    rag_state.ensure_dirs()
    rag_state.connect().close()
    print(f"数据目录：{rag_state.DATA_ROOT}")
    if rag_store.model_ready():
        print(f"embedding 模型已存在：{rag_store.model_dir()}")
    else:
        print(f"正在下载 {rag_store.MODEL_ID} 到 {rag_store.model_dir()} …")
        rag_store.prepare_model()
        print("下载完成")
    print(f"向量集合：{rag_store.COLLECTION_NAME}，当前记录 {rag_store.vector_count()} 条")
    return 0


def status() -> int:
    path = rag_state.service_info_path()
    if not path.exists():
        print("服务未运行（没有 rag_data/service.json）")
        return 1
    info = json.loads(path.read_text(encoding="utf-8"))
    try:
        reply = httpx.get(f"http://127.0.0.1:{info['port']}/health", timeout=3).json()
    except Exception as exc:
        print(f"service.json 存在但探活失败（可能是残留）：{exc}")
        return 1
    print(json.dumps({**info, "health": reply}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="file_service", description="Gagent 文件与 RAG 服务")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--serve", action="store_true", help="启动服务（默认动作）")
    group.add_argument("--prepare", action="store_true", help="建目录并下载 embedding 模型")
    group.add_argument("--status", action="store_true", help="查看当前服务状态")
    parser.add_argument("--port", type=int, default=None, help="监听端口，默认自动选择空闲端口")
    args = parser.parse_args(argv)
    if args.prepare:
        return prepare()
    if args.status:
        return status()
    return serve(args.port)


if __name__ == "__main__":
    raise SystemExit(main())
