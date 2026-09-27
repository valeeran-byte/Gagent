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

import conversation_summary
import session_store
from rag import state as rag_state
from rag import storage as file_storage
from rag import vectors as rag_store

PROTOCOL = rag_state.PROTOCOL
DEFAULT_PORT = 8732
MAX_POLL_SECONDS = 30.0

# 回忆接口的 evidence 上限：轮数与总字符数，避免把整段历史塞回上下文。
RECALL_MAX_EVIDENCE = 5
RECALL_MAX_CHARS = 6000
# 对话检索候选数量的默认值（与 rag/conversations.py 的默认一致）。
CONVERSATION_TOP_CHUNKS = 20
CONVERSATION_MAX_SESSIONS = 3
# 定向回忆（用户点名会话）时多取几个候选，尽量拿到目标会话的命中片段。
RECALL_TARGET_SESSIONS = 8


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
            if job.get("_queue") == "conversation":
                self._process_conversation(job)
            else:
                self._process(job)

    def _claim(self) -> dict | None:
        """取一个到期任务；文件任务优先，空闲时才处理对话索引任务。"""
        conn = rag_state.connect()
        try:
            job = rag_state.claim_job(conn, time.time())
            if job is not None:
                return {**job, "_queue": "file"}
            job = rag_state.claim_conversation_job(conn, time.time())
            if job is not None:
                return {**job, "_queue": "conversation"}
            return None
        finally:
            conn.close()

    def _wait(self) -> None:
        """两个队列里最近一个到期时间决定睡多久，不为对话任务空转轮询。"""
        conn = rag_state.connect()
        try:
            due = [value for value in (rag_state.next_due_at(conn),
                                       rag_state.conversation_next_due_at(conn))
                   if value is not None]
        finally:
            conn.close()
        timeout = MAX_POLL_SECONDS if not due else max(0.05, min(MAX_POLL_SECONDS, min(due) - time.time()))
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

    def _process_conversation(self, job: dict) -> None:
        """一个会话版本的对话索引：切块入库 → 切换有效版本 → 删掉被替换版本的旧向量。

        异常口径与文件任务一致：会话原文缺失、没有可索引内容属于确定性问题，不重试；
        其余异常交给自动重试（5 秒 / 30 秒，共 3 次）。
        """
        conn = rag_state.connect()
        started = time.monotonic()
        try:
            version = rag_state.get_conversation_version(conn, job["version_id"])
            if version is None:
                raise rag_state.PermanentError("会话版本记录已不存在")
            if not int(version.get("turn_count") or 0):
                raise rag_state.PermanentError("该会话没有可索引的问答原文")
            try:
                path = session_store.SessionStore(_sessions_dir()).path_of(version["session_id"])
            except session_store.SessionNotFound:
                raise rag_state.PermanentError(f"会话标识非法：{version['session_id']}")
            if not path.is_file():
                raise rag_state.PermanentError(f"会话原文缺失：{path.name}")
            module = _conversations()
            rag_state.set_conversation_job_stage(conn, job["job_id"], "embed")
            count = module.index_version(version, sessions_dir=_sessions_dir())
            if not count:
                raise rag_state.PermanentError("该会话没有可索引的问答原文")
            module.set_version_active(version["session_id"], version["revision"], True)
            result = rag_state.finish_conversation_job(conn, job["job_id"], ok=True)
            superseded = result.get("superseded")
            if superseded:
                old = rag_state.get_conversation_version(conn, superseded)
                if old is not None:
                    module.delete_version_vectors(old["session_id"], old["revision"])
            _logger.info("indexed conversation job=%s session=%s vectors=%d switched=%s %.1fs",
                         job["job_id"], version["session_id"], count, result.get("switched"),
                         time.monotonic() - started)
        except rag_state.PermanentError as exc:
            rag_state.finish_conversation_job(conn, job["job_id"], ok=False, error=str(exc),
                                              permanent=True)
            _logger.warning("对话索引任务 %s 不可重试：%s", job["job_id"], exc)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            result = rag_state.finish_conversation_job(conn, job["job_id"], ok=False, error=detail)
            _logger.warning("对话索引任务 %s 失败（第 %s 次，%s）：%s", job["job_id"], job["attempts"],
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


class ConversationSync(BaseModel):
    session_id: str | None = None
    all: bool = False


class ConversationSearch(BaseModel):
    query: str = ""
    exclude_sessions: list[str] = Field(default_factory=list)
    top_chunks: int = 20
    max_sessions: int = 3


class ConversationApiSettings(BaseModel):
    """只在本机请求内存里流转的模型配置；不写日志、不落库、不进向量。"""

    API_KEY: str = ""
    MODEL: str = ""
    BASE_URL: str = ""


class ConversationRecall(BaseModel):
    query: str = ""
    session_id: str | None = None
    exclude_session: str | None = None
    include_summary: bool = True
    api_settings: ConversationApiSettings | None = None


class SummaryWrite(BaseModel):
    source_revision: str = ""
    summary_schema_version: int = conversation_summary.SUMMARY_SCHEMA_VERSION
    summary_json: str = ""
    generated_at: str = ""


worker: IngestionWorker | None = None
started_at = ""
# 最近一次全量补齐索引时读不出来的会话文件，供 /memory 显示（/v1/state）。
conversations_broken: list[str] = []


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


# ---------------------------------------------------------------- 对话检索桥接
#
# 跨会话回忆的实现放在 rag/conversations.py（检索、切块、版本）与 conversation_summary.py
# （摘要缓存）。这里只做转发与响应整形；conversations.py 与本文件并行开发，导入失败时按
# "功能未就绪"处理，绝不影响文件服务启动和既有接口。

def _conversations():
    from rag import conversations as module
    return module


def _sessions_dir() -> Path:
    """会话原文目录；默认项目内 sessions/，测试可替换 session_store.SESSIONS_DIR。"""
    return Path(session_store.SESSIONS_DIR)


def _sync_session(session_id: str, conn=None) -> dict:
    """登记/入队一个会话；session 可以是 id，也可以是已读出的会话 dict。"""
    return _conversations().sync_session(session_id, sessions_dir=_sessions_dir(), conn=conn)


def _session_turns(session_id: str) -> list[dict]:
    """读会话原文的问答轮次；读不出来时返回空列表（证据退化成只有命中片段）。"""
    try:
        session = session_store.SessionStore(_sessions_dir()).read(session_id)
    except Exception:
        return []
    return session.get("turns") or []


def _clip_evidence(text: str, budget: int) -> tuple[str, bool]:
    """按剩余字符预算截断证据文本；返回 (文本, 是否被截断)。"""
    if budget <= 0:
        return "", bool(text)
    if len(text) <= budget:
        return text, False
    return text[:budget] + "…", True


def _excerpt(text: str, limit: int = 400) -> str:
    body = " ".join(str(text or "").split()).strip()
    if len(body) <= limit:
        return body
    return body[:limit].rstrip() + "…"


def _hit_payload(hit: dict, candidate: dict | None = None) -> dict:
    """检索命中的固定字段；会话标题/更新时间从所属候选补全。"""
    candidate = candidate or {}
    return {"session_id": hit.get("session_id") or candidate.get("session_id") or "",
            "session_title": hit.get("session_title") or candidate.get("title") or "",
            "session_updated_at": hit.get("session_updated_at") or candidate.get("updated_at"),
            "turn_id": hit.get("turn_id") or "", "turn_index": int(hit.get("turn_index") or 0),
            "role": hit.get("role") or hit.get("primary_role") or "",
            "chunk_index": int(hit.get("chunk_index") or 0), "score": hit.get("score"),
            "primary_hit": bool(hit.get("primary_hit")),
            "start_char": int(hit.get("start_char") or 0), "end_char": int(hit.get("end_char") or 0),
            "text": str(hit.get("text") or ""),
            "excerpt": hit.get("excerpt") or _excerpt(hit.get("text"))}


def _search_payload(found: dict, query: str) -> dict:
    """把对话检索结果整形成 client.py 约定的字段。"""
    candidates = []
    for item in found.get("candidates") or []:
        candidates.append({"session_id": item.get("session_id"), "title": item.get("title"),
                           "updated_at": item.get("updated_at"), "score": item.get("score"),
                           "best_turn_id": item.get("best_turn_id"),
                           "primary_hit": bool(item.get("primary_hit")),
                           "hits": [_hit_payload(hit, item) for hit in (item.get("hits") or [])]})
    metas = {item["session_id"]: item for item in candidates}
    hits = [_hit_payload(hit, metas.get(hit.get("session_id"))) for hit in (found.get("hits") or [])]
    return {"query": found.get("query") or query, "hits": hits, "candidates": candidates,
            "status": found.get("status") or "not_found"}


def _read_session_data(session_id: str) -> dict | None:
    """读会话原文；不存在或读不出来返回 None。"""
    try:
        return session_store.SessionStore(_sessions_dir()).read(session_id)
    except Exception:
        return None


def _primary_candidates(candidates: list[dict]) -> list[dict]:
    """只保留正文命中（primary_hit=True）的候选。

    候选结构里没有这个字段（旧结构/桩数据）时按"全部保留"处理，避免整段回忆失效。
    """
    if not any("primary_hit" in item for item in candidates):
        return list(candidates)
    return [item for item in candidates if item.get("primary_hit")]


def _evidence(session_id: str, hits: list[dict]) -> list[dict]:
    """最多 RECALL_MAX_EVIDENCE 轮真实问答、总字符不超过 RECALL_MAX_CHARS 的证据。"""
    turns = _session_turns(session_id)
    items: list[dict] = []
    seen: set[str] = set()
    used = 0
    for hit in hits:
        if len(items) >= RECALL_MAX_EVIDENCE:
            break
        turn_id = str(hit.get("turn_id") or "")
        if not turn_id or turn_id in seen:
            continue
        seen.add(turn_id)
        index = int(hit.get("turn_index") or 0)
        turn = turns[index] if 0 <= index < len(turns) else {}
        user, user_cut = _clip_evidence(str(turn.get("user") or ""), RECALL_MAX_CHARS - used)
        used += len(user)
        answer, answer_cut = _clip_evidence(str(turn.get("final_answer") or ""), RECALL_MAX_CHARS - used)
        used += len(answer)
        excerpt, excerpt_cut = _clip_evidence(str(hit.get("excerpt") or hit.get("text") or ""),
                                             RECALL_MAX_CHARS - used)
        used += len(excerpt)
        items.append({"turn_id": turn_id, "user": user, "final_answer": answer,
                      "score": hit.get("score"), "excerpt": excerpt,
                      "truncated": bool(user_cut or answer_cut or excerpt_cut),
                      "start_char": int(hit.get("start_char") or 0),
                      "end_char": int(hit.get("end_char") or 0)})
        if used >= RECALL_MAX_CHARS:
            break
    return items


def _tail_evidence(session_id: str, max_turns: int = RECALL_MAX_EVIDENCE) -> list[dict]:
    """定向回忆没有命中片段时，给出该会话最近的若干轮原文；截断仍受字符预算约束。"""
    turns = _session_turns(session_id)
    items: list[dict] = []
    used = 0
    for index in range(max(0, len(turns) - max_turns), len(turns)):
        turn = turns[index]
        user, user_cut = _clip_evidence(str(turn.get("user") or ""), max(0, RECALL_MAX_CHARS - used))
        used += len(user)
        answer, answer_cut = _clip_evidence(str(turn.get("final_answer") or ""),
                                           max(0, RECALL_MAX_CHARS - used))
        used += len(answer)
        items.append({"turn_id": session_store.turn_id(session_id, index), "user": user,
                      "final_answer": answer, "score": None, "excerpt": _excerpt(answer),
                      "truncated": bool(user_cut or answer_cut),
                      "start_char": 0, "end_char": len(str(turn.get("final_answer") or ""))})
        if used >= RECALL_MAX_CHARS:
            break
    return items


def _ambiguous_candidates(candidates: list[dict]) -> list[dict]:
    """模糊结果只给候选与整轮原文片段，不生成任何摘要。"""
    turns_by_session: dict[str, list[dict]] = {}
    out = []
    for item in candidates:
        session_id = item.get("session_id")
        if session_id not in turns_by_session:
            turns_by_session[session_id] = _session_turns(session_id)
        turns = turns_by_session[session_id]
        evidence = []
        for hit in (item.get("hits") or [])[:RECALL_MAX_EVIDENCE]:
            index = int(hit.get("turn_index") or 0)
            turn = turns[index] if 0 <= index < len(turns) else {}
            evidence.append({"turn_id": str(hit.get("turn_id") or ""),
                             "user": _clip_evidence(str(turn.get("user") or ""), 2000)[0],
                             "final_answer": _clip_evidence(str(turn.get("final_answer") or ""), 2000)[0]})
        out.append({"session_id": session_id, "title": item.get("title"),
                    "updated_at": item.get("updated_at"), "score": item.get("score"),
                    "primary_hit": bool(item.get("primary_hit")), "evidence": evidence})
    return out


def _conversation_counts(conn) -> tuple[int, int]:
    """已建索引/待处理会话数；对话模块不可用时退回直接查库，/v1/state 不能因此失败。"""
    try:
        module = _conversations()
        return int(module.indexed_count(conn)), int(module.pending_count(conn))
    except Exception:
        indexed = conn.execute("SELECT COUNT(*) AS n FROM conversations"
                               " WHERE active_revision IS NOT NULL").fetchone()["n"]
        pending = conn.execute("SELECT COUNT(*) AS n FROM conversation_jobs"
                               " WHERE status IN ('queued','processing','retry_wait')").fetchone()["n"]
        return indexed, pending


# ---------------------------------------------------------------- 应用

@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker, started_at, conversations_broken
    setup_logging()
    rag_state.ensure_dirs()
    conn = rag_state.connect()
    try:
        recovered = rag_state.recover_interrupted(conn, time.time())
        try:
            recovered_conversations = rag_state.recover_interrupted_conversations(conn, time.time())
        except Exception as exc:
            recovered_conversations = 0
            _logger.warning("恢复中断的对话索引任务失败：%s", exc)
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
    # 对话记忆：启动时把已有会话全部入队补齐索引，并清理残留向量、校对 active 标记。
    # conversations.py 与文件服务并行开发，任何一步失败都只记日志，不影响服务启动。
    conversation_fixed = 0
    conversations_module = None
    try:
        conversations_module = _conversations()
    except Exception as exc:
        _logger.warning("对话检索模块不可用：%s: %s", type(exc).__name__, exc)
    if conversations_module is not None:
        try:
            synced = conversations_module.sync_all(sessions_dir=_sessions_dir())
            conversations_broken = list(synced.get("broken") or [])
            _logger.info("启动补齐对话索引：会话=%s 入队=%s 读不出=%s", synced.get("sessions"),
                         synced.get("queued"), conversations_broken)
        except Exception as exc:
            _logger.warning("启动补齐对话索引失败：%s: %s", type(exc).__name__, exc)
        try:
            dropped = conversations_module.drop_orphans()
            if dropped:
                _logger.warning("清理了 %d 组没有版本记录的残留对话向量", dropped)
        except Exception as exc:
            _logger.warning("残留对话向量清理失败：%s", exc)
        try:
            conversation_fixed = conversations_module.reconcile_active()
        except Exception as exc:
            _logger.warning("对话向量 active 校对失败：%s", exc)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    worker = IngestionWorker()
    worker.wake()
    worker.start()
    threading.Thread(target=_warm_model, name="rag-warm", daemon=True).start()
    _logger.info("服务启动 pid=%s 恢复中断任务=%d 待处理=%d active 修正=%d 对话恢复=%d 对话active修正=%d",
                 os.getpid(), recovered, pending, fixed, recovered_conversations, conversation_fixed)
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
        conversations = conn.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()["n"]
        conversation_pending = conn.execute(
            "SELECT COUNT(*) AS n FROM conversation_jobs"
            " WHERE status IN ('queued','processing','retry_wait')").fetchone()["n"]
        conversation_failed = conn.execute(
            "SELECT COUNT(*) AS n FROM conversation_jobs WHERE status='failed'").fetchone()["n"]
    finally:
        conn.close()
    try:
        vectors = rag_store.vector_count()
    except Exception as exc:
        vectors = -1
        _logger.warning("向量库不可读：%s", exc)
    try:
        conversation_vectors = _conversations().vector_count()
    except Exception as exc:
        conversation_vectors = -1
        _logger.warning("对话向量库不可读：%s", exc)
    return {"ok": True, "protocol": PROTOCOL, "service_id": service_id(), "pid": os.getpid(),
            "started_at": started_at, "documents": documents, "vectors": vectors,
            "jobs_pending": pending, "jobs_failed": failed, "conversations": conversations,
            "conversation_vectors": conversation_vectors,
            "conversation_jobs_pending": conversation_pending,
            "conversation_jobs_failed": conversation_failed,
            "sessions_dir": str(_sessions_dir()),
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
        window = max(1, min(int(limit), 100))
        events = rag_state.recent_events(conn, window)
        jobs = [dict(row) for row in conn.execute(
            "SELECT j.job_id, j.status, j.attempts, j.max_attempts, j.round, j.last_error,"
            " v.file_name, v.kind, v.id AS version_id, v.status AS version_status"
            " FROM jobs j JOIN versions v ON v.id=j.version_id"
            " WHERE j.status IN ('failed','queued','processing','retry_wait')"
            " ORDER BY j.updated_at DESC LIMIT ?", (window,)).fetchall()]
        conversation_jobs = rag_state.conversation_job_snapshot(conn, window)
        indexed, conversations_pending = _conversation_counts(conn)
        return {"events": events, "jobs": jobs, "model_ready": rag_store.model_ready(),
                "conversation_jobs": conversation_jobs, "conversations_indexed": indexed,
                "conversations_pending": conversations_pending,
                "conversations_broken": list(conversations_broken)}
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


# ---------------------------------------------------------------- 对话记忆接口

@app.post("/v1/conversations/sync")
async def conversations_sync(request: ConversationSync):
    return await run_in_threadpool(sync_conversations, request)


def sync_conversations(request: ConversationSync) -> dict | JSONResponse:
    """把一个或全部会话登记进对话索引；只入队，切块与 embedding 交给后台执行器。"""
    global conversations_broken
    try:
        module = _conversations()
    except Exception as exc:
        return _error(503, "unavailable", f"对话检索模块未就绪（{type(exc).__name__}）")
    try:
        if request.all:
            summary = module.sync_all(sessions_dir=_sessions_dir())
            conversations_broken = list(summary.get("broken") or [])
            result = {"sessions": int(summary.get("sessions") or 0),
                      "queued": int(summary.get("queued") or 0),
                      "broken": conversations_broken,
                      "items": list(summary.get("items") or [])}
        else:
            session_id = (request.session_id or "").strip()
            if not session_id:
                return _error(400, "bad_request", "session_id 与 all 至少要给一个")
            item = _sync_session(session_id)
            result = {"sessions": 1, "queued": 1 if item.get("queued") else 0,
                      "broken": [], "items": [item]}
    except Exception as exc:
        _logger.warning("会话同步失败：%s: %s", type(exc).__name__, exc)
        return _error(500, "sync_error", f"{type(exc).__name__}: {exc}")
    _wakeup(result)
    return result


@app.post("/v1/conversations/search")
async def conversations_search(request: ConversationSearch):
    return await run_in_threadpool(search_conversations, request)


def search_conversations(request: ConversationSearch) -> dict | JSONResponse:
    """语义检索历史对话片段；空问题按参数错误返回，不做检索、不唤醒执行器。"""
    query = (request.query or "").strip()
    if not query:
        return _error(400, "bad_request", "query 不能为空")
    try:
        module = _conversations()
    except Exception as exc:
        return _error(503, "unavailable", f"对话检索模块未就绪（{type(exc).__name__}）")
    try:
        found = module.search_conversations(query, exclude_sessions=list(request.exclude_sessions or []),
                                           top_chunks=int(request.top_chunks),
                                           max_sessions=int(request.max_sessions),
                                           sessions_dir=_sessions_dir())
    except Exception as exc:
        _logger.warning("对话检索失败：%s: %s", type(exc).__name__, exc)
        return _error(500, "search_error", f"{type(exc).__name__}: {exc}")
    return _search_payload(found, query)


@app.post("/v1/conversations/recall")
async def conversations_recall(request: ConversationRecall):
    return await run_in_threadpool(recall_conversation, request)


def _recall_search(module, query: str, *, exclude_sessions, max_sessions: int) -> dict:
    return module.search_conversations(query, exclude_sessions=exclude_sessions,
                                       top_chunks=CONVERSATION_TOP_CHUNKS,
                                       max_sessions=max_sessions, sessions_dir=_sessions_dir())


def _recall_summary(session_id: str, request: ConversationRecall) -> dict:
    """按需整理摘要；任何失败都只降级成 summary=None，信息不回显、不落库。"""
    if not request.include_summary:
        return {"summary_status": "skipped", "summary": None, "note": None, "source_revision": None,
                "current_revision": None, "stale": False}
    settings = request.api_settings.model_dump() if request.api_settings is not None else None
    try:
        outcome = conversation_summary.use_or_create_summary(session_id, api_settings=settings)
    except Exception as exc:
        _logger.warning("会话摘要失败：%s: %s", type(exc).__name__, exc)
        outcome = {"status": "failed", "summary": None, "note": "摘要生成失败，本次只返回原文证据"}
    return {"summary_status": outcome.get("status") or "failed", "summary": outcome.get("summary"),
            "note": outcome.get("note"), "source_revision": outcome.get("source_revision"),
            "current_revision": outcome.get("current_revision"), "stale": bool(outcome.get("stale"))}


def _recall_found(query: str, session_meta: dict, evidence: list[dict], candidates: list[dict],
                  request: ConversationRecall) -> dict:
    """found 的固定响应：命中会话 + 摘要 + 真实问答证据 + 候选概览。"""
    summary = _recall_summary(session_meta["id"], request)
    return {"status": "found", "query": query, "session": session_meta,
            "summary_status": summary["summary_status"], "summary": summary["summary"],
            "summary_note": summary["note"], "summary_source_revision": summary["source_revision"],
            "summary_current_revision": summary["current_revision"], "summary_stale": summary["stale"],
            "evidence": evidence,
            "candidates": [{"session_id": item.get("session_id"), "title": item.get("title"),
                            "updated_at": item.get("updated_at"), "score": item.get("score"),
                            "primary_hit": bool(item.get("primary_hit"))} for item in candidates]}


def recall_conversation(request: ConversationRecall) -> dict | JSONResponse:
    """跨会话回忆：先检索历史对话；命中唯一候选时按需整理摘要，并附真实问答原文。

    session_id 有值 = 用户指定的定向回忆：那个会话就是答案来源（即使它不是最佳候选），
    检索只用来取它的命中片段；没有命中片段时退回该会话最近的几轮原文。
    session_id 为空 = 自动回忆：排除 exclude_session（当前会话）后按候选判定，且只认
    primary_hit=True 的候选（正文直接命中的才算）。摘要失败或超时都不能让回忆整体失败：
    摘要置空，evidence 照常返回。
    """
    query = (request.query or "").strip()
    if not query:
        return _error(400, "bad_request", "query 不能为空")
    try:
        module = _conversations()
    except Exception as exc:
        return _error(503, "unavailable", f"对话检索模块未就绪（{type(exc).__name__}）")
    target = (request.session_id or "").strip()
    excluded = [value for value in ((request.exclude_session or "").strip(),) if value]
    try:
        if target:
            found = _recall_search(module, query, exclude_sessions=excluded,
                                   max_sessions=RECALL_TARGET_SESSIONS)
            data = _read_session_data(target)
            if data is None:
                return {"status": "not_found", "query": query, "candidates": []}
            hit = next((item for item in (found.get("candidates") or [])
                        if item.get("session_id") == target and item.get("primary_hit")), None)
            hits = list((hit or {}).get("hits") or [])
            # 定向回忆以指定会话为准：有命中片段就用片段，没有就退回它最近的几轮原文
            evidence = _evidence(target, hits) if hits else _tail_evidence(target)
            meta = {"id": target, "title": data.get("title"), "updated_at": data.get("updated_at")}
            return _recall_found(query, meta, evidence, found.get("candidates") or [], request)
        found = _recall_search(module, query, exclude_sessions=excluded,
                               max_sessions=CONVERSATION_MAX_SESSIONS)
        candidates = _primary_candidates(found.get("candidates") or [])
        if (found.get("status") or "not_found") == "not_found" or not candidates:
            return {"status": "not_found", "query": query, "candidates": []}
        if (found.get("status") or "") == "ambiguous" and len(candidates) > 1:
            return {"status": "ambiguous", "query": query,
                    "candidates": _ambiguous_candidates(candidates)}
        best = candidates[0]
        meta = {"id": best.get("session_id"), "title": best.get("title"),
                "updated_at": best.get("updated_at")}
        return _recall_found(query, meta, _evidence(best.get("session_id"), best.get("hits") or []),
                             candidates, request)
    except Exception as exc:
        _logger.warning("回忆失败：%s: %s", type(exc).__name__, exc)
        return _error(500, "recall_error", f"{type(exc).__name__}: {exc}")


@app.get("/v1/conversations/{session_id}")
async def conversation_detail(session_id: str):
    return await run_in_threadpool(conversation_snapshot, session_id)


def conversation_snapshot(session_id: str) -> dict | JSONResponse:
    """一个会话的完整问答、索引状态、当前任务与摘要记录；管理、排错用。"""
    try:
        session = session_store.SessionStore(_sessions_dir()).read(session_id)
    except session_store.SessionNotFound:
        return _error(404, "not_found", f"没有会话 {session_id}")
    except Exception as exc:
        return _error(500, "session_error", f"{type(exc).__name__}: {exc}")
    revision = session_store.revision_of(session)
    conn = rag_state.connect()
    try:
        version = (rag_state.active_conversation_version(conn, session_id)
                   or rag_state.latest_conversation_version(conn, session_id))
        index = None
        job = None
        if version is not None:
            index = {"version_id": version["id"], "revision": version["revision"],
                     "status": version["status"], "turn_count": version["turn_count"],
                     "chunk_count": version["chunk_count"], "max_tokens": version["max_tokens"],
                     "active": rag_state.conversation_revision(conn, session_id) == version["revision"]}
            record = rag_state.current_conversation_job(conn, session_id, version["id"])
            if record is not None:
                job = {"job_id": record["job_id"], "status": record["status"], "stage": record["stage"],
                       "attempts": record["attempts"], "max_attempts": record["max_attempts"],
                       "round": record["round"], "last_error": record["last_error"]}
        summary_record = rag_state.get_summary_record(conn, session_id, revision,
                                                      conversation_summary.SUMMARY_SCHEMA_VERSION)
        if summary_record is None:
            summary_record = conn.execute(
                "SELECT * FROM conversation_summaries WHERE session_id=?"
                " ORDER BY generated_at DESC LIMIT 1", (session_id,)).fetchone()
    finally:
        conn.close()
    return {"session": {"id": session_id, "title": session["title"],
                        "created_at": session["created_at"], "updated_at": session["updated_at"],
                        "turn_count": len(session.get("turns") or [])},
            "turns": [{"turn_id": session_store.turn_id(session_id, position),
                       "turn_index": position, "user": turn["user"], "final_answer": turn["final_answer"]}
                      for position, turn in enumerate(session.get("turns") or [])],
            "index": index, "job": job,
            "summary": None if summary_record is None else {
                "source_revision": summary_record["source_revision"],
                "summary_schema_version": summary_record["summary_schema_version"],
                "generated_at": summary_record["generated_at"],
                "summary_json": summary_record["summary_json"]}}


@app.get("/v1/conversations/{session_id}/summary")
async def conversation_summary_record(session_id: str, revision: str | None = None):
    return await run_in_threadpool(summary_snapshot, session_id, revision)


def summary_snapshot(session_id: str, revision: str | None) -> dict | JSONResponse:
    """摘要缓存记录；current_revision 是会话原文当前的 revision，用于判断摘要是否过时。"""
    try:
        session = session_store.SessionStore(_sessions_dir()).read(session_id)
    except session_store.SessionNotFound:
        return _error(404, "not_found", f"没有会话 {session_id}")
    except Exception as exc:
        return _error(500, "session_error", f"{type(exc).__name__}: {exc}")
    current_revision = session_store.revision_of(session)
    target = (revision or "").strip() or current_revision
    conn = rag_state.connect()
    try:
        record = rag_state.get_summary_record(conn, session_id, target,
                                              conversation_summary.SUMMARY_SCHEMA_VERSION)
        if record is None:
            record = conn.execute(
                "SELECT * FROM conversation_summaries WHERE session_id=? AND source_revision=?"
                " ORDER BY summary_schema_version DESC LIMIT 1", (session_id, target)).fetchone()
    finally:
        conn.close()
    return {"session_id": session_id, "current_revision": current_revision,
            "record": None if record is None else {
                "source_revision": record["source_revision"],
                "summary_schema_version": record["summary_schema_version"],
                "generated_at": record["generated_at"],
                "summary_json": record["summary_json"]}}


@app.put("/v1/conversations/{session_id}/summary")
async def put_conversation_summary(session_id: str, request: SummaryWrite):
    return await run_in_threadpool(store_summary, session_id, request)


def store_summary(session_id: str, request: SummaryWrite) -> dict | JSONResponse:
    """写回摘要缓存：只接受 JSON 对象，且源版本必须真的是这个会话的当前版本或已登记版本。

    会话原文读不出来时不拦（服务看不到会话文件，不该替调用方猜）；能读到就校验，
    避免把摘要挂到一个凭空造出来的版本上，日后被当成"覆盖了某版本"的证据。
    """
    source_revision = (request.source_revision or "").strip()
    if not source_revision:
        return _error(400, "bad_request", "source_revision 不能为空")
    try:
        parsed = json.loads(request.summary_json)
    except ValueError:
        return _error(400, "bad_request", "summary_json 必须是合法 JSON")
    if not isinstance(parsed, dict):
        return _error(400, "bad_request", "summary_json 必须是 JSON 对象")
    schema_version = int(request.summary_schema_version)
    generated_at = (request.generated_at or "").strip() or \
        datetime.now().astimezone().isoformat(timespec="seconds")
    conn = rag_state.connect()
    try:
        try:
            current = session_store.revision_of(
                session_store.SessionStore(_sessions_dir()).read(session_id))
        except Exception:
            current = None
        if current is not None and current != source_revision and \
                rag_state.version_by_revision(conn, session_id, source_revision) is None:
            return _error(409, "revision_mismatch",
                          "source_revision 不是该会话的当前版本，也不是已登记的会话版本")
        rag_state.put_summary_record(conn, session_id=session_id, source_revision=source_revision,
                                     schema_version=schema_version,
                                     summary_json=json.dumps(parsed, ensure_ascii=False),
                                     generated_at=generated_at)
        conn.commit()
    finally:
        conn.close()
    return {"session_id": session_id, "source_revision": source_revision,
            "summary_schema_version": schema_version, "stored": True}


@app.post("/v1/jobs/{job_id}/retry")
async def retry(job_id: str):
    return await run_in_threadpool(start_retry, job_id)

def start_retry(job_id: str) -> dict | JSONResponse:
    """用户主动重试：对话索引任务（cjob_...）走对话入口，其余走文件入口。

    以前无条件调 retry_job()，cjob_... 在文件任务表里查不到，只会得到 404。
    两者都是新开一轮有上限的尝试；成功后统一唤醒执行器，否则新任务要等到
    下一轮轮询才会被取走。
    """
    conn = rag_state.connect()
    try:
        conversation = rag_state.get_conversation_job(conn, job_id) is not None
        retry = rag_state.retry_conversation_job if conversation else rag_state.retry_job
        job = retry(conn, job_id)
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
               # 服务身份还包含会话目录：客户端只用同一个会话目录的实例，否则对话记忆会错位。
               "sessions_dir": str(_sessions_dir()),
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
