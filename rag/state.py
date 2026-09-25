"""RAG 文件服务的持久状态：文件、版本、入库任务、通知事件（SQLite，标准库）。

数据根目录固定在本项目下（rag_data/），不随启动终端的工作目录变化；测试通过替换
rag_state.DATA_ROOT 隔离。这里只保存"任务与元数据"，向量本身在 Chroma（rag_store.py）。

任务重试口径（file_service 的调度器依赖本模块保证）：
- 一个文件版本对应一条当前任务（round 最大的那一轮），执行前先把 attempts +1 落盘；
- 服务重启不把 attempts 清零，进程中断的那一次同样计入；
- 只有用户主动重试才新增一轮（round + 1）尝试记录。
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

DATA_ROOT = Path(os.environ.get("GAGENT_RAG_DATA") or
                 Path(__file__).resolve().parents[1] / "rag_data")

PROTOCOL = "gagent-rag/1"
JOB_MAX_ATTEMPTS = 3
RETRY_DELAYS = (5.0, 30.0)  # 第 1、2 次失败后的等待秒数；第 3 次失败直接终止

# 版本状态：stored（已落盘待处理）→ ready（向量已写入但未生效）→ indexed（已激活）
# superseded 表示被更新版本替换，failed 表示本轮尝试用尽。
_ready_root: Path | None = None


def files_dir() -> Path:
    return DATA_ROOT / "files"


def vectors_dir() -> Path:
    return DATA_ROOT / "vectors"


def models_dir() -> Path:
    return DATA_ROOT / "models"


def logs_dir() -> Path:
    return DATA_ROOT / "logs"


def db_path() -> Path:
    return DATA_ROOT / "state.sqlite"


def service_info_path() -> Path:
    return DATA_ROOT / "service.json"


def service_lock_path() -> Path:
    return DATA_ROOT / "service.lock"


def client_state_path() -> Path:
    return DATA_ROOT / "notices.json"


def service_id() -> str:
    """按项目根目录算实例标识：同项目共用一个服务，不同项目不会互相误用。"""
    root = str(Path(__file__).resolve().parents[1]).casefold().replace("\\", "/")
    return hashlib.sha256(root.encode("utf-8")).hexdigest()[:12]


def ensure_dirs() -> None:
    for path in (DATA_ROOT, files_dir(), files_dir() / "pdf", files_dir() / "excel",
                 vectors_dir(), models_dir(), logs_dir()):
        path.mkdir(parents=True, exist_ok=True)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id         TEXT PRIMARY KEY,
    source_url     TEXT NOT NULL,
    kind           TEXT NOT NULL,
    file_name      TEXT NOT NULL,
    active_version INTEGER,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    UNIQUE (source_url, kind)
);
CREATE TABLE IF NOT EXISTS versions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id       TEXT NOT NULL REFERENCES documents (doc_id),
    sha256       TEXT NOT NULL,
    kind         TEXT NOT NULL,
    local_path   TEXT NOT NULL,
    size         INTEGER NOT NULL,
    file_name    TEXT NOT NULL,
    final_url    TEXT,
    summary      TEXT,
    page_count   INTEGER,
    empty_pages  TEXT,
    chunk_count  INTEGER,
    max_tokens   INTEGER,
    sheet_names  TEXT,
    status       TEXT NOT NULL,
    error        TEXT,
    created_at   TEXT NOT NULL,
    indexed_at   TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES documents (doc_id),
    version_id      INTEGER NOT NULL REFERENCES versions (id),
    kind            TEXT NOT NULL,
    status          TEXT NOT NULL,
    stage           TEXT,
    attempts        INTEGER NOT NULL DEFAULT 0,
    max_attempts    INTEGER NOT NULL DEFAULT 3,
    round           INTEGER NOT NULL DEFAULT 1,
    next_attempt_at REAL,
    last_error      TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE (doc_id, version_id, round)
);
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    doc_id     TEXT,
    version_id INTEGER,
    job_id     TEXT,
    file_name  TEXT,
    message    TEXT,
    detail     TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS versions_doc ON versions (doc_id, sha256);
CREATE INDEX IF NOT EXISTS jobs_due ON jobs (status, next_attempt_at);
CREATE INDEX IF NOT EXISTS jobs_version ON jobs (doc_id, version_id);
"""

_init_lock = threading.Lock()


def connect() -> sqlite3.Connection:
    """每次调用一条连接：服务是多进程内的多线程 + 单执行器，WAL 下按线程各用一条最省心。"""
    global _ready_root
    ensure_dirs()
    with _init_lock:
        if _ready_root != DATA_ROOT:
            conn = sqlite3.connect(str(db_path()), timeout=30, isolation_level=None)
            try:
                conn.executescript(_SCHEMA)
            finally:
                conn.close()
            _ready_root = DATA_ROOT
    conn = sqlite3.connect(str(db_path()), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _row(value: sqlite3.Row | None) -> dict | None:
    return dict(value) if value is not None else None


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def canonical_url(url: str) -> str:
    """资料身份用去掉 fragment 的规范地址：同一文档的 #page=5 不该算两份。"""
    parts = urlsplit(url.strip())
    path = unquote(parts.path) or "/"
    query = f"?{parts.query}" if parts.query else ""
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}{path}{query}"


# ---------------------------------------------------------------- 文件与版本

def get_or_create_document(conn: sqlite3.Connection, source_url: str, kind: str,
                           file_name: str) -> dict:
    key = canonical_url(source_url)
    stamp = now_text()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = _row(conn.execute("SELECT * FROM documents WHERE source_url=? AND kind=?",
                                (key, kind)).fetchone())
        if row is None:
            doc_id = "doc_" + uuid.uuid4().hex[:12]
            conn.execute("INSERT INTO documents (doc_id, source_url, kind, file_name, created_at, updated_at)"
                         " VALUES (?,?,?,?,?,?)", (doc_id, key, kind, file_name, stamp, stamp))
            row = {"doc_id": doc_id, "source_url": key, "kind": kind, "file_name": file_name,
                   "active_version": None, "created_at": stamp, "updated_at": stamp}
        elif row["file_name"] != file_name:
            conn.execute("UPDATE documents SET file_name=?, updated_at=? WHERE doc_id=?",
                         (file_name, stamp, row["doc_id"]))
            row["file_name"] = file_name
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return row


def version_by_sha(conn: sqlite3.Connection, doc_id: str, sha256: str) -> dict | None:
    return _row(conn.execute("SELECT * FROM versions WHERE doc_id=? AND sha256=?",
                             (doc_id, sha256)).fetchone())


def get_version(conn: sqlite3.Connection, version_id: int) -> dict | None:
    return _row(conn.execute("SELECT * FROM versions WHERE id=?", (version_id,)).fetchone())


def add_version(conn: sqlite3.Connection, *, doc_id: str, sha256: str, kind: str, local_path: str,
                size: int, file_name: str, final_url: str, summary: str | None,
                page_count: int | None, empty_pages: list[int] | None,
                sheet_names: list[str] | None) -> tuple[dict, bool]:
    """登记版本；同一 (doc_id, sha256) 复用已有记录，不新增版本。

    已存在时只补齐缺失元数据（例如首次入库前拿到的 summary）并把被删掉的本地文件路径刷新，
    不会重置状态，避免重复读取把 failed 洗成待处理。
    """
    stamp = now_text()
    conn.execute("BEGIN IMMEDIATE")
    try:
        existing = version_by_sha(conn, doc_id, sha256)
        if existing is None:
            conn.execute(
                "INSERT INTO versions (doc_id, sha256, kind, local_path, size, file_name, final_url,"
                " summary, page_count, empty_pages, sheet_names, status, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,'stored',?)",
                (doc_id, sha256, kind, local_path, size, file_name, final_url, summary, page_count,
                 json.dumps(empty_pages) if empty_pages is not None else None,
                 json.dumps(sheet_names or []), stamp))
            row = version_by_sha(conn, doc_id, sha256)
            conn.commit()
            return row, True
        updates, values = [], []
        if summary and not existing["summary"]:
            updates.append("summary=?")
            values.append(summary.strip())
        if local_path and existing["local_path"] != local_path and not Path(existing["local_path"]).is_file():
            updates.append("local_path=?")
            values.append(local_path)
        if page_count is not None and existing["page_count"] != page_count:
            updates.append("page_count=?")
            values.append(page_count)
        # 空页清单：NULL 表示"还没读过全文"，一旦知道就必须写进去，不能被后来的读取丢掉
        if empty_pages is not None and not existing["empty_pages"]:
            updates.append("empty_pages=?")
            values.append(json.dumps(empty_pages))
        if sheet_names and not json.loads(existing["sheet_names"] or "[]"):
            updates.append("sheet_names=?")
            values.append(json.dumps(sheet_names))
        if updates:
            values.append(existing["id"])
            conn.execute(f"UPDATE versions SET {','.join(updates)} WHERE id=?", values)
        conn.commit()
        return version_by_sha(conn, doc_id, sha256), False
    except Exception:
        conn.rollback()
        raise


def set_version_status(conn: sqlite3.Connection, version_id: int, status: str,
                       error: str | None = None) -> None:
    indexed_at = now_text() if status == "indexed" else None
    conn.execute("UPDATE versions SET status=?, error=?, indexed_at=COALESCE(?, indexed_at) WHERE id=?",
                 (status, error, indexed_at, version_id))


def set_version_pages(conn: sqlite3.Connection, version_id: int, page_count: int, empty_pages: list[int],
                      chunk_count: int = 0, max_tokens: int = 0) -> None:
    """后台读完整份文件后回填真实页数、无文字页和切块统计。

    读取阶段只知道当时的信息；chunk_count/max_tokens 让"是否覆盖全文、块长是否被截断"
    可以通过服务接口核对，不用第二个进程去读同一个 Chroma 目录。
    """
    conn.execute("UPDATE versions SET page_count=?, empty_pages=?, chunk_count=?, max_tokens=? WHERE id=?",
                 (page_count, json.dumps(empty_pages), chunk_count, max_tokens, version_id))


def latest_version(conn: sqlite3.Connection, doc_id: str) -> dict | None:
    return _row(conn.execute("SELECT * FROM versions WHERE doc_id=? ORDER BY id DESC LIMIT 1",
                             (doc_id,)).fetchone())


def active_version_id(conn: sqlite3.Connection, doc_id: str) -> int | None:
    row = conn.execute("SELECT active_version FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    return row["active_version"] if row is not None else None


def activate_version(conn: sqlite3.Connection, version_id: int) -> tuple[bool, int | None]:
    """把版本设为有效版本；返回 (是否切换, 被替换的版本 id)。

    只前进不后退：较早创建的版本即使更晚完成入库，也不会抢回有效版本，
    否则一次迟到的重试就能把资料倒回旧内容。
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        version = get_version(conn, version_id)
        if version is None:
            raise KeyError(f"版本不存在：{version_id}")
        doc = conn.execute("SELECT * FROM documents WHERE doc_id=?", (version["doc_id"],)).fetchone()
        current = doc["active_version"]
        set_version_status(conn, version_id, "indexed")
        if current == version_id:
            conn.commit()
            return True, None
        if current is not None and current > version_id:
            conn.commit()
            return False, None
        if current is not None:
            conn.execute("UPDATE versions SET status='superseded', error=NULL WHERE id=?", (current,))
        conn.execute("UPDATE documents SET active_version=?, updated_at=? WHERE doc_id=?",
                     (version_id, now_text(), version["doc_id"]))
        conn.commit()
        return True, current
    except Exception:
        conn.rollback()
        raise


def all_active_versions(conn: sqlite3.Connection) -> list[dict]:
    """启动时校对 Chroma 的 active 标记用。"""
    rows = conn.execute("SELECT v.* FROM versions v JOIN documents d ON d.doc_id=v.doc_id"
                        " WHERE d.active_version=v.id")
    return [dict(r) for r in rows]


def active_version_shas(conn: sqlite3.Connection) -> list[str]:
    """当前有效版本的内容摘要清单：检索只让这些版本参与候选竞争。"""
    rows = conn.execute("SELECT v.sha256 FROM versions v JOIN documents d ON d.active_version=v.id")
    return [row["sha256"] for row in rows]


def known_version_shas(conn: sqlite3.Connection) -> set[str]:
    return {row["sha256"] for row in conn.execute("SELECT sha256 FROM versions")}


def ingestion_state(conn: sqlite3.Connection, doc_ids: list[str]) -> dict[str, dict]:
    """每个资料的"当前有效版本"和"最新版本"，检索侧用它判断结果是否已经过期。"""
    state = {}
    for doc_id in dict.fromkeys(doc_ids):
        row = conn.execute(
            "SELECT d.doc_id AS doc_id, d.active_version AS active_version,"
            " (SELECT v.id FROM versions v WHERE v.doc_id=d.doc_id ORDER BY v.id DESC LIMIT 1) AS latest_id,"
            " (SELECT v.status FROM versions v WHERE v.doc_id=d.doc_id ORDER BY v.id DESC LIMIT 1) AS latest_status"
            " FROM documents d WHERE d.doc_id=?", (doc_id,)).fetchone()
        if row is not None:
            state[doc_id] = dict(row)
    return state


# ---------------------------------------------------------------- 任务

def current_job(conn: sqlite3.Connection, doc_id: str, version_id: int) -> dict | None:
    return _row(conn.execute("SELECT * FROM jobs WHERE doc_id=? AND version_id=?"
                             " ORDER BY round DESC LIMIT 1", (doc_id, version_id)).fetchone())


def get_job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    return _row(conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())


def ensure_job(conn: sqlite3.Connection, doc_id: str, version_id: int, kind: str) -> tuple[dict, bool]:
    """取得该版本当前这一轮任务；已索引或失败的版本不会因此重跑。"""
    stamp = now_text()
    conn.execute("BEGIN IMMEDIATE")
    try:
        existing = current_job(conn, doc_id, version_id)
        if existing is not None:
            conn.commit()
            return existing, False
        job = {"job_id": "job_" + uuid.uuid4().hex[:12], "doc_id": doc_id, "version_id": version_id,
               "kind": kind, "status": "queued", "stage": None, "attempts": 0,
               "max_attempts": JOB_MAX_ATTEMPTS, "round": 1, "next_attempt_at": time.time(),
               "last_error": None, "created_at": stamp, "updated_at": stamp}
        conn.execute(
            "INSERT INTO jobs (job_id, doc_id, version_id, kind, status, stage, attempts, max_attempts,"
            " round, next_attempt_at, last_error, created_at, updated_at)"
            " VALUES (:job_id,:doc_id,:version_id,:kind,:status,:stage,:attempts,:max_attempts,:round,"
            " :next_attempt_at,:last_error,:created_at,:updated_at)", job)
        conn.commit()
        return job, True
    except Exception:
        conn.rollback()
        raise


def claim_job(conn: sqlite3.Connection, now: float) -> dict | None:
    """取出一个到期任务并把尝试次数 +1 落盘（先记账再执行，崩溃也算一次）。"""
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status IN ('queued','retry_wait') AND"
            " COALESCE(next_attempt_at,0)<=? ORDER BY next_attempt_at, created_at LIMIT 1",
            (now,)).fetchone()
        if row is None:
            conn.commit()
            return None
        conn.execute("UPDATE jobs SET status='processing', attempts=attempts+1, updated_at=?"
                     " WHERE job_id=?", (now_text(), row["job_id"]))
        job = get_job(conn, row["job_id"])
        conn.commit()
        return job
    except Exception:
        conn.rollback()
        raise


def next_due_at(conn: sqlite3.Connection) -> float | None:
    row = conn.execute("SELECT MIN(next_attempt_at) AS due FROM jobs"
                       " WHERE status IN ('queued','retry_wait')").fetchone()
    return row["due"] if row is not None and row["due"] is not None else None


def set_job_stage(conn: sqlite3.Connection, job_id: str, stage: str) -> None:
    conn.execute("UPDATE jobs SET stage=?, updated_at=? WHERE job_id=?", (stage, now_text(), job_id))


class PermanentError(Exception):
    """确定性问题：参数错误、空 summary、无可提取文字等，不重复执行。"""


def finish_job(conn: sqlite3.Connection, job_id: str, *, ok: bool, error: str | None = None,
               permanent: bool = False, stage: str | None = None) -> dict:
    """收尾一次尝试：决定 indexed / retry_wait / failed，并在终态写事件。

    事件与状态写在同一个事务里，重启后不会重复弹提示。
    """
    stamp = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        job = get_job(conn, job_id)
        if job is None:
            raise KeyError(f"任务不存在：{job_id}")
        version = get_version(conn, job["version_id"])
        if ok:
            switched, superseded = activate_version_in_tx(conn, job["version_id"])
            conn.execute("UPDATE jobs SET status='indexed', stage=?, last_error=NULL, updated_at=?"
                         " WHERE job_id=?", ("activate", now_text(), job_id))
            if switched:
                append_event(conn, kind="indexed", doc_id=job["doc_id"], version_id=job["version_id"],
                             job_id=job_id, file_name=version["file_name"],
                             message=f"{version['file_name']} 已加入长期记忆",
                             detail=json.dumps({"sha256": version["sha256"], "kind": version["kind"]}))
            result = {"status": "indexed", "switched": switched, "superseded": superseded}
        else:
            attempts = job["attempts"]
            give_up = permanent or attempts >= job["max_attempts"]
            if give_up:
                conn.execute("UPDATE jobs SET status='failed', stage=?, last_error=?, updated_at=?"
                             " WHERE job_id=?", (stage or job["stage"], error, now_text(), job_id))
                conn.execute("UPDATE versions SET status='failed', error=? WHERE id=?",
                             (error, job["version_id"]))
                append_event(conn, kind="failed", doc_id=job["doc_id"], version_id=job["version_id"],
                             job_id=job_id, file_name=version["file_name"],
                             message=f"{version['file_name']} 加入长期记忆失败，已停止自动重试",
                             detail=error)
                result = {"status": "failed"}
            else:
                delay = RETRY_DELAYS[min(attempts - 1, len(RETRY_DELAYS) - 1)]
                conn.execute("UPDATE jobs SET status='retry_wait', stage=?, last_error=?,"
                             " next_attempt_at=?, updated_at=? WHERE job_id=?",
                             (stage or job["stage"], error, stamp + delay, now_text(), job_id))
                result = {"status": "retry_wait", "delay": delay}
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise


def activate_version_in_tx(conn: sqlite3.Connection, version_id: int) -> tuple[bool, int | None]:
    """activate_version 的事务内版本：finish_job 已经开了事务，不能再嵌套 BEGIN。"""
    version = get_version(conn, version_id)
    doc = conn.execute("SELECT * FROM documents WHERE doc_id=?", (version["doc_id"],)).fetchone()
    current = doc["active_version"]
    set_version_status(conn, version_id, "indexed")
    if current == version_id:
        return True, None
    if current is not None and current > version_id:
        return False, None
    if current is not None:
        conn.execute("UPDATE versions SET status='superseded', error=NULL WHERE id=?", (current,))
    conn.execute("UPDATE documents SET active_version=?, updated_at=? WHERE doc_id=?",
                 (version_id, now_text(), version["doc_id"]))
    return True, current


def recover_interrupted(conn: sqlite3.Connection, now: float) -> int:
    """服务启动时接管上次进程留下的 processing 任务；那一次尝试已经记在账上。"""
    rows = conn.execute("SELECT job_id, attempts, max_attempts FROM jobs WHERE status='processing'").fetchall()
    for row in rows:
        if row["attempts"] >= row["max_attempts"]:
            finish_job(conn, row["job_id"], ok=False, error="服务在处理该任务时中断，尝试次数已用尽",
                       permanent=True)
        else:
            conn.execute("UPDATE jobs SET status='retry_wait', next_attempt_at=?, updated_at=?"
                         " WHERE job_id=?", (now, now_text(), row["job_id"]))
    return len(rows)


def retry_job(conn: sqlite3.Connection, job_id: str) -> dict:
    """用户主动重试：新建一轮有上限的尝试，不动联网下载与问答本身。"""
    stamp = now_text()
    conn.execute("BEGIN IMMEDIATE")
    try:
        job = get_job(conn, job_id)
        if job is None:
            raise KeyError(f"任务不存在：{job_id}")
        current = current_job(conn, job["doc_id"], job["version_id"])
        if current["job_id"] != job_id:
            raise ValueError(f"该任务已被新一轮重试 {current['job_id']} 替代，请重试最新那一轮")
        if job["status"] in ("queued", "processing", "retry_wait"):
            raise ValueError("该任务仍在排队或执行中，不能重复提交重试")
        if job["status"] == "indexed":
            raise ValueError("该文件版本已经入库，无需重试")
        fresh = {"job_id": "job_" + uuid.uuid4().hex[:12], "doc_id": job["doc_id"],
                 "version_id": job["version_id"], "kind": job["kind"], "status": "queued",
                 "stage": None, "attempts": 0, "max_attempts": JOB_MAX_ATTEMPTS,
                 "round": current["round"] + 1, "next_attempt_at": time.time(), "last_error": None,
                 "created_at": stamp, "updated_at": stamp}
        conn.execute(
            "INSERT INTO jobs (job_id, doc_id, version_id, kind, status, stage, attempts, max_attempts,"
            " round, next_attempt_at, last_error, created_at, updated_at)"
            " VALUES (:job_id,:doc_id,:version_id,:kind,:status,:stage,:attempts,:max_attempts,:round,"
            " :next_attempt_at,:last_error,:created_at,:updated_at)", fresh)
        conn.execute("UPDATE versions SET status='stored', error=NULL WHERE id=?", (job["version_id"],))
        conn.commit()
        return fresh
    except Exception:
        conn.rollback()
        raise


# ---------------------------------------------------------------- 事件

def append_event(conn: sqlite3.Connection, *, kind: str, doc_id: str | None, version_id: int | None,
                 job_id: str | None, file_name: str | None, message: str, detail: str | None) -> dict:
    stamp = now_text()
    cursor = conn.execute(
        "INSERT INTO events (kind, doc_id, version_id, job_id, file_name, message, detail, created_at)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (kind, doc_id, version_id, job_id, file_name, message, detail, stamp))
    return {"id": cursor.lastrowid, "kind": kind, "doc_id": doc_id, "version_id": version_id,
            "job_id": job_id, "file_name": file_name, "message": message, "detail": detail,
            "created_at": stamp}


def events_after(conn: sqlite3.Connection, after: int, limit: int = 50) -> list[dict]:
    rows = conn.execute("SELECT * FROM events WHERE id>? ORDER BY id LIMIT ?", (after, limit)).fetchall()
    return [dict(r) for r in rows]


def latest_event_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(id) AS top FROM events").fetchone()
    return (row["top"] or 0) if row is not None else 0


def recent_events(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def content_digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
