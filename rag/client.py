"""CLI 侧的文件服务客户端：发现并按需拉起本机服务，再调用接口。

CLI 进程不直接读写文件版本、任务和向量库，全部通过这里，
保证同一项目只有一个进程在管这些数据；服务进程独立于 CLI 生命周期。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote

import httpx

from rag import state as rag_state

PROTOCOL = rag_state.PROTOCOL
ROOT = Path(__file__).resolve().parents[1]
SERVICE_MODULE = "rag.service"

READ_TIMEOUT = 120.0
SEARCH_TIMEOUT = float(os.environ.get("GAGENT_RAG_SEARCH_TIMEOUT", "5"))
EVENT_TIMEOUT = float(os.environ.get("GAGENT_RAG_EVENT_TIMEOUT", "5"))
# 跨会话回忆可能要现算摘要，比普通检索慢得多，不能沿用 5 秒。
RECALL_TIMEOUT = float(os.environ.get("GAGENT_RAG_RECALL_TIMEOUT", "90"))
START_TIMEOUT = float(os.environ.get("GAGENT_RAG_START_TIMEOUT", "60"))
CACHE_SECONDS = 5.0
LOCK_STALE_SECONDS = 120.0

# 服务侧的业务错误不该反复重启服务。
_NO_RESTART = {"bad_request", "not_found", "not_retryable", "search_error", "disabled",
               "sessions_mismatch"}

_cache = {"base": None, "checked": 0.0}
_cache_lock = threading.Lock()
_start_lock = threading.Lock()


class ServiceError(RuntimeError):
    """服务明确答复的失败（参数不合法、文件读不出来等），重试没有意义。"""

    def __init__(self, message: str, kind: str = "error") -> None:
        super().__init__(message)
        self.kind = kind


class ServiceUnavailable(ServiceError):
    """服务没起来或中途退出。"""


def _base(port) -> str:
    return f"http://127.0.0.1:{port}"


def _sessions_dir_text() -> str:
    """本次运行实际使用的会话目录（CLI 的 --sessions-dir 由 cli._use_sessions_dir 落在这里）。

    运行时读取模块属性，不缓存：同一进程里换目录（测试、隔离实例）必须立刻生效。
    """
    try:
        import session_store
        return str(session_store.SESSIONS_DIR)
    except Exception:  # pragma: no cover - 正常情况下 session_store 一定可用
        return ""


def _canonical_dir(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return os.path.normcase(os.path.abspath(text))


def _sessions_dir_matches(*sources: dict) -> bool:
    """服务上报的会话目录是否就是本次运行要用的那一个。

    旧版服务没有这个字段，无法确认时必须当成不一致（宁可拒绝，也不能把 CLI 的
    自定义会话目录交给一个正在索引默认 sessions/ 的旧服务）。
    """
    wanted = _canonical_dir(_sessions_dir_text())
    if not wanted:  # 取不到自己的目录时退回旧行为，不误伤
        return True
    return any(_canonical_dir((source or {}).get("sessions_dir")) == wanted for source in sources)


def _sessions_mismatch_error(info: dict) -> "ServiceUnavailable":
    """目录不一致的明确错误：说明双方目录、为什么不能复用、以及怎么恢复。"""
    health = info.get("health") or {}
    reported = str(health.get("sessions_dir") or info.get("sessions_dir") or "").strip()
    theirs = reported or "（旧版服务，未上报会话目录）"
    pid = info.get("pid") or "未知"
    return ServiceUnavailable(
        f"本机已有文件服务在运行（pid {pid}），但它的会话目录是 {theirs}，"
        f"与本次运行的 {_sessions_dir_text()} 不一致：对话记忆不能跨目录混用。"
        f"本次不复用该服务，也不会另起一个（两个进程同时写同一份数据会互相破坏）。"
        f"请先停止旧服务（Windows：taskkill /F /PID {pid}）或改用相同的 --sessions-dir。",
        "sessions_mismatch")


def service_info() -> dict | None:
    path = rag_state.service_info_path()
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(info, dict) or info.get("service_id") != rag_state.service_id():
        return None  # 别的项目留下的文件或内容损坏，一律按没有服务处理
    return info


def probe(base: str, timeout: float = 1.5) -> dict | None:
    try:
        data = httpx.get(base + "/health", timeout=timeout).json()
    except Exception:
        return None
    if not isinstance(data, dict) or not data.get("ok"):
        return None
    if data.get("protocol") != PROTOCOL or data.get("service_id") != rag_state.service_id():
        return None  # 端口被别的进程占用或协议不兼容，不能拿它当自己的服务
    return data


def running() -> dict | None:
    """当前项目可用的服务实例；不可用或会话目录不是本次要用的，返回 None。"""
    info = service_info()
    if info is None or not info.get("port"):
        return None
    health = probe(_base(info["port"]))
    if health is None:
        return None
    if not _sessions_dir_matches(info, health):
        return None  # 有服务但用的不是本次的会话目录：不复用，也不许再起一个
    return {**info, "health": health}


def foreign_service() -> dict | None:
    """当前项目有服务在跑，但会话目录与本次运行不一致；没有就返回 None。

    这是"先按默认目录起过一次服务、再用 --sessions-dir X 启动"的正常场景：
    服务在 CLI 退出后继续运行，必须在连接前就把它认出来。
    """
    info = service_info()
    if info is None or not info.get("port"):
        return None
    health = probe(_base(info["port"]))
    if health is None or _sessions_dir_matches(info, health):
        return None
    return {**info, "health": health}


def _refuse_foreign_service() -> None:
    """有服务在用别的会话目录：既不能复用，也不能再起一个（会同时写同一份数据）。"""
    foreign = foreign_service()
    if foreign is not None:
        raise _sessions_mismatch_error(foreign)


def _invalidate() -> None:
    with _cache_lock:
        _cache["base"], _cache["checked"] = None, 0.0


def enabled() -> bool:
    """GAGENT_RAG_DISABLE=1 时完全不碰文件服务：单测不该为此拉起后台进程。"""
    return os.environ.get("GAGENT_RAG_DISABLE", "").strip().lower() not in {"1", "true", "yes", "on"}


def endpoint(start: bool = True) -> str:
    """可用的服务地址；缓存几秒，避免每次工具调用都探活。

    start=False 只探现已运行的实例，不拉起服务：CLI 的事件轮询不该在每次启动时
    都开一个服务进程。
    """
    if not enabled():
        raise ServiceUnavailable("本机文件服务已关闭（GAGENT_RAG_DISABLE）", "disabled")
    now = time.monotonic()
    with _cache_lock:
        if _cache["base"] and now - _cache["checked"] < CACHE_SECONDS:
            return _cache["base"]
    info = running()
    if info is None and start:
        info = _ensure_started()
    if info is None:
        foreign = foreign_service()
        if foreign is not None:
            raise _sessions_mismatch_error(foreign)
        raise ServiceUnavailable(
            "文件服务不可用" if not start else
            f"文件服务未能启动或响应；日志：{rag_state.logs_dir() / 'service.log'}", "not_running")
    base = _base(info["port"])
    with _cache_lock:
        _cache["base"], _cache["checked"] = base, time.monotonic()
    return base


def _ensure_started() -> dict | None:
    """确保有且只有一个服务实例：抢启动锁，没抢到就等对方起来。

    已经有服务在跑、但会话目录不是本次要用的，直接拒绝：复用会把 CLI 的对话记忆
    写进另一个目录，另起一个又会让两个进程同时写同一份资料库。
    """
    deadline = time.monotonic() + START_TIMEOUT
    with _start_lock:
        info = running()
        if info is not None:
            return info
        _refuse_foreign_service()
        holder = _acquire_lock()
        if holder:
            try:
                info = running()
                if info is None:
                    _refuse_foreign_service()
                    _spawn()
                else:
                    return info
            finally:
                _release_lock()
        return _wait_until_ready(deadline)


def _acquire_lock() -> bool:
    path = rag_state.service_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in (0, 1):
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if attempt:
                return False
            try:
                if time.time() - path.stat().st_mtime > LOCK_STALE_SECONDS:
                    path.unlink(missing_ok=True)  # 上次启动方已经死了，锁不能永久挡路
                    continue
            except OSError:
                return False
            return False
        except OSError:
            return False
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "at": time.time()}))
        return True
    return False


def _release_lock() -> None:
    try:
        rag_state.service_lock_path().unlink(missing_ok=True)
    except OSError:
        pass


def _wait_until_ready(deadline: float) -> dict | None:
    while time.monotonic() < deadline:
        info = running()
        if info is not None:
            return info
        _refuse_foreign_service()  # 等的是别人的服务且目录不对：立刻说清楚，别干等
        time.sleep(0.25)
    return None


def _spawn() -> subprocess.Popen:
    """在后台启动服务进程：不弹控制台窗口，CLI 退出也不影响它继续处理任务。

    显式带上 GAGENT_SESSIONS_DIR：CLI 可能用 --sessions-dir 或隔离的会话目录，
    服务进程是另一个进程，只有环境变量传得过去（否则它会去索引默认的 sessions/）。
    """
    rag_state.ensure_dirs()
    handle = (rag_state.logs_dir() / "service.log").open("a", encoding="utf-8")
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        import session_store
        environment["GAGENT_SESSIONS_DIR"] = str(session_store.SESSIONS_DIR)
    except Exception:
        pass
    try:
        return subprocess.Popen([sys.executable, "-X", "utf8", "-m", SERVICE_MODULE, "--serve"],
                                stdin=subprocess.DEVNULL, stdout=handle, stderr=handle,
                                cwd=str(ROOT), close_fds=True, creationflags=flags,
                                start_new_session=os.name != "nt", env=environment)
    finally:
        handle.close()  # 子进程已持有该 fd，父进程这边要关掉，否则日志文件一直开着


def _decode(response: httpx.Response) -> dict:
    try:
        body = response.json()
    except ValueError:
        raise ServiceError(f"文件服务返回了无法解析的内容（HTTP {response.status_code}）", "bad_response")
    if response.status_code < 400:
        if isinstance(body, dict):
            return body
        raise ServiceError("文件服务返回结构异常", "bad_response")
    kind = str(body.get("error") or "error") if isinstance(body, dict) else "error"
    message = str(body.get("message") or f"文件服务返回 HTTP {response.status_code}") if isinstance(body, dict) \
        else f"文件服务返回 HTTP {response.status_code}"
    error = ServiceError(message, kind)
    raise error


def _request(method: str, path: str, *, payload: dict | None = None, params: dict | None = None,
             timeout: float = READ_TIMEOUT, start: bool = True, attempts: int = 2) -> dict:
    last: Exception | None = None
    for _ in range(attempts):
        try:
            base = endpoint(start=start)
            response = httpx.request(method, base + path, json=payload, params=params, timeout=timeout)
            return _decode(response)
        except ServiceError as exc:
            if exc.kind in _NO_RESTART:
                raise
            _invalidate()
            last = exc
        except httpx.RequestError as exc:
            _invalidate()
            last = ServiceUnavailable(f"文件服务无响应：{type(exc).__name__}", "unavailable")
    raise last or ServiceUnavailable("文件服务不可用", "unavailable")


def health(timeout: float = 3.0) -> dict | None:
    info = service_info()
    if info is None:
        return None
    return probe(_base(info["port"]), timeout=timeout)


# ---------------------------------------------------------------- 事件消费位置

def notices_path() -> Path:
    return rag_state.client_state_path()


def read_cursor() -> int:
    """上次消费到的事件 id：关闭期间完成的任务，下次打开还能补提示。"""
    try:
        data = json.loads(notices_path().read_text(encoding="utf-8"))
        return max(0, int(data.get("cursor", 0)))
    except (OSError, ValueError, TypeError, AttributeError):
        return 0


def write_cursor(cursor: int) -> None:
    try:
        path = notices_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        temporary.write_text(json.dumps({"cursor": int(cursor), "updated_at": time.time()}),
                             encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        pass  # 记不住位置只会让下次多弹一次，不该影响问答


# ---------------------------------------------------------------- 接口封装

def read_pdf(url: str, **params) -> dict:
    return _request("POST", "/v1/read/pdf", payload={"url": url, **params})


def read_excel(url: str, summary: str, **params) -> dict:
    return _request("POST", "/v1/read/excel", payload={"url": url, "summary": summary, **params})


def find_in_pdf(url: str, query: str, **params) -> dict:
    return _request("POST", "/v1/find/pdf", payload={"url": url, "query": query, **params})


def search(query: str, **options) -> dict:
    """每轮提问前的检索：任何故障都只让本轮不带 RAG 内容，不阻塞问答。

    返回 {"results": [...], "reason": "..."}，reason 只在没有结果时说明原因。
    """
    text = (query or "").strip()
    if not text:
        return {"results": [], "reason": "空问题"}
    started = time.monotonic()
    try:
        data = _request("POST", "/v1/search", payload={"query": text, **options},
                        timeout=SEARCH_TIMEOUT)
    except Exception as exc:
        return {"results": [], "reason": f"{type(exc).__name__}: {exc}",
                "elapsed": round(time.monotonic() - started, 2)}
    elapsed = round(time.monotonic() - started, 2)
    results = data.get("results")
    if not isinstance(results, list):
        return {"results": [], "reason": "服务返回结构异常", "elapsed": elapsed}
    return {**data, "results": results, "elapsed": elapsed}


def events_after(cursor: int = 0, limit: int = 50, start: bool = False) -> dict:
    """取新事件。默认不为了弹提示就把服务拉起来：没有服务时本来也没有新事件。"""
    return _request("GET", "/v1/events", params={"after": cursor, "limit": limit},
                    timeout=EVENT_TIMEOUT, start=start)


def documents(limit: int = 50, start: bool = False) -> dict:
    """各文件版本的索引统计；默认不为看状态而拉起服务。"""
    return _request("GET", "/v1/documents", params={"limit": limit}, timeout=EVENT_TIMEOUT, start=start)


def state(limit: int = 20, start: bool = True) -> dict:
    return _request("GET", "/v1/state", params={"limit": limit}, timeout=EVENT_TIMEOUT, start=start)


def job(job_id: str) -> dict:
    return _request("GET", f"/v1/jobs/{job_id}", timeout=EVENT_TIMEOUT)


def retry_job(job_id: str) -> dict:
    return _request("POST", f"/v1/jobs/{job_id}/retry", timeout=EVENT_TIMEOUT)


# ---------------------------------------------------------------- 对话记忆（跨会话回忆）

def _conversation_path(session_id) -> str:
    return f"/v1/conversations/{quote(str(session_id), safe='')}"


def conversations_sync(session_id: str | None = None, *, all: bool = False, start: bool = True) -> dict:
    """让服务把会话的问答轮次补进对话索引。

    all=True 同步全部会话（首次启用时把已有 session 全部入队），此时不带 session_id；
    否则只同步指定的一个会话。异常照常抛出，由后台调用方兜住，不在这里吞掉。
    """
    payload = {"session_id": None if all else session_id, "all": bool(all)}
    return _request("POST", "/v1/conversations/sync", payload=payload, timeout=READ_TIMEOUT, start=start)


def conversations_search(query: str, *, exclude_sessions: list[str] | None = None,
                         top_chunks: int = 20, max_sessions: int = 3, start: bool = True) -> dict:
    """按语义检索历史对话片段（含候选会话）。

    与 search 一样不阻塞问答：空问题不请求服务，服务故障降级成 status=error，
    由工具层决定怎么如实回答。
    """
    text = (query or "").strip()
    if not text:
        return {"status": "invalid", "reason": "空问题", "query": "", "hits": [], "candidates": []}
    payload = {"query": text, "exclude_sessions": list(exclude_sessions or []),
               "top_chunks": int(top_chunks), "max_sessions": int(max_sessions)}
    try:
        return _request("POST", "/v1/conversations/search", payload=payload, timeout=READ_TIMEOUT, start=start)
    except Exception as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}",
                "query": text, "hits": [], "candidates": []}


def conversation_history(session_id: str, *, start: bool = True) -> dict:
    """一个会话的完整问答、索引状态与摘要记录；管理、排错用。"""
    return _request("GET", _conversation_path(session_id), timeout=READ_TIMEOUT, start=start)


def conversation_summary(session_id: str, *, revision: str | None = None, start: bool = True) -> dict:
    """取会话的摘要记录；revision 用来确认摘要对应的是哪个源版本。"""
    params = {"revision": revision} if revision else None
    return _request("GET", _conversation_path(session_id) + "/summary", params=params,
                    timeout=READ_TIMEOUT, start=start)


def put_conversation_summary(session_id, *, source_revision, summary_schema_version, summary_json,
                             generated_at, start=True) -> dict:
    """写回摘要：调用方负责把生成好的摘要按源版本落库，版本不匹配由服务拒绝。"""
    payload = {"source_revision": source_revision, "summary_schema_version": summary_schema_version,
               "summary_json": summary_json, "generated_at": generated_at}
    return _request("PUT", _conversation_path(session_id) + "/summary", payload=payload,
                    timeout=READ_TIMEOUT, start=start)


def recall_conversation(query: str, *, session_id: str | None = None, exclude_session: str | None = None,
                        include_summary: bool = True, api_settings: dict | None = None,
                        start: bool = True) -> dict:
    """跨会话回忆：检索历史对话，并让服务按需整理成摘要回答。

    任何故障都不阻塞当前问答：空问题返回 status=invalid，检索失败或服务不可用返回
    status=error，由工具层如实回答“没找到/服务不可用”。
    api_settings 只在请求体里发给本机服务，不写日志也不落盘。
    """
    text = (query or "").strip()
    if not text:
        return {"status": "invalid", "reason": "空问题", "evidence": []}
    try:
        if api_settings is None:
            from api_setup import active_api_settings
            api_settings = active_api_settings()
        payload = {"query": text, "session_id": session_id, "exclude_session": exclude_session,
                   "include_summary": bool(include_summary),
                   "api_settings": dict(api_settings) if isinstance(api_settings, dict) else {}}
        # 现算摘要可能很久，只试一次：重试只会让用户多等一轮超时。
        return _request("POST", "/v1/conversations/recall", payload=payload,
                        timeout=RECALL_TIMEOUT, start=start, attempts=1)
    except Exception as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}", "evidence": []}
