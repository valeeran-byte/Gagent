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

import httpx

from rag import state as rag_state

PROTOCOL = rag_state.PROTOCOL
ROOT = Path(__file__).resolve().parents[1]
SERVICE_MODULE = "rag.service"

READ_TIMEOUT = 120.0
SEARCH_TIMEOUT = float(os.environ.get("GAGENT_RAG_SEARCH_TIMEOUT", "5"))
EVENT_TIMEOUT = float(os.environ.get("GAGENT_RAG_EVENT_TIMEOUT", "5"))
START_TIMEOUT = float(os.environ.get("GAGENT_RAG_START_TIMEOUT", "60"))
CACHE_SECONDS = 5.0
LOCK_STALE_SECONDS = 120.0

# 服务侧的业务错误不该反复重启服务。
_NO_RESTART = {"bad_request", "not_found", "not_retryable", "search_error", "disabled"}

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
    """当前项目可用的服务实例；返回健康信息，不可用返回 None。"""
    info = service_info()
    if info is None or not info.get("port"):
        return None
    health = probe(_base(info["port"]))
    return None if health is None else {**info, "health": health}


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
        raise ServiceUnavailable(
            "文件服务不可用" if not start else
            f"文件服务未能启动或响应；日志：{rag_state.logs_dir() / 'service.log'}", "not_running")
    base = _base(info["port"])
    with _cache_lock:
        _cache["base"], _cache["checked"] = base, time.monotonic()
    return base


def _ensure_started() -> dict | None:
    """确保有且只有一个服务实例：抢启动锁，没抢到就等对方起来。"""
    deadline = time.monotonic() + START_TIMEOUT
    with _start_lock:
        info = running()
        if info is not None:
            return info
        holder = _acquire_lock()
        if holder:
            try:
                info = running()
                if info is None:
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
        time.sleep(0.25)
    return None


def _spawn() -> subprocess.Popen:
    """在后台启动服务进程：不弹控制台窗口，CLI 退出也不影响它继续处理任务。"""
    rag_state.ensure_dirs()
    handle = (rag_state.logs_dir() / "service.log").open("a", encoding="utf-8")
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        return subprocess.Popen([sys.executable, "-X", "utf8", "-m", SERVICE_MODULE, "--serve"],
                                stdin=subprocess.DEVNULL, stdout=handle, stderr=handle,
                                cwd=str(ROOT), close_fds=True, creationflags=flags,
                                start_new_session=os.name != "nt")
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
