"""RAG 文件服务的下载、验证、版本落盘与内容解析。

只在服务进程里使用；CLI 通过 file_service_client 走 HTTP，不直接碰这些数据。
资料目录固定为 rag_data/files/<kind>/，索引信息在 rag_state 的 SQLite 里（不再有 index.json）。

保存时机沿用既有口径：确认字节是有效 PDF/Excel 才落盘（验证页、损坏文件不进资料库），
之后的参数错误（工作表名写错、页码越界）不撤销这次保存。覆盖旧文件时先暂存，
写库失败就放回，避免出现"文件是新版、记录是旧版"。
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx

from rag import state as rag_state

DOWNLOAD_TIMEOUT = 20
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_DEFAULT_FILENAMES = {"pdf": "document.pdf", "excel": "workbook.xlsx"}
_EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm", ".xlsb"}
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_FILENAMES = {"CON", "PRN", "AUX", "NUL",
                       *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_WRITE_LOCK = threading.Lock()  # 服务是多线程的，同名文件与版本登记要串行


@dataclass
class Fetched:
    content: bytes
    url: str            # 最终地址（重定向后）
    source_url: str     # 最初请求的地址，作为资料身份
    file_name: str
    from_cache: bool
    remote_checked: bool
    verified_at: str | None  # 缓存命中时上次真正联网核验的时间
    headers: dict = field(default_factory=dict)  # 判定"验证页还是文件"要看真实 content-type


def _check_url(url: str) -> str:
    value = (url or "").strip()
    if urlsplit(value).scheme not in {"http", "https"}:
        raise ValueError("url 必须是完整的 http:// 或 https:// 地址")
    return value


def _header_filename(disposition: str) -> str:
    """从 Content-Disposition 取文件名，支持 RFC 5987 的 filename*。"""
    if not disposition:
        return ""
    star = re.search(r"filename\*\s*=\s*([^;]+)", disposition, re.IGNORECASE)
    if star:
        value = star.group(1).strip().strip('"')
        parts = value.split("'", 2)
        charset, encoded = (parts[0], parts[2]) if len(parts) == 3 else ("utf-8", value)
        try:
            return unquote(encoded, encoding=charset or "utf-8", errors="replace")
        except (LookupError, ValueError):
            return unquote(encoded)
    plain = re.search(r'filename\s*=\s*"([^"]*)"|filename\s*=\s*([^;]+)', disposition, re.IGNORECASE)
    if not plain:
        return ""
    return (plain.group(1) if plain.group(1) is not None else plain.group(2)).strip()


def download_name(headers, final_url: str, kind: str) -> str:
    """确定落盘文件名：Content-Disposition → 最终 URL 的文件名 → 默认名。"""
    name = _header_filename(headers.get("content-disposition", ""))
    if not name:
        name = unquote(urlsplit(str(final_url)).path)
    name = _INVALID_FILENAME_CHARS.sub("_", name.replace("\\", "/").rsplit("/", 1)[-1])
    name = re.sub(r"\s+", " ", name).strip().strip(".")
    stem, dot, suffix = name.rpartition(".")
    stem, suffix = (stem, f".{suffix}") if dot else (name, "")
    stem = stem.strip().strip(".")
    if not stem:
        return _DEFAULT_FILENAMES[kind]
    if stem.upper() in _RESERVED_FILENAMES:
        stem = f"_{stem}"
    if suffix.casefold() not in ({".pdf"} if kind == "pdf" else _EXCEL_SUFFIXES):
        suffix = ".pdf" if kind == "pdf" else ".xlsx"
    return f"{stem[:100]}{suffix.casefold()}"


def _looks_like_pdf(content: bytes) -> bool:
    return b"%PDF" in content[:1024]


def _looks_like_excel(content: bytes) -> bool:
    return content.startswith((b"PK\x03\x04", b"\xd0\xcf\x11\xe0"))


def _challenge_page(content: bytes, headers) -> None:
    if "html" not in headers.get("content-type", ""):
        return
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(content, "html.parser")
    title = soup.title.get_text(" ", strip=True).casefold() if soup.title else ""
    if title in {"making sure you're not a bot!", "just a moment...", "attention required! | cloudflare",
                 "verify you are human", "access denied"}:
        raise ValueError("网站返回人机验证或拒绝访问页面，未取得文件；请换来源")


def _local_copy(url: str, kind: str) -> Fetched | None:
    """同 URL 已在资料库时直接用本地字节；本地被改动过（摘要不符）就重新下载。"""
    conn = rag_state.connect()
    try:
        doc = conn.execute("SELECT * FROM documents WHERE source_url=? AND kind=?",
                           (rag_state.canonical_url(url), kind)).fetchone()
        if doc is None:
            return None
        row = conn.execute("SELECT v.* FROM versions v WHERE v.doc_id=? ORDER BY v.id DESC LIMIT 1",
                           (doc["doc_id"],)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    path = Path(row["local_path"] or "")
    if not path.is_file():
        return None
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != row["sha256"]:
        return None
    if kind == "pdf" and not _looks_like_pdf(content):
        return None
    return Fetched(content=content, url=row["final_url"] or doc["source_url"],
                   source_url=doc["source_url"], file_name=row["file_name"], from_cache=True,
                   remote_checked=False, verified_at=row["created_at"])


def fetch(url: str, kind: str, refresh: bool = False) -> Fetched:
    """取文件字节：默认优先复用本地副本，refresh=True 才强制回源核验。

    缓存命中不能声称刚刚核验过远端，因此结果里带 remote_checked/verified_at。
    """
    target = _check_url(url)
    if not refresh:
        cached = _local_copy(target, kind)
        if cached is not None:
            return cached
    source = rag_state.canonical_url(target)
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = httpx.get(target, follow_redirects=True, timeout=DOWNLOAD_TIMEOUT,
                                 headers={"User-Agent": "Mozilla/5.0"})
            final = str(response.url)
            content = response.content
            name = download_name(response.headers, final, kind)
            if kind == "pdf" and response.status_code == 404 and _looks_like_pdf(content):
                # 有些下载端点（例如 Logitech 的 dam 路径）会把 PDF 的响应体也标成 404。
                return Fetched(content, final, source, name, False, True, None, dict(response.headers))
            if response.status_code in _TRANSIENT_STATUS and attempt == 0:
                last_error = httpx.HTTPStatusError("transient", request=response.request, response=response)
                continue
            response.raise_for_status()
            return Fetched(content, final, source, name, False, True, None, dict(response.headers))
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            transient = isinstance(exc, httpx.RequestError) or exc.response.status_code in _TRANSIENT_STATUS
            if attempt or not transient:
                raise
            last_error = exc
    raise last_error or RuntimeError("下载失败")


def open_pdf(content: bytes, headers=None):
    """确认字节是可用 PDF；验证页和损坏文件在这里就被挡掉。"""
    from pypdf import PdfReader
    if not _looks_like_pdf(content):
        _challenge_page(content, headers or {})
        raise ValueError("响应不是 PDF 文件，可能是下载错误页")
    reader = PdfReader(BytesIO(content))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF 已加密，需要解密后的文件")
    try:
        # 页树坏掉的 PDF 在打开时不报错，取页时才抛 pypdf 内部异常；这里先探一次，
        # 让调用方拿到一句"文件有问题"而不是 500。
        if len(reader.pages):
            reader.pages[0].extract_text()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"PDF 结构损坏，无法读取页面：{type(exc).__name__}") from exc
    return reader


def open_workbook(content: bytes):
    import pandas as pd
    try:
        return pd.ExcelFile(BytesIO(content))
    except Exception as exc:
        raise ValueError(f"响应不是可解析的 Excel 工作簿：{type(exc).__name__}") from exc


def extract_pages(local_path: str) -> list[str]:
    """按物理页提取整份 PDF 的文字；空字符串表示该页没有可提取文字（扫描页）。"""
    reader = open_pdf(Path(local_path).read_bytes())
    return [(page.extract_text() or "").strip() for page in reader.pages]


def read_sheet_shape(local_path: str, sheet_name: str | None) -> tuple[list[str], int, int]:
    import pandas as pd
    workbook = open_workbook(Path(local_path).read_bytes())
    names = list(workbook.sheet_names)
    selected = sheet_name if sheet_name is not None else names[0]
    if selected not in names:
        raise ValueError(f"工作表不存在，可选值：{names}")
    frame = workbook.parse(selected, header=None, dtype=object, keep_default_na=False)
    return names, int(frame.shape[0]), int(frame.shape[1])


def _unique_path(folder: Path, name: str, keep: Path | None = None) -> Path:
    """不同来源撞上同一个文件名时另存为 name-1、name-2，不覆盖已有资料。"""
    path = folder / name
    if not path.exists() or (keep is not None and path == keep):
        return path
    for index in range(1, 1000):
        candidate = folder / f"{path.stem}-{index}{path.suffix}"
        if not candidate.exists():
            return candidate
    raise OSError(f"同名文件过多：{name}")


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f"{path.name}.part")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    except OSError:
        _quiet_unlink(temporary)
        raise


def _quiet_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _restore(backup: Path, path: Path) -> None:
    try:
        os.replace(backup, path)
    except OSError:
        pass


def register(fetched: Fetched, kind: str, *, summary: str | None = None,
             page_count: int | None = None, empty_pages: list[int] | None = None,
             sheet_names: list[str] | None = None) -> dict:
    """保存文件版本并确保入库任务存在（同一版本复用同一任务）。

    返回 saved=False 时只说明资料库这一侧失败，读取本身的内容仍然可用，调用方不能把它
    说成"已接受入库任务"。
    """
    digest = hashlib.sha256(fetched.content).hexdigest()
    folder = rag_state.files_dir() / kind
    with _WRITE_LOCK:
        conn = rag_state.connect()
        try:
            doc = rag_state.get_or_create_document(conn, fetched.source_url, kind, fetched.file_name)
            known = rag_state.version_by_sha(conn, doc["doc_id"], digest)
            known_path = Path(known["local_path"]) if known is not None else None
            reused = known is not None and _bytes_match(known_path, digest)
            updated = known is None and doc["active_version"] is not None
            backup: Path | None = None
            try:
                folder.mkdir(parents=True, exist_ok=True)
                if reused:
                    target = known_path
                elif known_path is not None and known_path.is_file():
                    # 字节变了：旧版本先暂存，登记失败时放回去，不留半新半旧。
                    target = known_path
                    backup = target.with_name(f"{target.name}.bak")
                    os.replace(target, backup)
                    _atomic_write(target, fetched.content)
                else:
                    target = known_path if known_path is not None else _unique_path(folder, fetched.file_name)
                    _atomic_write(target, fetched.content)  # 本地副本被删：在原路径重建
            except OSError as exc:
                if backup is not None:
                    _restore(backup, target)
                return _register_failure(digest, f"{type(exc).__name__}: {exc}")
            try:
                version, _created = rag_state.add_version(
                    conn, doc_id=doc["doc_id"], sha256=digest, kind=kind,
                    local_path=str(target), size=len(fetched.content), file_name=fetched.file_name,
                    final_url=fetched.url, summary=summary, page_count=page_count,
                    empty_pages=empty_pages, sheet_names=sheet_names)
                job, _ = rag_state.ensure_job(conn, doc["doc_id"], version["id"], kind)
            except Exception as exc:
                if backup is not None:
                    _restore(backup, target)  # 旧文件和旧登记一起保留
                elif known is None:
                    _quiet_unlink(target)
                return _register_failure(digest, f"资料库登记失败：{type(exc).__name__}: {exc}")
            if backup is not None:
                _quiet_unlink(backup)
            return {"saved": True, "document_id": doc["doc_id"], "version": digest,
                    "version_id": version["id"], "local_path": str(target), "size": len(fetched.content),
                    "file_name": version["file_name"], "summary": version["summary"],
                    "reused": reused, "updated": updated,
                    "ingestion": {"job_id": job["job_id"], "status": job["status"],
                                  "attempts": job["attempts"], "round": job["round"],
                                  "last_error": job["last_error"]}}
        finally:
            conn.close()


def _bytes_match(path: Path | None, digest: str) -> bool:
    """是否复用本地文件按实际字节判断，不能只信登记里的摘要。"""
    try:
        return path is not None and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
    except OSError:
        return False


def _register_failure(digest: str, error: str) -> dict:
    return {"saved": False, "document_id": None, "version": digest, "local_path": None,
            "file_name": None, "summary": None, "reused": False, "updated": False,
            "ingestion": {"job_id": None, "status": "submit_failed", "error": error}}


def _read_fields(fetched: Fetched, registered: dict, extra: dict) -> dict:
    """统一的读取返回：读取结果 + 版本信息 + 入库状态。"""
    return {**extra, "read_ok": True, "url": fetched.url, "source_url": fetched.source_url,
            "document_id": registered.get("document_id"), "version": registered.get("version"),
            "file_name": registered.get("file_name"), "local_path": registered.get("local_path"),
            "artifact_saved": bool(registered.get("saved")),
            "ingestion": registered.get("ingestion") or {"status": "submit_failed"},
            "from_cache": fetched.from_cache, "remote_checked": fetched.remote_checked,
            "verified_at": fetched.verified_at,
            "version_note": ("本地副本，本次未重新核验远端" if fetched.from_cache
                             else "本次已重新核验远端内容")}


def _page_slice(text: str, start_char: int, max_chars: int) -> dict:
    if start_char < 0 or not 1 <= max_chars <= 50000:
        raise ValueError("start_char >= 0；max_chars 必须为 1 到 50000")
    if start_char > len(text):
        raise ValueError(f"start_char 超出内容长度 {len(text)}")
    end = min(start_char + max_chars, len(text))
    return {"content": text[start_char:end], "total_chars": len(text), "start_char": start_char,
            "truncated": end < len(text), "next_start_char": end if end < len(text) else None}


def read_pdf(url: str, max_pages: int = 5, start_page: int = 1, start_char: int = 0,
             max_chars: int = 20000, refresh: bool = False) -> dict:
    """下载或复用 PDF，读取指定页内容，并确保整份文件的入库任务存在。

    这里只返回调用方要的那几页；全文切块由服务端后台任务重新读取本地文件完成。
    """
    if start_page < 1 or not 1 <= max_pages <= 50 or start_char < 0 or not 1 <= max_chars <= 50000:
        raise ValueError("页码从 1 开始；max_pages 为 1 到 50；字符偏移 >= 0；max_chars 为 1 到 50000")
    fetched = fetch(url, "pdf", refresh)
    reader = open_pdf(fetched.content, fetched.headers)
    count = len(reader.pages)
    if start_page > count:
        raise ValueError(f"start_page 超出总页数 {count}")
    pages, remaining = [], max_chars
    next_page, next_char = None, None
    for index in range(start_page - 1, min(count, start_page - 1 + max_pages)):
        text = (reader.pages[index].extract_text() or "").strip()
        offset = start_char if index == start_page - 1 else 0
        part = _page_slice(text, offset, remaining)
        pages.append({"page": index + 1, "text": part["content"], "start_char": offset,
                      "total_chars": len(text), "needs_ocr": not bool(text)})
        remaining -= len(part["content"])
        if part["truncated"]:
            next_page, next_char = index + 1, part["next_start_char"]
            break
        next_page, next_char = (index + 2, 0) if index + 1 < count else (None, None)
        if remaining == 0:
            break
    registered = register(fetched, "pdf", page_count=count)
    result = _read_fields(fetched, registered, {
        "page_count": count, "pages": pages,
        "needs_ocr": any(p["needs_ocr"] for p in pages), "truncated": next_page is not None,
        "next_start_page": next_page, "next_start_char": next_char})
    if not registered.get("saved"):
        result["artifact_error"] = registered["ingestion"].get("error") or "未知原因"
    return result


def find_in_pdf(url: str, query: str, start_page: int = 1, max_matches: int = 5,
                context_chars: int = 600, refresh: bool = False) -> dict:
    """在 PDF 全文定位关键词所在页；下载、保存与缓存和 read_pdf 共用同一套逻辑。"""
    if not query.strip() or start_page < 1 or not 1 <= max_matches <= 20 or not 50 <= context_chars <= 3000:
        raise ValueError("query 非空；页码从1开始；max_matches 为1到20；context_chars 为50到3000")
    fetched = fetch(url, "pdf", refresh)
    reader = open_pdf(fetched.content, fetched.headers)
    if start_page > len(reader.pages):
        raise ValueError("起始页超出 PDF 页数")
    page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
    matches, empty_pages = [], []
    next_page = None
    for index in range(start_page - 1, len(page_texts)):
        text = page_texts[index]
        if not text:
            empty_pages.append(index + 1)
        hit = re.search(re.escape(query), text, re.IGNORECASE)
        if hit:
            matches.append({"page": index + 1, "match_char": hit.start(),
                            "text": text[max(0, hit.start() - context_chars):hit.end() + context_chars]})
        if len(matches) == max_matches:
            next_page = index + 2 if index + 1 < len(page_texts) else None
            break
    registered = register(fetched, "pdf", page_count=len(page_texts),
                          empty_pages=[i + 1 for i, t in enumerate(page_texts) if not t])
    result = _read_fields(fetched, registered, {
        "query": query, "page_count": len(page_texts), "matches": matches,
        "next_start_page": next_page, "empty_pages": empty_pages, "needs_ocr": bool(empty_pages)})
    if not registered.get("saved"):
        result["artifact_error"] = registered["ingestion"].get("error") or "未知原因"
    return result


def read_excel(url: str, summary: str, sheet_name: str | None = None, max_rows: int = 100,
               max_columns: int = 30, start_row: int = 0, start_column: int = 0,
               preview: bool = True, preview_rows: int = 5, refresh: bool = False) -> dict:
    """下载或复用 Excel，返回结构预览，并确保 summary 的入库任务存在。

    summary 必须非空白：本版本不调用摘要模型，检索侧只用调用方给出的这段描述。
    同一版本重复读取时保留首次接受的 summary，不因措辞变化重建索引。
    """
    import pandas as pd
    text = str(summary or "")
    if not text.strip():
        raise ValueError("summary 不能为空：请根据网页介绍、文件名称及已看到的信息描述整个工作簿的主题和用途")
    if start_row < 0 or start_column < 0 or not 1 <= max_rows <= 1000 or not 1 <= max_columns <= 100:
        raise ValueError("偏移 >= 0；max_rows 为 1 到 1000；max_columns 为 1 到 100")
    if not 1 <= preview_rows <= 20:
        raise ValueError("preview_rows 为 1 到 20")
    fetched = fetch(url, "excel", refresh)
    workbook = open_workbook(fetched.content)
    names = list(workbook.sheet_names)
    selected = sheet_name if sheet_name is not None else names[0]
    if selected not in names:
        raise ValueError(f"工作表不存在，可选值：{names}")
    frame = workbook.parse(selected, header=None, dtype=object, keep_default_na=False)
    total_rows, total_columns = frame.shape
    if start_row > total_rows or start_column > total_columns:
        raise ValueError(f"偏移超出工作表范围：{total_rows} 行，{total_columns} 列")
    registered = register(fetched, "excel", summary=text.strip(), sheet_names=names)
    result = _read_fields(fetched, registered, {
        "sheet_names": names, "selected_sheet": selected, "total_rows": int(total_rows),
        "total_columns": int(total_columns), "start_row": start_row, "start_column": start_column,
        "summary": registered.get("summary") or text.strip(),
        "summary_kept": bool(registered.get("summary")) and registered["summary"] != text.strip()})
    if not registered.get("saved"):
        result["artifact_error"] = registered["ingestion"].get("error") or "未知原因"
    frame_block = frame.iloc[start_row:start_row + (preview_rows if preview else max_rows),
                             start_column:start_column + max_columns]
    if preview:
        result.update(mode="preview", columns=_columns(frame, start_column, max_columns),
                      sample_rows=_json_rows(frame_block), sample_row_count=int(frame_block.shape[0]),
                      note="完整数据可用 run_python 的代码读取 local_path 计算；需要原始单元格时用 preview=false 分块读取。")
        return result
    next_row = start_row + max_rows if start_row + max_rows < total_rows else None
    next_column = start_column + max_columns if start_column + max_columns < total_columns else None
    result.update(mode="block", rows=_json_rows(frame_block),
                  truncated=next_row is not None or next_column is not None,
                  next_start_row=next_row, next_start_column=next_column)
    return result


def _cell(value):
    """把单元格转成可 JSON 序列化的标量：NaN/NaT 归 None，时间按 ISO 文本。"""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value if not isinstance(value, float) or value == value else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _json_rows(block) -> list:
    return [[_cell(value) for value in row] for row in block.itertuples(index=False, name=None)]


def _columns(frame, start_column: int, max_columns: int) -> list:
    """用首行当表头给出列名，空单元格补 column_N，重名加序号。"""
    if frame.shape[0] == 0:
        return []
    headers, used = [], {}
    for index in range(start_column, min(start_column + max_columns, frame.shape[1])):
        text = str(_cell(frame.iat[0, index]) or "").strip() or f"column_{index + 1}"
        used[text] = used.get(text, 0) + 1
        headers.append(text if used[text] == 1 else f"{text}_{used[text]}")
    return headers
