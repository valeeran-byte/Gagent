"""G_agent 的工具层。导入不创建模型。

PDF/Excel 工具是本地文件服务的接口包装：下载、验证、保存、解析和入库任务提交都在
file_service 进程里完成，本模块只负责调用和把结果交给模型。文件版本、向量库和任务队列
都归服务管，CLI 不再直接写 downloads/。

Excel 和网页表格默认只向模型返回结构预览（工作表、列名、行列数、少量样例），完整数据由
主模型可调用 run_python，让代码直接读取 local_path 计算；网页表格另存为 downloads/tables/*.csv，不进资料索引。
"""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime
from functools import wraps
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Literal
from urllib.parse import unquote, urljoin, urlsplit
import uuid

from bs4 import BeautifulSoup
from ddgs import DDGS
import httpx
from langchain_core.tools import ToolException, tool

from rag import client as rag_client


_resources = ContextVar("agent_resources", default=None)

# 网页表格的 CSV 落盘位置（不属于资料索引）。测试可替换该常量以隔离落盘。
DOWNLOADS_DIR = Path(__file__).resolve().parent / "downloads"
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_session_lock = threading.Lock()
_FIRST_WEB_SEARCH = ("__agent_internal__", "first_web_search")


@contextmanager
def tool_session():
    """每次运行使用同一份下载内容，保证分页稳定；运行之间不共享缓存。"""
    token = _resources.set({})
    try:
        yield
    finally:
        _resources.reset(token)


def _tool_error(exc: Exception) -> str:
    if isinstance(exc, rag_client.ServiceError):
        return str(exc)  # 服务已经把原因说成可执行的一句话，再加类型前缀只是噪声
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        hint = {401: "认证失败", 403: "拒绝访问", 404: "资源不存在，不要重复请求或拼接附件路径",
                429: "服务限流"}.get(status, "远程服务请求失败")
        return f"HTTP {status}: {hint}。"
    if isinstance(exc, httpx.RequestError):
        return "网络连接失败或超时；检查网络/代理，勿连续重复相同请求。"
    return f"{type(exc).__name__}: {exc}"


def reliable_tool(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            message = _tool_error(exc)
            if func.__name__ == "visit_webpage" and isinstance(
                    exc, (httpx.HTTPStatusError, httpx.RequestError)):
                message += " 请从搜索结果中更换目标 URL，不要再次访问这个地址。"
            raise ToolException(message) from exc
    result = tool(wrapped)
    result.handle_tool_error = True
    # pydantic 原文会回显整个入参，长参数既费 token 又不利于模型纠正；只回一句可执行的提示。
    result.handle_validation_error = lambda exc: (
        f"参数错误：{func.__name__} 的入参不符合 schema，请改正后重发，不要重复同样的参数。"
        f"详情：{str(exc)[:300]}")
    return result


def _get_resource(url: str, params: dict | None = None) -> httpx.Response:
    if urlsplit(url.strip()).scheme not in {"http", "https"}:
        raise ValueError("url 必须是完整的 http:// 或 https:// 地址")
    cache = _resources.get()
    key = (url.strip(), json.dumps(params, sort_keys=True))
    if cache is not None and key in cache:
        return cache[key]
    for attempt in range(2):
        try:
            response = httpx.get(url.strip(), params=params, follow_redirects=True,
                                 timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            if cache is not None:
                cache[key] = response
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            transient = isinstance(exc, httpx.RequestError) or exc.response.status_code in {429, 500, 502, 503, 504}
            if attempt or not transient:
                raise
            time.sleep(0.5)


def _page(text: str, start_char: int, max_chars: int) -> dict:
    if start_char < 0 or not 1 <= max_chars <= 50000:
        raise ValueError("start_char >= 0；max_chars 必须为 1 到 50000")
    if start_char > len(text):
        raise ValueError(f"start_char 超出内容长度 {len(text)}")
    end = min(start_char + max_chars, len(text))
    return {"content": text[start_char:end], "total_chars": len(text), "start_char": start_char,
            "truncated": end < len(text), "next_start_char": end if end < len(text) else None}


def _html(url: str):
    response = _get_resource(url)
    kind = response.headers.get("content-type", "").lower()
    if response.content.startswith(b"%PDF") or "application/pdf" in kind:
        raise ValueError("这是 PDF，请使用 read_pdf")
    if response.content.startswith((b"PK\x03\x04", b"\xd0\xcf\x11\xe0")):
        raise ValueError("这是工作簿或其他二进制文件，请使用对应文件工具")
    if kind.startswith(("image/", "audio/", "video/")):
        raise ValueError(f"HTML 工具不支持 {kind}")
    soup = BeautifulSoup(response.content, "html.parser")
    _check_challenge(soup)
    if not soup.find() or (kind and "html" not in kind and not soup.find(["html", "body", "table"])):
        raise ValueError("响应不是 HTML，文本请使用 read_text_file")
    return response, soup


def _check_challenge(soup):
    title = soup.title.get_text(" ", strip=True).casefold() if soup.title else ""
    if title in {"making sure you're not a bot!", "just a moment...", "attention required! | cloudflare",
                 "verify you are human", "access denied"}:
        raise ValueError("网站返回人机验证或拒绝访问页面，未取得正文；请换来源，不要把验证页当作资料")


def _quiet_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _links(soup, url: str) -> list[dict]:
    base = soup.find("base", href=True)
    base_url = urljoin(url, base["href"]) if base else url
    found = {}
    for anchor in soup.select("a[href], area[href]"):
        target = urljoin(base_url, anchor["href"])
        if urlsplit(target).scheme in {"http", "https"}:
            found.setdefault(target, {"text": anchor.get_text(" ", strip=True) or anchor.get("aria-label", ""),
                                      "url": target})
    return list(found.values())


def _metadata(soup, url: str) -> dict:
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    meta = soup.select_one('meta[name="description" i], meta[property="og:description"]')
    return {"url": url, "name": title, "title": title,
            "description": str(meta.get("content", "")) if meta else ""}


def _web_document(url: str, selector: str = ""):
    response, soup = _html(url)
    metadata = _metadata(soup, str(response.url))
    links = _links(soup, str(response.url))
    for tag in soup.select("script, style, template, svg, canvas, [hidden], [aria-hidden='true']"):
        tag.decompose()
    root = soup.select_one(selector) if selector else (soup.body or soup)
    if root is None:
        raise ValueError(f"CSS 选择器未匹配任何元素：{selector}")
    return metadata, root, links


def _ddgs_proxy() -> str | None:
    proxy = os.environ.get("DDGS_PROXY", "").strip()
    if not proxy:
        import api
        proxy = str(getattr(api, "DDGS_PROXY", "") or "").strip()
    return proxy or None


def _ddgs_search(query: str, max_results: int, include_domains: list[str]) -> list[dict]:
    filters = []
    for domain in include_domains:
        domain = domain.strip().lower()
        if not re.fullmatch(r"[a-z0-9.-]+", domain):
            raise ValueError(f"无效域名：{domain}")
        if not re.search(rf"(?<!\S)site:{re.escape(domain)}(?:\s|$)", query, re.IGNORECASE):
            filters.append(f"site:{domain}")
    search_query = query if not filters else f"{query} ({' OR '.join(filters)})"
    return DDGS(proxy=_ddgs_proxy(), timeout=15).text(
        search_query, max_results=max_results, backend="auto"
    )


def _claim_first_web_search() -> bool:
    """每个 tool_session 只让一次 web_search 补充 Wikipedia。

    ToolNode 可能并行执行多个搜索，因此用锁保证只有一个调用拿到资格。
    脱离 tool_session 单独调用工具时不做额外搜索。
    """
    session = _resources.get()
    if session is None:
        return False
    with _session_lock:
        if _FIRST_WEB_SEARCH in session:
            return False
        session[_FIRST_WEB_SEARCH] = True
        return True


def _search_result(item: dict) -> dict | None:
    url = item.get("href") or item.get("url") or ""
    if not isinstance(url, str) or urlsplit(url).scheme not in {"http", "https"}:
        return None
    title = item.get("title") or ""
    return {"url": url, "name": title, "title": title,
            "summary": item.get("body") or item.get("content") or "", "score": None}


def _is_wikipedia(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host == "wikipedia.org" or host.endswith(".wikipedia.org")


def _is_unreliable_wiki(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host == "fandom.com" or host.endswith(".fandom.com") or host.endswith(".wiki.gg") or "fextralife.com" in host


def _url_key(url: str) -> str:
    parts = urlsplit(url.strip())
    path = unquote(parts.path).rstrip("/") or "/"
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}{path}?{parts.query}".rstrip("?")


@reliable_tool
def web_search(query: str, max_results: int = 5,
               include_domains: list[str] | None = None) -> dict:
    """使用已安装的 DDGS 搜索库，返回 URL、标题和来源摘录

    include_domains 可限制网站，例如 ["wikipedia.org"]。搜索摘录只是线索，
    得到相关链接后必须打开原文核实。
    """
    if not query.strip() or not 1 <= max_results <= 20:
        raise ValueError("query 不能为空；max_results 为 1 到 20")
    domains = include_domains or re.findall(r"(?<!\S)site:([^\s/]+)", query)
    only_wikipedia = bool(domains) and all(domain.lower().endswith("wikipedia.org") for domain in domains)
    add_wikipedia = _claim_first_web_search() and not only_wikipedia
    primary_error = None
    try:
        data = _ddgs_search(query.strip(), max_results, domains)
    except Exception as exc:
        if not add_wikipedia:
            raise
        primary_error, data = exc, []
    results = []
    seen = set()
    for item in data:
        result = _search_result(item)
        if result is None or result["url"] in seen:
            continue
        seen.add(result["url"])
        results.append(result)

    wikipedia_search = None
    if add_wikipedia:
        try:
            candidates = (_search_result(item) for item in
                          _ddgs_search(query.strip(), 1, ["wikipedia.org"]))
            wikipedia = next((item for item in candidates
                              if item is not None and _is_wikipedia(item["url"])), None)
            if wikipedia is None:
                wikipedia_search = {"status": "not_found"}
            else:
                results = [item for item in results if item["url"] != wikipedia["url"]]
                results = results[:max_results - 1] + [wikipedia]
                wikipedia_search = {"status": "success", "url": wikipedia["url"]}
        except Exception as exc:
            # Wikipedia 是首次搜索的补充来源；它失败不能吞掉已成功的普通搜索。
            wikipedia_search = {"status": "error", "error": _tool_error(exc)}

    if not results and primary_error is not None:
        # 普通搜索和首次 Wikipedia 补充都没有带回结果，才将工具判为失败。
        raise primary_error

    output = {"query": query, "provider": "ddgs", "results": results[:max_results],
              "next_action": "打开相关来源核实原文。" if results else "没有搜索结果，请简化关键词或直接读取已知来源。"}
    if primary_error is not None:
        output["primary_search"] = {"status": "error", "error": _tool_error(primary_error)}
    if wikipedia_search is not None:
        output["first_wikipedia_search"] = wikipedia_search
    return output


@reliable_tool
def visit_webpage(url: str, max_chars: int = 8000, start_char: int = 0,
                  start_link: int = 0, max_links: int = 40, selector: str = "") -> dict:
    """读取网页概览和文字：URL、名称、摘要、文字及全页链接（包括导航和下载链接）。

    默认读取整个 body；文字用 next_start_char 续读，链接用 next_start_link 续读。
    selector 可选 CSS 定位指定区域。
    需要标签/CSS 结构时用 read_html。
    仅读取服务器返回的 HTML，不执行 JavaScript。
    """
    if start_link < 0 or not 1 <= max_links <= 200:
        raise ValueError("start_link >= 0；max_links 为 1 到 200")
    result, root, links = _web_document(url, selector)
    content = root.get_text("\n", strip=True)
    result.update(_page(content, start_char, max_chars))
    end_link = start_link + max_links
    result.update(summary=result["description"] or content[:400], links=links[start_link:end_link],
                  total_links=len(links), links_truncated=end_link < len(links),
                  next_start_link=end_link if end_link < len(links) else None)
    headings = root.select("h1, h2, h3, h4")
    result["headings"] = [{"text": h.get_text(" ", strip=True), "id": h.get("id", "")} for h in headings[:60]]
    result["headings_truncated"] = len(headings) > 60
    return result


@reliable_tool
def search_and_read(query: str, include_domains: list[str] | str | None = None,
                    max_chars: int = 8000, selector: str = "") -> dict:
    """复用 web_search 和 visit_webpage，读取 Wikipedia 与一个普通搜索来源。

    include_domains 可以是域名列表，也兼容模型传入的 JSON 列表字符串。
    目标读取失败时会继续尝试同类的下一个 URL，重复 URL 只读一次。
    返回 search、sources、attempted_urls、errors 和 complete。PDF/Excel 仍使用对应工具。
    """
    if not 1 <= max_chars <= 50000:
        raise ValueError("max_chars 必须为 1 到 50000")
    if isinstance(include_domains, str):
        try:
            decoded = json.loads(include_domains)
        except ValueError as exc:
            raise ValueError("include_domains 字符串必须是 JSON 列表") from exc
        if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
            raise ValueError("include_domains 字符串必须是 JSON 字符串列表")
        include_domains = decoded

    search = web_search.invoke({"query": query, "include_domains": include_domains})
    # 子工具会将异常转换为字符串，需要继续标记为错误，不能当成正常结果。
    if not isinstance(search, dict):
        raise ToolException(f"搜索失败：{search}")
    if not search["results"]:
        raise ToolException("没有搜索结果，请调整关键词或指定来源。")

    wikipedia, ordinary = [], []
    for item in search["results"]:
        url = str(item.get("url", ""))
        if not url:
            continue
        if _is_wikipedia(url):
            wikipedia.append(url)
        elif not _is_unreliable_wiki(url):
            ordinary.append(url)

    sources, attempted, errors, seen = [], [], [], set()
    for kind, candidates in (("wikipedia", wikipedia), ("ordinary", ordinary)):
        for url in candidates:
            key = _url_key(url)
            if key in seen:
                continue
            seen.add(key)
            attempted.append(url)
            page = visit_webpage.invoke({"url": url, "max_chars": max_chars, "selector": selector})
            if isinstance(page, dict):
                sources.append({**page, "source_type": kind})
                break
            errors.append({"url": url, "error": str(page)})

    if not sources:
        detail = (f"{errors[0]['url']}：{errors[0]['error']}" if errors else
                  "没有可读取的 Wikipedia 或普通网页结果")
        raise ToolException(f"所有候选读取失败：{detail}")
    complete = (any(item["source_type"] == "wikipedia" for item in sources) and
                any(item["source_type"] == "ordinary" for item in sources))
    return {"query": query, "search": search, "sources": sources,
            "attempted_urls": attempted, "errors": errors, "complete": complete}


@reliable_tool
def find_in_page(url: str, query: str, start_match: int = 0,
                 max_matches: int = 5, context_chars: int = 600) -> dict:
    """在完整网页正文中查找词句，返回命中附近文字和字符位置。忽略大小写。

    适用于长文章定位 Discography、Studio albums、Deposited 等；与 visit_webpage 的默认正文偏移一致。
    """
    if not query.strip() or start_match < 0 or not 1 <= max_matches <= 20 or not 50 <= context_chars <= 3000:
        raise ValueError("query 非空；start_match >= 0；max_matches 为 1 到 20；context_chars 为 50 到 3000")
    metadata, root, _ = _web_document(url)
    text = root.get_text("\n", strip=True)
    hits = list(re.finditer(re.escape(query), text, re.IGNORECASE))
    matches = [{"start_char": max(0, hit.start() - context_chars), "match_char": hit.start(),
                "content": text[max(0, hit.start() - context_chars):hit.end() + context_chars]}
               for hit in hits[start_match:start_match + max_matches]]
    end = start_match + max_matches
    return {**metadata, "query": query, "matches": matches, "total_matches": len(hits),
            "next_start_match": end if end < len(hits) else None}


@reliable_tool
def read_html(url: str, selector: str = "", mode: Literal["structure", "html"] = "structure",
              start_char: int = 0, max_chars: int = 12000) -> dict:
    """查看 HTML 标签结构或源码。selector 是 CSS 选择器，例如 main、#content、table。

    structure 返回缩进标签树、id/class/href/src 和短文本；html 返回选中区域原始标签。
    空 selector 读取整个页面；输出截断时用 next_start_char 续读。
    """
    response, soup = _html(url)
    roots = soup.select(selector) if selector else [soup]
    if not roots:
        raise ValueError(f"CSS 选择器未匹配任何元素：{selector}")
    if mode == "html":
        content = "\n".join(str(root) for root in roots)
    else:
        lines = []
        for root in roots:
            stack = [(root, 0)]
            while stack:
                node, depth = stack.pop()
                if not getattr(node, "name", None):
                    continue
                attrs = {k: node.attrs[k] for k in ("id", "class", "href", "src", "role", "type", "name") if k in node.attrs}
                own_text = " ".join(str(c).strip() for c in node.children if not getattr(c, "name", None)).strip()
                lines.append("  " * min(depth, 30) + node.name + " " + json.dumps(attrs, ensure_ascii=False)
                             + (" " + own_text[:100] if node.name not in {"script", "style"} else ""))
                stack.extend((c, depth + 1) for c in reversed(list(node.children)) if getattr(c, "name", None))
        content = "\n".join(lines)
    return {**_metadata(soup, str(response.url)), "selector": selector, "mode": mode,
            "matches": len(roots), **_page(content, start_char, max_chars)}


@reliable_tool
def read_pdf(url: str, max_pages: int = 5, start_page: int = 1,
             start_char: int = 0, max_chars: int = 20000, refresh: bool = False) -> dict:
    """读取 PDF 文字，页码从 1 开始。用 next_start_page 和 next_start_char 一起续读。

    下载、保存和切块入库由本地文件服务完成：local_path 是服务保存的绝对路径，
    可直接交给 run_python 读取。read_ok 只说明本次读到了内容，ingestion.status
    才说明这份文件是否已进长期记忆（queued/processing/indexed/failed），两者不要混用。
    默认复用服务里已保存的版本；需要重新确认远端有没有变化时才传 refresh=True。
    start_char 是起始页内的字符偏移；扫描页会标记 needs_ocr，本工具不进行 OCR。
    """
    return rag_client.read_pdf(url, max_pages=max_pages, start_page=start_page, start_char=start_char,
                        max_chars=max_chars, refresh=refresh)


@reliable_tool
def find_in_pdf(url: str, query: str, start_page: int = 1,
                max_matches: int = 5, context_chars: int = 600, refresh: bool = False) -> dict:
    """在 PDF 中定位关键词所在页和上下文，忽略大小写。用于长论文中找馆藏、机构等。

    每页最多返回一个片段；页码从1开始。用 next_start_page 继续查找，或 read_pdf 读取命中整页。
    与 read_pdf 共用服务上的同一份文件缓存。
    """
    return rag_client.find_in_pdf(url, query, start_page=start_page, max_matches=max_matches,
                           context_chars=context_chars, refresh=refresh)


@reliable_tool
def read_excel(url: str, summary: str, sheet_name: str | None = None, max_rows: int = 100,
               max_columns: int = 30, start_row: int = 0, start_column: int = 0,
               preview: bool = True, preview_rows: int = 5, refresh: bool = False) -> dict:
    """读取 xlsx/xls 工作簿：默认只返回工作表、列名、行列数和少量样例行。summary 必填。

    summary 用来描述整个工作簿的主题和用途：根据网页介绍、文件名称和已经看到的信息来写，
    保留有助于检索的名称、指标和已知范围，未知内容不要编造。它不能替代文件里的实际数据，
    也不要求先读完整个工作簿；服务端不会再生成摘要，这段文字就是该文件在长期记忆里的检索描述，
    同一文件版本的后续读取不会替换它。
    完整数据在 local_path，由 run_python 用 pandas 读取计算，不要为了计算把整张表分页回传。
    preview=False 时才返回原始单元格二维数组（保留表头），行列偏移从 0 开始，
    用 next_start_row/next_start_column 分块读取。公式返回文件中已保存的计算值，不重新计算公式。
    """
    return rag_client.read_excel(url, summary, sheet_name=sheet_name, max_rows=max_rows,
                          max_columns=max_columns, start_row=start_row, start_column=start_column,
                          preview=preview, preview_rows=preview_rows, refresh=refresh)


def _cell(value):
    """把单元格转成可 JSON 序列化的标量：NaN/NaT 归 None，时间按 ISO 文本。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value if not isinstance(value, float) or value == value else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and value != value:
        return None
    return str(value)


def _json_rows(block) -> list:
    return [[_cell(value) for value in row] for row in block.itertuples(index=False, name=None)]


def _tables_dir() -> Path:
    return DOWNLOADS_DIR / "tables"


def _save_table(frame, url: str, index: int) -> dict:
    """把整张网页表格存成 CSV，供 run_python 用 pandas 读取完整数据。

    文件名带页面 URL 的短摘要：同一站点不同页面的 table0 不会互相覆盖，
    而同一页面重复读取仍落到同一个文件，local_path 始终指向最近一次读到的版本。
    """
    folder = _tables_dir()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        host = _INVALID_FILENAME_CHARS.sub("_", urlsplit(url).hostname or "page")
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
        target = folder / f"{host}-{digest}-table{index}.csv"
        frame.to_csv(target, index=False, encoding="utf-8-sig")
    except (OSError, ValueError, TypeError) as exc:
        return {"saved": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"saved": True, "local_path": target.relative_to(DOWNLOADS_DIR.parent).as_posix()}


@reliable_tool
def read_webpage_tables(url: str, max_tables: int = 5, max_rows: int = 100,
                        start_table: int = 0, start_row: int = 0,
                        preview: bool = True, preview_rows: int = 5) -> dict:
    """读取 HTML 表格（保留列名，支持 rowspan/colspan）。

    默认对每张表只返回列名、行数和少量样例，并把完整表格保存为 downloads/tables/*.csv；
    需要完整数据时让 run_python 的代码读取 local_path 计算，不要分页回传整张表。
    preview=False 时按 start_row/max_rows 返回原始行。表格和数据行偏移从 0 开始。
    """
    import pandas as pd
    if start_table < 0 or start_row < 0 or not 1 <= max_tables <= 20 or not 1 <= max_rows <= 1000:
        raise ValueError("偏移 >= 0；max_tables 为 1 到 20；max_rows 为 1 到 1000")
    if not 1 <= preview_rows <= 20:
        raise ValueError("preview_rows 为 1 到 20")
    response, soup = _html(url)
    page_url = str(response.url)
    if not soup.find("table"):
        return {"url": page_url, "tables": [], "total_tables": 0, "truncated": False, "next_start_table": None}
    frames = pd.read_html(StringIO(str(soup)))
    next_table = start_table + max_tables if start_table + max_tables < len(frames) else None
    results = []
    for index in range(start_table, min(len(frames), start_table + max_tables)):
        frame = frames[index]
        entry = {"table_index": index, "columns": [str(c) for c in frame.columns],
                 "total_rows": len(frame), "local_path": None, "artifact_saved": False}
        saved = _save_table(frame, page_url, index)
        entry["local_path"] = saved.get("local_path")
        entry["artifact_saved"] = bool(saved["saved"])
        if not saved["saved"]:
            entry["artifact_error"] = str(saved.get("error") or "未知原因")
        if preview:
            sample = frame.iloc[start_row:start_row + preview_rows]
            entry.update(mode="preview", sample_rows=_json_rows(sample), sample_row_count=int(sample.shape[0]))
        else:
            next_row = start_row + max_rows if start_row + max_rows < len(frame) else None
            entry.update(mode="block", rows=_json_rows(frame.iloc[start_row:start_row + max_rows]),
                         truncated=next_row is not None, next_start_row=next_row)
        results.append(entry)
    return {"url": page_url, "tables": results, "total_tables": len(frames),
            "truncated": next_table is not None or any(t.get("truncated") for t in results),
            "next_start_table": next_table}


@reliable_tool
def read_text_file(url: str, start_char: int = 0, max_chars: int = 20000) -> dict:
    """读取 TXT/CSV/源码文字，按字符续读；只读取，不执行。支持 UTF-8 BOM、UTF-16 BOM。"""
    response = _get_resource(url)
    data = response.content
    kind = response.headers.get("content-type", "").lower()
    if kind.startswith(("image/", "audio/", "video/")) or data.startswith((b"%PDF", b"PK\x03\x04", b"\xd0\xcf\x11\xe0")):
        raise ValueError("这是二进制文件，请选择对应的文件工具")
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    text = data.decode(encoding)
    if "\x00" in text:
        raise ValueError("响应包含二进制内容")
    return {"url": str(response.url), **_page(text, start_char, max_chars)}


# 主模型直接提交 Python 代码；子进程只负责执行，不再请求第二个模型。
# 它没有隔离权限：能访问本机文件、环境变量和网络。
PYTHON_MAX_OUTPUT = 8000
PYTHON_POLL_SECONDS = 0.02
PYTHON_DEFAULT_TIMEOUT = 120
PYTHON_MAX_TIMEOUT = 900
PYTHON_SCRIPT_DIR = Path(tempfile.gettempdir()) / "run_python"
PYTHON_CWD = Path(__file__).resolve().parent


def _read_output_file(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def _kill_process_tree(process) -> None:
    """超时或输出过多时，结束运行代码及其子进程。"""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           capture_output=True, timeout=20)
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        process.kill()
        process.wait(timeout=10)
    except Exception:
        pass


def _run_python(code: str, timeout: int) -> dict:
    """在本机子进程执行代码，返回退出状态和有限长度的 stdout/stderr。"""
    try:
        PYTHON_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        script_path = PYTHON_SCRIPT_DIR / f"run_{uuid.uuid4().hex}.py"
        script_path.write_text(code, encoding="utf-8")
    except OSError as exc:
        return {"status": "error", "output": f"无法写入执行文件：{type(exc).__name__}: {exc}"}
    try:
        log = tempfile.NamedTemporaryFile(prefix="run_python_", suffix=".log", delete=False,
                                          dir=PYTHON_SCRIPT_DIR)
    except OSError as exc:
        _quiet_unlink(script_path)
        return {"status": "error", "output": f"无法准备输出文件：{type(exc).__name__}: {exc}"}
    log_path = Path(log.name)
    started, process, raw, status = time.monotonic(), None, b"", "error"
    try:
        with log:
            process = subprocess.Popen(
                [sys.executable, "-X", "utf8", str(script_path)],
                cwd=str(PYTHON_CWD), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt")
            while True:
                try:
                    process.wait(timeout=PYTHON_POLL_SECONDS)
                    status = "ok" if process.returncode == 0 else "error"
                    break
                except subprocess.TimeoutExpired:
                    raw = _read_output_file(log_path)
                    if len(raw) > PYTHON_MAX_OUTPUT:
                        status = "output_limit"
                        _kill_process_tree(process)
                        break
                    if time.monotonic() - started > timeout:
                        status = "timeout"
                        _kill_process_tree(process)
                        break
            raw = _read_output_file(log_path)
        if len(raw) > PYTHON_MAX_OUTPUT:
            status = "output_limit"
    except OSError as exc:
        raw = f"无法启动 Python：{type(exc).__name__}: {exc}".encode("utf-8")
    finally:
        if process is not None and process.poll() is None:
            _kill_process_tree(process)
        _quiet_unlink(log_path)
        _quiet_unlink(script_path)
    return {"status": status, "output": raw[:PYTHON_MAX_OUTPUT].decode("utf-8", errors="replace"),
            "exit_code": process.returncode if process is not None else None,
            "elapsed": round(time.monotonic() - started, 2)}


@tool
def run_python(code: str, timeout: int = PYTHON_DEFAULT_TIMEOUT) -> dict:
    """执行你写的 Python 代码并返回 print 输出或错误。

    直接把已核实的少量数字写进 code；大表格用 read_excel/read_webpage_tables 返回的
    local_path，在代码中读取本地文件。工作目录是项目目录。代码在本机执行，
    没有沙箱隔离，可读写文件和联网。默认超时 120 秒，上限 900 秒。
    """
    if not code.strip():
        return {"success": False, "status": "bad_request", "error": "code 不能为空"}
    limit = max(1, min(int(timeout), PYTHON_MAX_TIMEOUT))
    result = _run_python(code, limit)
    if result["status"] == "ok":
        return {"success": True, "status": "ok", "output": result["output"],
                "elapsed": result["elapsed"]}
    reason = {
        "timeout": f"执行超过 {limit} 秒，已终止子进程",
        "output_limit": f"输出超过 {PYTHON_MAX_OUTPUT} 字节，已终止",
    }.get(result["status"], f"执行失败（exit={result.get('exit_code')}）")
    return {"success": False, "status": result["status"],
            "error": f"{reason}：{result['output'] or '没有输出'}", "elapsed": result.get("elapsed")}



TOOLS = [web_search, visit_webpage, search_and_read, read_html, find_in_page, read_pdf, find_in_pdf,
         read_excel, read_webpage_tables, read_text_file, run_python]
