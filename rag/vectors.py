"""本地 embedding、PDF 切块与 Chroma 向量读写。

约定：
- 模型 intfloat/multilingual-e5-small，资料用 `passage: ` 前缀、问题用 `query: ` 前缀，向量归一化。
  （模型作者要求这种前缀配对，缺省会明显拉低跨语种检索质量。）
- 块长按 tokenizer 实际看到的 token 数计（约 350，重叠约 50），超过模型长度上限的输入必须继续拆，
  不能被模型静默截断。
- 一个集合 file_chunks，用 kind 区分 pdf_chunk / excel_summary；active 标记表示该版本当前有效。

Chroma 只由服务进程读写，因此这里的客户端是进程内单例。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
from pathlib import Path

import numpy as np

from rag import state as rag_state

MODEL_ID = os.environ.get("GAGENT_RAG_MODEL") or "intfloat/multilingual-e5-small"
MODEL_DIR_NAME = MODEL_ID.rsplit("/", 1)[-1]
MODELS_ROOT = Path(__file__).resolve().parents[1] / "rag_data" / "models"
COLLECTION_NAME = "file_chunks"
# 只取 CPU 推理需要的文件：e5 仓库还带 onnx/openvino/tf/flax 副本，全量下载是 2.2GB，
# 实际只要 safetensors + tokenizer 配置（约 470MB）。
MODEL_ALLOW_PATTERNS = ("*.json", "*.txt", "*.safetensors", "*.md", "1_Pooling/*")
KEEP_MODEL_DIRS = ("1_Pooling",)  # modules.json 指向它，删了就加载不出模型
KEEP_MODEL_FILES = ("config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
                    "modules.json", "sentence_bert_config.json", "model.safetensors",
                    "sentencepiece.bpe.model", "README.md", "rust_tokenizer.json")
CHUNK_TARGET_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 50
MODEL_MAX_TOKENS = 512  # 超出会被截断，所以切块必须留出自适应余量
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
EMBED_BATCH = 16

PDF_KIND = "pdf_chunk"
EXCEL_KIND = "excel_summary"

_model = None
_model_lock = threading.Lock()
_client = None
_collection = None
_collection_root = None
_store_lock = threading.Lock()

# 检索默认配置：候选与注入上限。
PDF_CANDIDATES = 8
EXCEL_CANDIDATES = 8
MAX_RESULTS = 4
MAX_CONTEXT_CHARS = 6000
# 门槛按真实文件实测校准（WHR24 158 页 + 微软 Financial Sample，18 个相关/无关问题）：
# e5-small 的余弦整体偏高，无关片段也能到 0.81-0.84，所以门槛只能贴着上沿取。
# 试过"减去语料均值再归一化"的重排：能把 Excel 的分差拉开，却会把 PDF 片段的相关性排序打乱
# （相关片段反而掉到 0.21，无关的 0.36），所以这里坚持用原始余弦。
# 中文问题问英文 PDF 时相关/无关只差千分位，门槛是"要不要带资料"的粗筛，不是正确率；
# 过不了门槛就返回空列表，让模型继续用工具取证。
PDF_MIN_SCORE = 0.85
EXCEL_MIN_SCORE = 0.865


def model_dir():
    """模型固定在项目的 rag_data/models 下：它是共享资产，不随测试替换的 DATA_ROOT 漂移。"""
    return MODELS_ROOT / MODEL_DIR_NAME


def model_ready() -> bool:
    path = model_dir()
    required = ("config.json", "modules.json", "sentence_bert_config.json", "model.safetensors")
    tokenizer_ready = (path / "tokenizer.json").is_file() or (path / "sentencepiece.bpe.model").is_file()
    return path.is_dir() and tokenizer_ready and all((path / name).is_file() for name in required)


def prepare_model(force: bool = False) -> str:
    """安装准备阶段把模型拉到项目本地目录，服务运行期间不再联网取模型。"""
    if model_ready() and not force:
        prune_model_dir()
        return str(model_dir())
    from huggingface_hub import snapshot_download
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(MODEL_ID, local_dir=str(model_dir()), allow_patterns=list(MODEL_ALLOW_PATTERNS))
    prune_model_dir()
    return path


def use_or_download() -> str:
    """返回可用的本地 embedding 模型目录；本地没有完整模型时自动下载。"""
    if model_ready():
        prune_model_dir()
        return str(model_dir())
    return prepare_model()


def prune_model_dir() -> int:
    """删掉推理用不到的模型文件（onnx / openvino / 重复权重），一次准备能省几 GB。"""
    path = model_dir()
    if not path.is_dir():
        return 0
    removed = 0
    for child in path.iterdir():
        if child.is_dir():
            if child.name in KEEP_MODEL_DIRS:
                continue
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
        elif child.name not in KEEP_MODEL_FILES:
            child.unlink(missing_ok=True)
            removed += 1
    return removed


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            local_model = use_or_download()
            import logging
            from sentence_transformers import SentenceTransformer
            # 整页文字在切块前会超过模型长度，tokenizer 的告警会淹掉服务日志。
            logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
            _model = SentenceTransformer(local_model, device="cpu", local_files_only=True)
            _model.max_seq_length = MODEL_MAX_TOKENS
        return _model


def _tokenize(texts: list[str], prefix: str) -> list[list[int]]:
    return get_model().tokenizer([prefix + text for text in texts], add_special_tokens=True)["input_ids"]


def count_tokens(text: str, prefix: str = PASSAGE_PREFIX) -> int:
    """按模型 tokenizer 计数，并把前缀算进去（这才是真正会被编码的长度）。"""
    if not text:
        return 0
    return max(len(ids) for ids in _tokenize([text], prefix))


def encode_passages(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype="float32")
    vectors = get_model().encode([PASSAGE_PREFIX + t for t in texts], batch_size=EMBED_BATCH,
                                 convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False)
    return vectors.astype("float32")


def encode_query(text: str) -> list[float]:
    vector = get_model().encode([QUERY_PREFIX + text], batch_size=1, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
    return [float(v) for v in vector[0]]


# ---------------------------------------------------------------- 向量库

def collection():
    global _client, _collection, _collection_root
    with _store_lock:
        if _collection is None or _collection_root != rag_state.DATA_ROOT:
            import chromadb
            from chromadb.config import Settings
            rag_state.ensure_dirs()
            _client = chromadb.PersistentClient(path=str(rag_state.vectors_dir()),
                                                settings=Settings(anonymized_telemetry=False,
                                                                  allow_reset=True))
            _collection = _client.get_or_create_collection(COLLECTION_NAME,
                                                           configuration={"hnsw": {"space": "cosine"}})
            _collection_root = rag_state.DATA_ROOT
        return _collection


def vector_count() -> int:
    return collection().count()


def _metadata_of(version: dict, kind: str, *, page: int = 0, start_char: int = 0,
                 end_char: int = 0) -> dict:
    """向量记录自带的出处：文件、版本、页码/片段位置、来源，Excel 另带实际工作表名。"""
    meta = {"version": version["sha256"], "version_id": version["id"], "doc_id": version["doc_id"],
            "kind": kind, "page": page, "start_char": start_char, "end_char": end_char,
            "file_name": version["file_name"], "source_url": version["final_url"] or "",
            "local_path": version["local_path"], "active": 0}
    if kind == EXCEL_KIND:
        meta["sheets"] = ",".join(json.loads(version.get("sheet_names") or "[]"))
    elif kind == PDF_KIND:
        meta["page_count"] = int(version.get("page_count") or 0)
    return meta


def _upsert(ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
    """分批编码后写入；同一版本重跑落到相同 id，不会产生重复记录。"""
    store = collection()
    for start in range(0, len(ids), 64):
        part = slice(start, start + 64)
        store.upsert(ids=ids[part], documents=texts[part], metadatas=metadatas[part],
                     embeddings=encode_passages(texts[part]).tolist())


def _ids_of(doc_id: str, version_sha: str) -> list[str]:
    found = collection().get(where={"$and": [{"doc_id": doc_id}, {"version": version_sha}]},
                             include=[])
    return list(found.get("ids") or [])


def set_version_active(doc_id: str, version_sha: str, active: bool) -> int:
    """翻转某版本全部向量的 active 标记；返回受影响的向量条数。"""
    ids = _ids_of(doc_id, version_sha)
    if not ids:
        return 0
    store = collection()
    existing = store.get(ids=ids, include=["metadatas"])
    metadatas = [{**(meta or {}), "active": 1 if active else 0} for meta in existing["metadatas"]]
    store.update(ids=existing["ids"], metadatas=metadatas)
    return len(existing["ids"])


def delete_version_vectors(doc_id: str, version_sha: str) -> int:
    count = len(_ids_of(doc_id, version_sha))
    if count:
        collection().delete(where={"$and": [{"doc_id": doc_id}, {"version": version_sha}]})
    return count


def version_chunk_count(doc_id: str, version_sha: str) -> int:
    return len(_ids_of(doc_id, version_sha))


def version_index(doc_id: str, version_sha: str) -> dict:
    """某个版本在向量库里的实际状态：条数、覆盖到哪些页、active 标记。"""
    got = collection().get(where={"$and": [{"doc_id": doc_id}, {"version": version_sha}]},
                           include=["metadatas"])
    metas = got["metadatas"] or []
    return {"vectors": len(got["ids"]),
            "pages": sorted({int(meta["page"]) for meta in metas if meta.get("page")}),
            "active": sorted({int(meta.get("active") or 0) for meta in metas})}


def drop_orphans() -> int:
    """删掉 SQLite 里已经没有版本记录的向量：换数据目录、删库重来都会留下这种残留。"""
    known = rag_state.known_version_shas(rag_state.connect())
    found = collection().get(include=["metadatas"])
    orphans = [meta.get("version") for meta in (found["metadatas"] or [])
               if (meta or {}).get("version") not in known]
    for sha in dict.fromkeys(orphans):
        if sha:
            collection().delete(where={"version": sha})
    return len(set(orphans))


def reconcile_active() -> int:
    """启动校对：以 SQLite 记录的有效版本为准修 Chroma 的 active 标记。

    切换过程中进程被打断，可能留下"向量已写入但标记没翻转"的中间状态；
    不修的话要么查不到资料，要么把已被替换的版本当成证据。
    """
    conn = rag_state.connect()
    try:
        active_ids = {row["id"] for row in rag_state.all_active_versions(conn)}
    finally:
        conn.close()
    found = collection().get(include=["metadatas"])
    wanted: dict[int, list[str]] = {}
    for identifier, meta in zip(found["ids"], found["metadatas"]):
        meta = meta or {}
        version_id = int(meta.get("version_id") or 0)
        want = 1 if version_id in active_ids else 0
        if int(meta.get("active") or 0) != want:
            wanted.setdefault(want, []).append(identifier)
    for want, ids in wanted.items():
        existing = collection().get(ids=ids, include=["metadatas"])
        collection().update(ids=existing["ids"],
                            metadatas=[{**(meta or {}), "active": want}
                                       for meta in existing["metadatas"]])
    return sum(len(ids) for ids in wanted.values())


# ---------------------------------------------------------------- 切块

_PARAGRAPH_SPLIT = re.compile(r"\n[ \t]*\n")
_SENTENCE_END = re.compile(r"[。！？!?；;]\s*|\.\s+|\n")


def _paragraphs(text: str) -> list[tuple[int, str]]:
    """段落及其在页内原文中的字符起点，供切块记录真实 start_char。"""
    pieces, cursor = [], 0
    for match in _PARAGRAPH_SPLIT.finditer(text):
        block = text[cursor:match.start()]
        if block.strip():
            pieces.append((cursor + block.index(block.strip()), block.strip()))
        cursor = match.end()
    tail = text[cursor:]
    if tail.strip():
        pieces.append((cursor + tail.index(tail.strip()), tail.strip()))
    return pieces


def _sentences(block: str) -> list[tuple[int, str]]:
    parts, position = [], 0
    for match in _SENTENCE_END.finditer(block):
        piece = block[position:match.end()]
        if piece.strip():
            parts.append((position, piece))
        position = match.end()
    if block[position:].strip():
        parts.append((position, block[position:]))
    return parts


def _hard_split(text: str, limit: int) -> list[tuple[int, str]]:
    """按 token 上限切割并带回片段在原文中的起点；尽量落在空格词边界。"""
    pieces, remaining, consumed = [], text, 0
    while remaining:
        keep = _longest_prefix(remaining, limit)
        if keep <= 0:
            keep = 1  # 单字符就超限时仍然保留，不能让内容凭空消失
        head = remaining[:keep]
        word_end = head.rfind(" ")
        if keep < len(remaining) and word_end > int(keep * 0.6):
            head, keep = head[:word_end], word_end
        if head.strip():
            pieces.append((consumed, head))
        consumed += keep
        remaining = remaining[keep:]
    return pieces


def _longest_prefix(text: str, limit: int) -> int:
    if _tokens(text) <= limit:
        return len(text)
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if _tokens(text[:middle]) <= limit:
            low = middle
        else:
            high = middle - 1
    return low


def _tokens(text: str) -> int:
    return count_tokens(text) if text else 0


def _segments(text: str, limit: int) -> list[tuple[int, str]]:
    """段落 → 句子 → 硬切，逐级细化直到每段都不超过 limit token。"""
    units = []
    for start, block in _paragraphs(text):
        if _tokens(block) <= limit:
            units.append((start, block))
            continue
        for offset, sentence in _sentences(block):
            if _tokens(sentence) <= limit:
                units.append((start + offset, sentence))
            else:
                for piece_offset, piece in _hard_split(sentence, limit):
                    units.append((start + offset + piece_offset, piece))
    return units


def _fit(text: str, limit: int, depth: int = 0) -> list[tuple[int, str]]:
    """保证每段都不超过 limit token：BPE 合并让 token 数对长度不单调，只能实测后再拆。"""
    if not text.strip() or _tokens(text) <= limit or len(text) <= 1 or depth > 12:
        return [(0, text)]
    pieces = _hard_split(text, limit)
    if len(pieces) < 2:
        middle = len(text) // 2
        pieces = [(0, text[:middle]), (middle, text[middle:])]
    out = []
    for piece_offset, piece in pieces:
        for inner_offset, body in _fit(piece, limit, depth + 1):
            out.append((piece_offset + inner_offset, body))
    return out


def chunk_page(text: str, page: int, key: str, target: int = CHUNK_TARGET_TOKENS,
               overlap: int = CHUNK_OVERLAP_TOKENS) -> list[dict]:
    """页内切块：每块约 target token，下一块带上上一块结尾约 overlap token 的片段。"""
    units = _segments(text, target)
    if not units:
        return []
    token_counts = [_tokens(piece) for _, piece in units]
    groups, start, tokens = [], 0, 0
    for index, (offset, piece) in enumerate(units):
        if start < index and tokens + token_counts[index] > target:
            groups.append((units[start][0], [unit[1] for unit in units[start:index]]))
            tail, tail_tokens = index, 0
            while tail > start and tail_tokens + token_counts[tail - 1] <= overlap:
                tail_tokens += token_counts[tail - 1]
                tail -= 1
            start, tokens = tail, tail_tokens + token_counts[index]
        else:
            tokens += token_counts[index]
    groups.append((units[start][0], [piece for _, piece in units[start:]]))
    records = []
    for group_start, pieces in groups:
        body = "\n".join(piece.strip() for piece in pieces).strip()
        # 拼接后的实际 token 数会高于各片段之和（换行单独计 token），所以还要复查一遍。
        for offset, fitted in _fit(body, target):
            records.append({"id": f"{key}-p{page}-{len(records)}", "text": fitted,
                            "page": page, "start_char": int(group_start) + offset,
                            "end_char": int(group_start) + offset + len(fitted)})
    return records


def _record_key(version: dict) -> str:
    """向量 id 带上 doc_id：同一份字节挂在不同来源 URL 时不会互相覆盖。"""
    return f"{version['doc_id']}-{version['sha256'][:16]}"


def build_pdf_records(version: dict) -> tuple[list[dict], int, list[int]]:
    """读取完整本地 PDF 生成切块记录；返回 (记录, 页数, 无文字页)。

    只处理本次 read_pdf 返回的那几页是不够的：入库必须覆盖整份文件的可提取文字。
    """
    from rag import storage as file_storage
    pages = file_storage.extract_pages(version["local_path"])
    if not pages:
        raise rag_state.PermanentError("PDF 没有任何页面")
    key = _record_key(version)
    records = []
    for index, text in enumerate(pages, start=1):
        if text:
            records += chunk_page(text, index, key)
    empty_pages = [i for i, text in enumerate(pages, start=1) if not text]
    if not records:
        raise rag_state.PermanentError(
            f"扫描版 PDF 无可提取文字（共 {len(pages)} 页），本版本不支持 OCR")
    return records, len(pages), empty_pages


def build_excel_records(version: dict) -> list[dict]:
    """Excel 只对工作模型给出的 summary 建向量，不逐单元格向量化。"""
    summary = str(version.get("summary") or "").strip()
    if not summary:
        raise rag_state.PermanentError("Excel 缺少 summary，无法建立检索描述")
    limit = MODEL_MAX_TOKENS - 16
    pieces = [(0, summary)] if _tokens(summary) <= limit else _fit(summary, limit)
    key = _record_key(version)
    return [{"id": f"{key}-summary-{position}", "text": body.strip(), "page": 0,
             "start_char": offset, "end_char": offset + len(body.strip())}
            for position, (offset, body) in enumerate(pieces) if body.strip()]


def index_version(version: dict) -> int:
    """生成并写入向量（active=0），是否生效由调用方在写完后统一切换。"""
    conn = rag_state.connect()
    try:
        if version["kind"] == "pdf":
            records, page_count, empty_pages = build_pdf_records(version)
            kind = PDF_KIND
        else:
            records = build_excel_records(version)
            kind = EXCEL_KIND
        rag_state.set_version_pages(conn, version["id"], page_count if kind == PDF_KIND else 0,
                                    empty_pages if kind == PDF_KIND else [], chunk_count=len(records),
                                    max_tokens=max((_tokens(item["text"]) for item in records), default=0))
        metadatas = [_metadata_of(version, kind, page=item["page"], start_char=item["start_char"],
                                  end_char=item["end_char"]) for item in records]
        _upsert([item["id"] for item in records], [item["text"] for item in records], metadatas)
        rag_state.set_version_status(conn, version["id"], "ready")
        conn.commit()
        return len(records)
    finally:
        conn.close()


# ---------------------------------------------------------------- 检索

MAX_ACTIVE_FILTER = 400  # 有效版本清单过长时退回 active 标记，避免把过滤条件撑爆


def _hits(kind: str, vector: list[float], candidates: int, active_shas: list[str] | None = None) -> list[dict]:
    """按余弦取候选，并把 metadata 摊平成接口需要的出处字段。

    候选阶段就用 SQLite 的有效版本清单过滤：只靠向量自带的 active 标记时，
    库里残留的孤儿向量（记录已不在 SQLite 里）会占掉 top-K 名额，随后又被
    版本核对丢掉，结果就变成"明明有资料却什么都查不到"。
    """
    if active_shas is None:
        active_shas = rag_state.active_version_shas(rag_state.connect())
    if not active_shas:
        return []
    conditions = [{"kind": kind}, {"active": 1}]
    if len(active_shas) <= MAX_ACTIVE_FILTER:
        conditions.append({"version": {"$in": active_shas}})
    found = collection().query(query_embeddings=[vector], n_results=candidates,
                               where={"$and": conditions},
                               include=["metadatas", "documents", "distances"])
    rows = []
    for index, identifier in enumerate((found.get("ids") or [[]])[0]):
        meta = found["metadatas"][0][index] or {}
        rows.append({"id": identifier, "text": found["documents"][0][index], "kind": kind,
                     "score": 1.0 - float(found["distances"][0][index]),
                     "doc_id": meta.get("doc_id"), "version": meta.get("version"),
                     "version_id": int(meta.get("version_id") or 0),
                     "page": int(meta.get("page") or 0),
                     "start_char": int(meta.get("start_char") or 0),
                     "end_char": int(meta.get("end_char") or 0),
                     "file_name": meta.get("file_name") or "", "source_url": meta.get("source_url") or "",
                     "local_path": meta.get("local_path") or "",
                     "sheets": [name for name in str(meta.get("sheets") or "").split(",") if name],
                     "page_count": int(meta.get("page_count") or 0)})
    return rows


def _dedupe_pdf(rows: list[dict]) -> list[dict]:
    """同页内字符范围明显重叠的片段只留分数最高的一条；同页不同片段仍可同时命中。"""
    kept = []
    for row in sorted(rows, key=lambda item: -item["score"]):
        crowded = any(_overlap_ratio((row["start_char"], row["end_char"]),
                                     (other["start_char"], other["end_char"])) > 0.3
                      for other in kept
                      if other["version"] == row["version"] and other["page"] == row["page"])
        if not crowded:
            kept.append(row)
    return kept


def _overlap_ratio(a: tuple[int, int], b: tuple[int, int]) -> float:
    shared = min(a[1], b[1]) - max(a[0], b[0])
    if shared <= 0:
        return 0.0
    return shared / max(1, min(a[1] - a[0], b[1] - b[0]))


def query_segments(text: str) -> list[str]:
    """本轮输入实际会被编码成几段；分段只切输入本身，不引入历史或改写。"""
    text = (text or "").strip()
    if not text:
        return []
    if _tokens(text) <= MODEL_MAX_TOKENS - 32:
        return [text]
    return [piece.strip() for _, piece in _fit(text, 300) if piece.strip()]


def _query_vectors(text: str) -> list[list[float]]:
    return [encode_query(segment) for segment in query_segments(text)]


def search(query: str, *, pdf_candidates: int = PDF_CANDIDATES, excel_candidates: int = EXCEL_CANDIDATES,
           pdf_min_score: float = PDF_MIN_SCORE, excel_min_score: float = EXCEL_MIN_SCORE,
           max_results: int = MAX_RESULTS, max_chars: int = MAX_CONTEXT_CHARS) -> list[dict]:
    """只检索已激活版本；没有合适结果返回空列表，不放宽阈值。"""
    text = (query or "").strip()
    if not text or vector_count() == 0:
        return []
    active_shas = rag_state.active_version_shas(rag_state.connect())
    if not active_shas:
        return []
    best: dict[str, dict] = {}
    for vector in _query_vectors(text):
        for row in (_hits(PDF_KIND, vector, pdf_candidates, active_shas)
                    + _hits(EXCEL_KIND, vector, excel_candidates, active_shas)):
            current = best.get(row["id"])
            if current is None or row["score"] > current["score"]:
                best[row["id"]] = row
    rows = list(best.values())
    pdf = _dedupe_pdf([r for r in rows if r["kind"] == PDF_KIND and r["score"] >= pdf_min_score])
    excel = _dedupe_excel([r for r in rows if r["kind"] == EXCEL_KIND and r["score"] >= excel_min_score])
    merged, used_chars = [], 0
    for row in sorted(pdf + excel, key=lambda item: -item["score"]):
        if len(merged) >= max_results or (used_chars + len(row["text"]) > max_chars and merged):
            continue
        merged.append(row)
        used_chars += len(row["text"])
    return _apply_version_state(merged)


def _apply_version_state(rows: list[dict]) -> list[dict]:
    """以 SQLite 里的有效版本为准再核对一遍，并给"还有更新版本没入库"的结果打标记。"""
    if not rows:
        return []
    conn = rag_state.connect()
    try:
        state = rag_state.ingestion_state(conn, [row["doc_id"] for row in rows])
    finally:
        conn.close()
    kept = []
    for row in rows:
        info = state.get(row["doc_id"]) or {}
        if info.get("active_version") != row["version_id"]:
            continue
        latest = info.get("latest_id")
        row["newer_version_pending"] = bool(latest and latest != row["version_id"])
        row["latest_version_status"] = info.get("latest_status")
        kept.append(row)
    return kept


def _dedupe_excel(rows: list[dict]) -> list[dict]:
    """同一个 Excel 版本只留一条，多段 summary 不重复注入。"""
    kept, seen = [], set()
    for row in sorted(rows, key=lambda item: -item["score"]):
        if row["version"] in seen:
            continue
        seen.add(row["version"])
        kept.append(row)
    return kept
