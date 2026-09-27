"""跨会话对话记忆：按轮切块、写入 Chroma 的对话集合、按会话聚合候选。

边界（与文件 RAG 的区别只体现在切块、排序和阈值上，不换 embedding 模型）：
- 向量库新增一个集合 conversation_chunks，仍用 multilingual-e5-small 与 `query:`/`passage:`
  前缀；文件集合 file_chunks 完全不动。
- 基本单位是"问答轮次"：本轮 user 与 final_answer 保留在同一条记录里，另附会话标题和少量
  上一轮原文，帮助理解"这个""方案 A"这类指代；上下文只来自真实历史，不预先改写。
- 整份会话对应一个版本（内容指纹 revision）。检测到旧轮次被修改或删除时重建该会话索引，
  旧版本向量在切换完成后删除，重建期间旧索引继续可用（与文件版本切换同一个口径）。
- 检索先取前 N 个片段，去掉同一轮相邻重复片段，再按 session 聚合：会话排序用该会话最佳
  片段的相似度（不是所有命中求和，避免长会话天然占优），最多返回 3 个候选。
- 阈值是相似度筛选参数，不叫置信度；对话有独立阈值，不套用文件的 0.85。

本模块只在服务进程里被调用（Chroma 只由服务进程读写）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path

from rag import state as rag_state
from rag import vectors as rag_store
from session_store import (SessionCorrupted, SessionNotFound, SessionStore, revision_of,
                           turn_id, turn_text)
import session_store

_logger = logging.getLogger("gagent.file_service")

COLLECTION_NAME = "conversation_chunks"
CHUNK_KIND = "conversation_turn"
CONTEXT_KIND = "conversation_context"

CHUNK_TARGET_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 50
CHUNK_LIMIT_TOKENS = rag_store.MODEL_MAX_TOKENS - 16  # 超过会被模型截断，切块必须留余量
TITLE_BUDGET_TOKENS = 64
# 标题 + 上一轮上下文的总预算：太少就看不懂"这个""方案 A"，太多就把本轮主体挤成配角
# （实测：上下文占到片段 88% 时，一条同领域无关问题也能拿到 0.86+，正文反而被拉低到门槛下）。
CONTEXT_TOTAL_BUDGET_TOKENS = 96
CONTEXT_PREV_USER_TOKENS = 44
CONTEXT_PREV_ANSWER_TOKENS = 44
# 正文命中优先：只有上下文确实让片段更贴近本轮正文时才把它编进向量。
# 实测 e5-small 上要点很紧：带上下文会让片段向"上一轮在聊什么"漂移，
# 松一点就会让同领域无关问题靠别人的正文越过门槛（0.004 时误命中回到 4/18）。
CONTEXT_EMBED_MARGIN = 0.02

TOP_CHUNKS = 20
MAX_CANDIDATES = 3

# 对话检索阈值：与文件不同的一套参数，用真实对话原文校准（11 个会话 / 40-52 条向量，
# 14 个标注正样本 + 18 个无关问题 + 1 组近似重复会话；验收脚本
#  scripts/acceptance_conversations.py 可一键复现）：
#   阈值 0.855 → 正样本 13/14、found/ambiguous 误命中 3/18
#   阈值 0.860 → 正样本 13/14、误命中 1/18
#   阈值 0.865 → 正样本 13/14、误命中 1/18（换了一种误命中）
#   阈值 0.870 → 正样本 13/14、误命中 0/18   ← 采用
#   阈值 0.880 → 正样本 11/14、误命中 0/18
# 正样本 top1 正确的最低分约 0.876、最高无关分约 0.866，间隔 +0.010；
# 0.870 既清零误命中又保住召回，是本套语料上误差最小的点。
# MIN_SCORE 是"要不要回一段历史"的相似度粗筛，不是正确率或置信度；MARGIN 决定第一、第二候选
# 是否算"明显相关"：分差小于它说明两个会话都可能，交给模型按原文判断。
CONVERSATION_MIN_SCORE = 0.870
CONVERSATION_AMBIGUOUS_MARGIN = 0.006

# not_found 时也会把最接近的候选报给调用方，但明显不相干的片段不该占这个位置：
# 候选的噪声线按相似度门槛的比例取值（0.93 → 默认门槛 0.86 时约 0.80），
# 这样换 embedding 或改门槛时不用重新标定；真实会话里无关问题最高分 0.833-0.855，
# 而同一会话里只由上一轮上下文带出来的残留命中只有 0.15-0.2，用这条线正好把它们分开。
CONVERSATION_CANDIDATE_FLOOR_RATIO = 0.93

_LABELS = {"title": "会话标题", "prev_user": "上一轮用户问题", "prev_answer": "上一轮回答中的邻近内容",
           "user": "本轮用户问题", "answer": "本轮最终回答"}

_store_lock = threading.Lock()
_collection = None
_collection_root = None


def _store():
    """对话集合的进程内单例；与文件集合共用同一个持久目录，只是不同 collection。"""
    global _collection, _collection_root
    with _store_lock:
        if _collection is None or _collection_root != rag_state.DATA_ROOT:
            import chromadb
            from chromadb.config import Settings
            rag_state.ensure_dirs()
            client = chromadb.PersistentClient(path=str(rag_state.vectors_dir()),
                                               settings=Settings(anonymized_telemetry=False,
                                                                 allow_reset=True))
            _collection = client.get_or_create_collection(
                COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}})
            _collection_root = rag_state.DATA_ROOT
        return _collection


def _sessions_dir() -> Path:
    """会话目录的唯一出处：session_store.SESSIONS_DIR。

    这里必须**运行时**读取（而不是 import 时把值绑下来）：服务进程是另一个进程，
    只有环境变量和显式传参能传进去；测试替换 session_store.SESSIONS_DIR 时也要立刻生效，
    否则隔离实例会去索引真实用户的 sessions/。
    """
    return session_store.default_sessions_dir()


def _store_for(sessions_dir) -> SessionStore:
    return SessionStore(sessions_dir) if sessions_dir else SessionStore(session_store.SESSIONS_DIR)


# ---------------------------------------------------------------- 长度工具

def _tokens(text: str) -> int:
    if not text:
        return 0
    try:
        return rag_store.count_tokens(text)
    except Exception:
        # 模型不可用时按字符粗估（中文约 1 字 1 token），只影响切块大小，不影响正确性。
        return max(1, len(text) // 2)


def _fit(text: str, limit: int) -> list[str]:
    """把文本切成每段不超过 limit token 的片段，不静默丢内容。"""
    text = (text or "").strip()
    if not text:
        return []
    if _tokens(text) <= limit:
        return [text]
    pieces, remaining = [], text
    guard = 0
    while remaining and guard < 10000:
        guard += 1
        keep = _longest_prefix(remaining, limit)
        if keep <= 0:
            keep = min(len(remaining), max(1, limit))
        head = remaining[:keep]
        if keep < len(remaining):
            cut = max(head.rfind("\n"), head.rfind("。"), head.rfind(" "), head.rfind("；"))
            if cut > int(keep * 0.5):
                head = head[:cut + 1]
        body = head.strip()
        if body:
            pieces.append(body)
        remaining = remaining[len(head):]
    if remaining.strip():
        pieces.extend(_fit(remaining, limit))
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


def _clip(text: str, limit: int) -> str:
    """把一段上下文压到 limit token 以内（只用于标题和上一轮邻近原文）。"""
    text = " ".join(str(text or "").split()).strip()
    if not text or limit <= 0:
        return ""
    if _tokens(text) <= limit:
        return text
    return text[:max(1, _longest_prefix(text, limit))].strip()


# ---------------------------------------------------------------- 切块

def _prev_tail(previous: dict, limit: int) -> str:
    """上一轮回答的邻近片段：优先取回答开头（结论），保留真实原文不做改写。"""
    answer = str((previous or {}).get("final_answer") or "").strip()
    return _clip(answer, limit)


_SEGMENT_END = re.compile(r"[。！？!?；;]\s*|\n+|\.\s+")
_WHITESPACE = re.compile(r"\s")


def _split_spans(text: str, limit: int) -> list[dict]:
    """把一段原文切成不超过 limit token 的片段，并保留每片在原文中的真实字符范围。

    段落 → 句子 → 硬切逐级细化；返回 [{"text", "start", "end"}]，不静默丢内容。
    """
    text = text or ""
    if not text.strip():
        return []
    if _tokens(text) <= limit:
        return [{"text": text.strip(), "start": _lstrip_at(text), "end": len(text.rstrip())}]
    spans: list[dict] = []
    position = 0
    for match in _SEGMENT_END.finditer(text):
        piece = text[position:match.end()]
        spans.extend(_hard_spans(piece, position, limit))
        position = match.end()
    spans.extend(_hard_spans(text[position:], position, limit))
    return [span for span in spans if span["text"].strip()]


def _hard_spans(text: str, offset: int, limit: int) -> list[dict]:
    if not text.strip():
        return []
    if _tokens(text) <= limit:
        start = _lstrip_at(text)
        return [{"text": text.strip(), "start": offset + start, "end": offset + len(text.rstrip())}]
    keep = _longest_prefix(text, limit)
    if keep <= 0:
        keep = max(1, min(len(text), limit))
    head, head_keep = text[:keep], keep
    cut = max(head.rfind(" "), head.rfind("\n"))
    if cut > int(keep * 0.6):
        head, head_keep = head[:cut], cut
    return _hard_spans(head, offset, limit) + _hard_spans(text[head_keep:], offset + head_keep, limit)


def _lstrip_at(text: str) -> int:
    return len(text) - len(text.lstrip())


def _blocks(session: dict) -> list[dict]:
    """把会话原文摊成带标签的块，按轮次先后排列：标题、每轮的上一轮邻近上下文、本轮主体。

    每个上下文块挂在它所属的轮次上，每个主体块带 parts（每片在原文中的真实字符范围）。
    """
    title = _clip(session.get("title"), TITLE_BUDGET_TOKENS)
    turns = session.get("turns") or []
    blocks: list[dict] = []
    if title:
        blocks.append({"role": "title", "turn_index": -1, "label": _LABELS["title"], "text": title,
                       "parts": [{"text": title, "start": 0, "end": len(title)}]})
    for index, turn in enumerate(turns):
        user = str(turn.get("user") or "")
        answer = str(turn.get("final_answer") or "")
        if not user.strip() and not answer.strip():
            continue
        previous = turns[index - 1] if index > 0 else None
        if previous:
            prev_user = _clip(previous.get("user"), CONTEXT_PREV_USER_TOKENS)
            prev_answer = _prev_tail(previous, CONTEXT_PREV_ANSWER_TOKENS)
            if prev_user:
                blocks.append({"role": "prev_user", "turn_index": index, "label": _LABELS["prev_user"],
                               "text": prev_user,
                               "parts": [{"text": prev_user, "start": 0, "end": len(prev_user)}]})
            if prev_answer:
                blocks.append({"role": "prev_answer", "turn_index": index,
                               "label": _LABELS["prev_answer"], "text": prev_answer,
                               "parts": [{"text": prev_answer, "start": 0, "end": len(prev_answer)}]})
        for role, body, label in (("user", user, _LABELS["user"]), ("answer", answer, _LABELS["answer"])):
            if body.strip():
                blocks.append({"role": role, "turn_index": index, "label": label, "text": body,
                               "parts": _split_spans(body, CHUNK_LIMIT_TOKENS - TITLE_BUDGET_TOKENS)})
    return blocks


def _render(pieces: list[dict]) -> str:
    return "\n".join(f"{piece['label']}：{piece['text']}" for piece in pieces)


def _trim(pieces: list[dict], limit: int) -> list[dict]:
    """按 token 预算裁掉整块：先丢上下文，再保留本轮主体。"""
    kept = list(pieces)
    order = {"prev_answer": 0, "prev_user": 1, "title": 2, "answer": 3, "user": 4}
    while len(kept) > 1 and _tokens(_render(kept)) > limit:
        victim = min(kept, key=lambda piece: order.get(piece["role"], 5))
        kept.remove(victim)
    return kept


def build_records(session: dict, *, target: int = CHUNK_TARGET_TOKENS,
                  limit: int = CHUNK_LIMIT_TOKENS) -> list[dict]:
    """按问答轮次切块：本轮 user 与 final_answer 保持关联，标题与上一轮上下文计入长度。

    长问题、长回答继续分块（带着角色和轮次继续往下切），不静默截断。
    每一轮先独立落盘再进入下一轮：上一轮的所有片段都只带"它自己那一轮"的上下文，
    不会把更早轮次的上下文累积进来（累积会让片段里 80% 都是别人的正文）。
    """
    session_id = str(session.get("id") or "")
    blocks = _blocks(session)
    title = next((block for block in blocks if block["role"] == "title"), None)
    groups: dict[int, list[dict]] = {}
    order: list[int] = []
    for block in blocks:
        if block["role"] == "title":
            continue
        if block["turn_index"] not in groups:
            groups[block["turn_index"]] = []
            order.append(block["turn_index"])
        groups[block["turn_index"]].append(block)
    records: list[dict] = []
    for turn_index in order:
        records.extend(_turn_records(session_id, len(records), title, groups[turn_index],
                                     target=target, limit=limit))
    return [record for record in records if record["text"].strip()]


def _turn_records(session_id: str, chunk_index: int, title: dict | None, blocks: list[dict], *,
                  target: int, limit: int) -> list[dict]:
    """一轮的片段：上下文块只来自这一轮（标题 + 紧邻上一轮），主体块按预算分片。"""
    breadcrumb = [block for block in blocks if block["role"] in ("prev_user", "prev_answer")]
    base = ([title] if title else []) + breadcrumb
    records: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        nonlocal current, chunk_index
        if current:
            records.append(_record(session_id, chunk_index, _trim(base + current, limit)))
            chunk_index += 1
            current = []

    for block in blocks:
        if block["role"] not in ("user", "answer"):
            continue
        for part in block["parts"]:
            piece = {**block, **part}
            if current and _tokens(_render(base + current + [piece])) > limit:
                flush()
            current.append(piece)
            if _tokens(_render(base + current)) >= target:
                flush()
    flush()
    return records


def _record(session_id: str, chunk_index: int, pieces: list[dict]) -> dict:
    """一条切块记录：片段标识、真实文本、角色归属和原文位置。

    片段前缀只包含标题与上一轮邻近上下文；角色标签随文本一起进向量，检索结果据此标注。
    """
    body = [piece for piece in pieces if piece["role"] not in ("title", "prev_user", "prev_answer")]
    turn_pieces = [piece for piece in body if piece["role"] in ("user", "answer")]
    turn_index = turn_pieces[0]["turn_index"] if turn_pieces else 0
    tid = turn_id(session_id, turn_index)
    roles = list(dict.fromkeys(piece["role"] for piece in turn_pieces)) or ["context"]
    primary = "answer" if "answer" in roles else ("user" if "user" in roles else roles[0])
    starts = [piece["start"] for piece in turn_pieces if piece["role"] == primary]
    ends = [piece["end"] for piece in turn_pieces if piece["role"] == primary]
    return {"chunk_id": f"{tid}:{chunk_index}", "session_id": session_id, "turn_id": tid,
            "turn_index": turn_index, "chunk_index": chunk_index, "roles": roles,
            "primary_role": primary, "start_char": min(starts) if starts else 0,
            "end_char": max(ends) if ends else 0, "text": _render(pieces),
            "body_text": _render(turn_pieces)}


def _chunk_key(session_id: str, revision: str, chunk_index: int) -> str:
    return f"{session_id}-{revision[:16]}-{chunk_index}"


def _metadata(version: dict, record: dict) -> dict:
    return {"kind": CHUNK_KIND, "session_id": version["session_id"], "revision": version["revision"],
            "version_id": version["id"], "turn_id": record["turn_id"],
            "turn_index": record["turn_index"], "chunk_index": record["chunk_index"],
            "session_title": version.get("title") or "", "primary_role": record["primary_role"],
            "roles": ",".join(record["roles"]), "start_char": record["start_char"],
            "end_char": record["end_char"], "turn_count": int(version.get("turn_count") or 0),
            "active": 0}


def _context_helps(body: str, text: str, embedding) -> bool:
    """带上下文编码是否明显更贴近本轮正文：比较两种编码到正文自身的余弦。

    上下文只用来帮助理解指代，不该把片段拉向"别人那轮在聊什么"。只有上下文确实让
    片段更代表本轮正文时，才把它编进向量；否则只编码正文。
    """
    if text == body:
        return False
    try:
        import numpy as np
        base = np.asarray(embedding(body), dtype="float32")
        full = np.asarray(embedding(text), dtype="float32")
        if base.size == 0 or full.size != base.size:
            return False
        similarity = float(np.dot(base, full) / (float(np.linalg.norm(base)) *
                                                 float(np.linalg.norm(full)) or 1.0))
        return similarity > 1.0 - CONTEXT_EMBED_MARGIN
    except Exception:
        return True  # 判断不了时按原样编码，宁可多带一点上下文也不要丢指代信息


def index_version(version: dict, *, sessions_dir=None) -> int:
    """读会话原文重新切块并写入向量（active=0），是否生效由调用方在写完后切换。"""
    session = _read_session(version["session_id"], sessions_dir)
    if session is None:
        raise rag_state.PermanentError(f"会话文件不存在或读不出来：{version['session_id']}")
    if revision_of(session) != version["revision"]:
        raise rag_state.PermanentError("会话内容已变化，该版本不再与原文一致，请按新版本重建")
    records = build_records(session)
    if not records:
        raise rag_state.PermanentError("该会话没有可索引的问答原文")
    # 编码用的文本按片段各自决定：多数情况就是正文本身，保证正文相关性不被前缀稀释。
    def embedding(value: str) -> list[float]:
        return rag_store.encode_passages([value])[0]

    payloads = []
    for item in records:
        text, body = item["text"], item["body_text"]
        if text == body or not body.strip():
            payloads.append(text if text.strip() else body)
            continue
        payloads.append(text if _context_helps(body, text, embedding) else body)
    store = _store()
    for start in range(0, len(records), 64):
        part = records[start:start + 64]
        payload = payloads[start:start + 64]
        # documents 存完整片段：检索结果返回给模型时要能看到标题与上一轮上下文，
        # 用于理解指代；embeddings 用 payload（多数情况就是正文），避免上下文主导相似度。
        store.upsert(ids=[_chunk_key(version["session_id"], version["revision"], item["chunk_index"])
                          for item in part],
                     documents=[item["text"] for item in part],
                     metadatas=[_metadata(version, item) for item in part],
                     embeddings=rag_store.encode_passages(payload).tolist())
    conn = rag_state.connect()
    try:
        rag_state.set_conversation_version_status(
            conn, version["id"], "ready", chunk_count=len(records),
            max_tokens=max((_tokens(item["text"]) for item in records), default=0))
    finally:
        conn.close()
    return len(records)


def _read_session(session_id: str, sessions_dir=None) -> dict | None:
    try:
        return _store_for(sessions_dir).read(session_id)
    except (SessionNotFound, SessionCorrupted, OSError, ValueError):
        return None


def vector_count() -> int:
    return _store().count()


def _ids_of(session_id: str, revision: str) -> list[str]:
    found = _store().get(where={"$and": [{"session_id": session_id}, {"revision": revision}]},
                         include=[])
    return list(found.get("ids") or [])


def set_version_active(session_id: str, revision: str, active: bool) -> int:
    ids = _ids_of(session_id, revision)
    if not ids:
        return 0
    store = _store()
    existing = store.get(ids=ids, include=["metadatas"])
    store.update(ids=existing["ids"],
                 metadatas=[{**(meta or {}), "active": 1 if active else 0}
                            for meta in existing["metadatas"]])
    return len(existing["ids"])


def delete_version_vectors(session_id: str, revision: str) -> int:
    count = len(_ids_of(session_id, revision))
    if count:
        _store().delete(where={"$and": [{"session_id": session_id}, {"revision": revision}]})
    return count


def drop_orphans() -> int:
    """删掉 SQLite 里已经没有版本记录的对话向量（换数据目录、删库重来会留下残留）。"""
    conn = rag_state.connect()
    try:
        known = rag_state.known_conversation_revisions(conn)
    finally:
        conn.close()
    found = _store().get(include=["metadatas"])
    orphans = []
    for meta in found["metadatas"] or []:
        meta = meta or {}
        key = f"{meta.get('session_id')}@{meta.get('revision')}"
        if key not in known:
            orphans.append((meta.get("session_id"), meta.get("revision")))
    for session_id, revision in dict.fromkeys(orphans):
        if session_id and revision:
            delete_version_vectors(session_id, revision)
    return len(set(orphans))


def reconcile_active() -> int:
    """启动校对：以 SQLite 的有效版本为准修 Chroma 的 active 标记。"""
    conn = rag_state.connect()
    try:
        wanted = set()
        for row in rag_state.all_conversations(conn):
            revision = row.get("active_revision")
            if revision:
                wanted.add(f"{row['session_id']}@{revision}")
    finally:
        conn.close()
    found = _store().get(include=["metadatas"])
    updates: dict[int, list[str]] = {}
    for identifier, meta in zip(found["ids"], found["metadatas"]):
        meta = meta or {}
        key = f"{meta.get('session_id')}@{meta.get('revision')}"
        want = 1 if key in wanted else 0
        if int(meta.get("active") or 0) != want:
            updates.setdefault(want, []).append(identifier)
    store = _store()
    for want, ids in updates.items():
        existing = store.get(ids=ids, include=["metadatas"])
        store.update(ids=existing["ids"],
                     metadatas=[{**(meta or {}), "active": want} for meta in existing["metadatas"]])
    return sum(len(ids) for ids in updates.values())


# ---------------------------------------------------------------- 同步 / 版本登记

def conversations_dir() -> Path:
    """当前生效的会话目录（供服务侧与排错查看）。"""
    return _sessions_dir()


def session_version(conn, session_id: str) -> str | None:
    return rag_state.conversation_revision(conn, session_id)


def sync_session(session, *, sessions_dir=None, conn=None, titles=None) -> dict:
    """把一个会话的最新原文登记为版本，必要时入队重建；幂等，可被重复调用。

    session 可以是 session_id 字符串，也可以是已读出的会话 dict（避免重复读盘）。
    返回 {session_id, title, revision, turn_count, status, job_id, version_id, queued, skipped}。
    """
    own_conn = conn is None
    conn = conn or rag_state.connect()
    try:
        if isinstance(session, dict):
            data = session
        else:
            data = _read_session(str(session), sessions_dir)
            if data is None:
                # 文件不存在和文件读不出来要分开：后者不该被安静跳过，提示里要能看出原因。
                status = "missing"
                note = ""
                try:
                    _store_for(sessions_dir).read_meta(str(session))
                except SessionNotFound:
                    pass
                except SessionCorrupted as exc:
                    status, note = "broken", str(exc)
                return {"session_id": str(session), "title": "", "revision": None, "turn_count": 0,
                        "status": status, "note": note, "job_id": None, "version_id": None,
                        "queued": False, "skipped": True}
        session_id = str(data.get("id") or "")
        stamp = rag_state.now_text()
        rag_state.upsert_conversation(conn, session_id=session_id, title=str(data.get("title") or ""),
                                      turn_count=len(data.get("turns") or []),
                                      created_at=str(data.get("created_at") or stamp),
                                      updated_at=str(data.get("updated_at") or stamp))
        revision = revision_of(data)
        if not revision or not (data.get("turns") or []):
            return {"session_id": session_id, "title": str(data.get("title") or ""), "revision": revision,
                    "turn_count": 0, "status": "empty", "job_id": None, "version_id": None,
                    "queued": False, "skipped": True}
        if rag_state.conversation_revision(conn, session_id) == revision:
            # 已经是指向这份原文的版本：重复通知不重复产生向量。
            version = rag_state.version_by_revision(conn, session_id, revision)
            return {"session_id": session_id, "title": str(data.get("title") or ""), "revision": revision,
                    "turn_count": len(data.get("turns") or []),
                    "status": (version or {}).get("status") or "indexed",
                    "job_id": None, "version_id": (version or {}).get("id"), "queued": False,
                    "skipped": True}
        version, _created = rag_state.ensure_conversation_version(
            conn, session_id, revision, len(data.get("turns") or []), str(data.get("title") or ""))
        job, _fresh = rag_state.ensure_conversation_job(conn, session_id, version["id"])
        conn.commit()
        return {"session_id": session_id, "title": str(data.get("title") or ""), "revision": revision,
                "turn_count": len(data.get("turns") or []), "status": job["status"],
                "job_id": job["job_id"], "version_id": version["id"],
                "queued": job["status"] == "queued", "skipped": False}
    finally:
        if own_conn:
            conn.close()


def sync_all(sessions_dir=None, *, wake=None, store=None) -> dict:
    """扫描全部会话补建缺失索引；已有会话首次启用时同样入队。

    用 scan_ids 逐个交代：读不出来的会话文件进 broken，不会被安静跳过。
    """
    root = _store_for(sessions_dir)
    conn = rag_state.connect()
    items, broken, queued = [], [], 0
    try:
        ids, malformed = root.scan_ids()
        broken.extend(malformed)
        for session_id in ids:
            item = sync_session(session_id, sessions_dir=sessions_dir, conn=conn)
            if item["status"] in ("broken", "missing"):
                broken.append(f"{session_id}（{item.get('note') or item['status']}）")
                continue
            items.append(item)
            queued += 1 if item["queued"] else 0
        conn.commit()
    finally:
        conn.close()
    if wake is not None and queued:
        try:
            wake()
        except Exception:
            pass
    return {"sessions": len(items), "queued": queued, "broken": broken, "items": items}


def pending_count(conn) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM conversation_jobs j WHERE j.status IN ('queued','processing','retry_wait')"
        " AND j.round=(SELECT MAX(j2.round) FROM conversation_jobs j2"
        "              WHERE j2.session_id=j.session_id AND j2.version_id=j.version_id)").fetchone()
    return row["n"] if row is not None else 0


def failed_count(conn) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM conversation_jobs j WHERE j.status='failed'"
        " AND j.round=(SELECT MAX(j2.round) FROM conversation_jobs j2"
        "              WHERE j2.session_id=j.session_id AND j2.version_id=j.version_id)").fetchone()
    return row["n"] if row is not None else 0


def indexed_count(conn) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM conversations WHERE active_revision IS NOT NULL").fetchone()
    return row["n"] if row is not None else 0


def session_count(conn) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()
    return row["n"] if row is not None else 0


# ---------------------------------------------------------------- 检索与候选聚合

def age_of_session(session_id: str, sessions_dir=None) -> str | None:
    session = _read_session(session_id, sessions_dir)
    return str(session.get("updated_at") or "") or None if session else None


def _active_revisions(conn) -> list[str]:
    """当前有效版本的 (session_id, revision) 清单，检索据此限定候选范围。"""
    out = []
    for row in rag_state.all_conversations(conn):
        revision = row.get("active_revision")
        if revision:
            out.append(f"{row['session_id']}@{revision}")
    return out


def _hits(query: str, top: int, active: set[str], primary_only: bool = False,
          context_only: bool = False) -> list[dict]:
    """只从"当前有效版本"的向量里取候选；按 revision 分批查询，保证 top-N 都来自有效版本。

    primary_only=True 只留"本片段命中的是那一轮自己的 user/final_answer"的记录；
    context_only=True 只留只命中标题或上一轮上下文的记录。标题和上一轮原文都只是帮助理解
    指代的上下文，拿上下文当命中会把相邻话题也算进来，所以它只能做兜底，不能抢正文命中。
    """
    vector = rag_store.encode_query(query)
    store = _store()
    rows: list[dict] = []
    chunks = list(active)
    for start in range(0, len(chunks), rag_store.MAX_ACTIVE_FILTER):
        part = chunks[start:start + rag_store.MAX_ACTIVE_FILTER]
        revisions = [chunk.split("@", 1)[1] for chunk in part]
        found = store.query(query_embeddings=[vector], n_results=max(1, top),
                            where={"$and": [{"kind": CHUNK_KIND}, {"active": 1},
                                            {"revision": {"$in": revisions}}]},
                            include=["metadatas", "documents", "distances"])
        for index, identifier in enumerate((found.get("ids") or [[]])[0]):
            meta = found["metadatas"][0][index] or {}
            key = f"{meta.get('session_id')}@{meta.get('revision')}"
            if key not in active:
                continue
            primary = str(meta.get("primary_role") or "")
            roles = [name for name in str(meta.get("roles") or "").split(",") if name]
            is_primary = primary in ("user", "answer")
            if primary_only and not is_primary:
                continue
            if context_only and is_primary:
                continue
            rows.append({"id": identifier, "text": found["documents"][0][index],
                         "score": 1.0 - float(found["distances"][0][index]),
                         "session_id": meta.get("session_id") or "", "revision": meta.get("revision") or "",
                         "session_title": meta.get("session_title") or "",
                         "turn_id": meta.get("turn_id") or "", "turn_index": int(meta.get("turn_index") or 0),
                         "chunk_index": int(meta.get("chunk_index") or 0),
                         "primary_role": primary, "roles": roles,
                         "start_char": int(meta.get("start_char") or 0),
                         "end_char": int(meta.get("end_char") or 0),
                         "primary_hit": primary in ("user", "answer")})
    rows.sort(key=lambda row: -row["score"])
    return rows[:top]


def _turn_key(row: dict) -> tuple[str, int]:
    return row["session_id"], row["turn_index"]


def dedupe_hits(rows: list[dict]) -> list[dict]:
    """同一个会话同一轮里字符范围重叠的相邻片段只留分数最高的一条。"""
    kept: list[dict] = []
    for row in sorted(rows, key=lambda item: -item["score"]):
        crowded = any(other["session_id"] == row["session_id"]
                      and other["turn_index"] == row["turn_index"]
                      and _overlap_ratio((row["start_char"], row["end_char"]),
                                         (other["start_char"], other["end_char"])) > 0.3
                      for other in kept)
        if not crowded:
            kept.append(row)
    return kept


def _overlap_ratio(a: tuple[int, int], b: tuple[int, int]) -> float:
    shared = min(a[1], b[1]) - max(a[0], b[0])
    if shared <= 0:
        return 0.0
    return shared / max(1, min(a[1] - a[0], b[1] - b[0]))


def _session_meta_map(conn, session_ids: list[str]) -> dict[str, dict]:
    wanted = list(dict.fromkeys(session_ids))
    if not wanted:
        return {}
    marks = ",".join("?" for _ in wanted)
    rows = conn.execute(f"SELECT * FROM conversations WHERE session_id IN ({marks})", wanted).fetchall()
    return {row["session_id"]: dict(row) for row in rows}


def _excerpt(text: str, role: str, start: int, end: int, limit: int = 400) -> str:
    """命中片段里属于该角色的部分；截断时给出实际范围。"""
    label = _LABELS.get(role, "")
    marker = f"{label}：" if label else ""
    body = text
    if marker and marker in text:
        body = text.split(marker, 1)[1]
    body = " ".join(body.split()).strip()
    if len(body) <= limit:
        return body
    return body[:limit].rstrip() + "…"


def _rank_sessions(conn, rows: list[dict], sessions_dir=None) -> list[dict]:
    """按 session 聚合命中：会话分数取该会话最佳片段的相似度，其余命中作为补充证据。

    不把同一会话所有命中的分数求和，否则长会话天然占优。
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["session_id"], []).append(row)
    if not grouped:
        return []
    metas = _session_meta_map(conn, list(grouped))
    ranked = []
    for session_id, hits in grouped.items():
        best = max(hits, key=lambda row: row["score"])
        meta = metas.get(session_id) or {}
        ranked.append({"session_id": session_id,
                       "title": meta.get("title") or best.get("session_title") or "",
                       "updated_at": meta.get("updated_at") or age_of_session(session_id, sessions_dir),
                       "score": round(float(best["score"]), 6), "best_turn_id": best["turn_id"],
                       "hits": sorted(hits, key=lambda row: -row["score"]),
                       "primary_hit": bool(best.get("primary_hit"))})
    # 本轮正文命中优先：同一会话里只要有正文命中，就用它当这个会话的最佳片段，
    # 否则上一轮的上下文片段可能顶掉真正该排在最前的讨论。
    for item in ranked:
        primary = [hit for hit in item["hits"] if hit.get("primary_hit")]
        if primary:
            best = max(primary, key=lambda row: row["score"])
            item["score"] = round(float(best["score"]), 6)
            item["best_turn_id"] = best["turn_id"]
            item["primary_hit"] = True
    ranked.sort(key=lambda item: -item["score"])
    return ranked


def decide_candidates(ranked: list[dict], *, threshold: float, noise: float, margin: float,
                      max_sessions: int) -> dict:
    """从按分数排好的会话候选里判定 found / ambiguous / not_found。

    纯函数，便于单独验证阈值与分差口径：
    - 低于 noise 的候选连"最接近的讨论"都算不上，不报；
    - 达到 threshold 的第一候选明显高于第二候选才算 found；
    - 多个候选都过门槛且分差小于 margin → ambiguous，交给模型按原文判断；
    - 都没过门槛 → not_found，但仍报出高于 noise 的候选供调用方判断。
    """
    limit = max(1, int(max_sessions))
    visible = [item for item in ranked if item["score"] >= noise]
    result = {"status": "not_found", "candidates": []}
    if not visible:
        return result
    relevant = [item for item in visible[:limit] if item["score"] >= threshold]
    if not relevant:
        result["candidates"] = [_candidate(item, position)
                                for position, item in enumerate(visible[:limit], start=1)]
        return result
    ambiguous = len(relevant) > 1 and (relevant[0]["score"] - relevant[1]["score"]) < margin
    result["status"] = "ambiguous" if ambiguous else "found"
    result["candidates"] = [_candidate(item, position) for position, item in enumerate(relevant, start=1)]
    return result


def search_conversations(query: str, *, exclude_sessions=(), top_chunks: int = TOP_CHUNKS,
                         max_sessions: int = MAX_CANDIDATES, sessions_dir=None,
                         min_score: float | None = None,
                         ambiguous_margin: float | None = None,
                         candidate_floor: float | None = None) -> dict:
    """检索相关片段 → 去重 → 按 session 聚合 → 判定 found / ambiguous / not_found。

    两轮候选：先只看"命中本轮正文"的片段；只有正文完全没有命中时，才退回到把标题和
    上一轮上下文也算命中的片段（那种命中自己说明不了讨论发生在哪里，只能兜底）。
    """
    text = (query or "").strip()
    result = {"query": text, "hits": [], "candidates": [], "status": "not_found"}
    if not text:
        return result
    try:
        if vector_count() == 0:
            return result
    except Exception as exc:
        _logger.warning("对话向量库不可读：%s", exc)
        return result
    conn = rag_state.connect()
    try:
        active = set(_active_revisions(conn))
        if not active:
            return result
        threshold = CONVERSATION_MIN_SCORE if min_score is None else float(min_score)
        noise = (threshold * CONVERSATION_CANDIDATE_FLOOR_RATIO if candidate_floor is None
                 else float(candidate_floor))
        margin = CONVERSATION_AMBIGUOUS_MARGIN if ambiguous_margin is None else float(ambiguous_margin)
        excluded = {str(item) for item in exclude_sessions if item}

        def gather(primary_only: bool, context_only: bool) -> list[dict]:
            rows = _hits(text, max(1, int(top_chunks)), active, primary_only, context_only)
            return [row for row in dedupe_hits(rows) if row["session_id"] not in excluded]

        ranked = _rank_sessions(conn, gather(True, False), sessions_dir)
        if not any(item["primary_hit"] for item in ranked):
            # 正文没有命中：此时才允许用上一轮上下文兜底。
            fallback = _rank_sessions(conn, gather(False, True), sessions_dir)
            if fallback and (not ranked or fallback[0]["score"] > ranked[0]["score"]):
                ranked = fallback
        decision = decide_candidates(ranked, threshold=threshold, noise=noise, margin=margin,
                                     max_sessions=max_sessions)
        result["status"] = decision["status"]
        result["candidates"] = decision["candidates"]
        if result["status"] == "found":
            result["hits"] = list(decision["candidates"][0]["hits"])
        elif result["status"] == "ambiguous":
            result["hits"] = [hit for item in decision["candidates"] for hit in item["hits"]]
        return result
    finally:
        conn.close()


def _candidate(item: dict, rank: int) -> dict:
    return {"session_id": item["session_id"], "title": item["title"], "updated_at": item["updated_at"],
            "score": item["score"], "best_turn_id": item["best_turn_id"], "rank": rank,
            "primary_hit": bool(item.get("primary_hit")), "hits": item["hits"]}


def recall_candidates(session_id: str, *, turn_ids=None, sessions_dir=None, max_turns: int = 4,
                      code_chars: int = 1200, deep_chars: int = 600) -> list[dict]:
    """按会话取整轮证据：直接从保存的问答原文取 user / final_answer，不用摘要代替原文。

    turn_ids 给出命中轮次；缺失或为空时取最靠后的几轮（deep 模式）。截断时返回真实范围。
    """
    session = _read_session(session_id, sessions_dir)
    if session is None:
        return []
    turns = session.get("turns") or []
    wanted: list[int] = []
    for item in turn_ids or []:
        try:
            index = int(str(item).rsplit(":", 1)[1])
        except (IndexError, ValueError):
            continue
        if 0 <= index < len(turns) and index not in wanted:
            wanted.append(index)
    if not wanted:
        wanted = list(range(max(0, len(turns) - max_turns), len(turns)))
    evidence = []
    for index in wanted[:max_turns]:
        turn = turns[index]
        user = str(turn.get("user") or "")
        answer = str(turn.get("final_answer") or "")
        evidence.append({
            "turn_id": turn_id(session_id, index), "turn_index": index,
            "user": user[:code_chars] + ("…" if len(user) > code_chars else ""),
            "user_truncated": len(user) > code_chars,
            "final_answer": answer[:deep_chars] + ("…" if len(answer) > deep_chars else ""),
            "final_answer_truncated": len(answer) > deep_chars,
            "user_chars": [0, min(len(user), code_chars)],
            "answer_chars": [0, min(len(answer), deep_chars)]})
    return evidence


def recall_candidates_deep(session_id: str, *, query: str = "", sessions_dir=None,
                           max_turns: int = 3) -> list[dict]:
    """模糊结果下按原文继续判断用：整轮原文给全一点，仍受长度上限约束。"""
    return recall_candidates(session_id, sessions_dir=sessions_dir, max_turns=max_turns,
                             code_chars=2000, deep_chars=2000)
