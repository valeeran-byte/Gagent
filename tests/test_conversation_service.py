"""跨会话回忆的服务侧验收测试：摘要缓存/并发/stale、对话接口与对话索引后台任务。

全部用桩摘要模型与假 embedding，不联网、不读写项目自己的 rag_data/ 与 sessions/；
会话原文放在临时 sessions 目录里（替换 session_store.SESSIONS_DIR）。
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import conversation_summary
import session_store
from rag import service as file_service
from rag import state as rag_state
from rag import vectors as rag_store

try:  # conversations.py 与文件服务并行开发：缺失时相关用例整体跳过
    from rag import conversations as conversations_module
except Exception:  # pragma: no cover - 只在并行开发的中间态出现
    conversations_module = None

HAS_CONVERSATIONS = conversations_module is not None
needs_conversations = unittest.skipUnless(HAS_CONVERSATIONS, "rag/conversations.py 未就绪")

MODEL_TOKENS = re.compile(r"\w+|[^\w\s]")
API_SETTINGS = {"API_KEY": "sk-test-must-not-leak", "MODEL": "stub-model",
                "BASE_URL": "http://127.0.0.1:1/v1"}
SESSION_ID = "abc123"

# 所有临时资料目录集中在一个父目录下；每次运行开头整体清空，句柄不释放也不会越攒越多。
SCRATCH = pathlib.Path(tempfile.gettempdir()) / "gagent_conversation_tests"
shutil.rmtree(SCRATCH, ignore_errors=True)


def uuid_hex() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


class FakeModel:
    """替代 SentenceTransformer：向量由词重叠决定，tokenizer 供 token 计数。"""

    def __init__(self):
        self.dimension = 64
        self.max_seq_length = 512

    def tokenizer(self, texts, add_special_tokens=True):
        return {"input_ids": [[0] + [1] * len(MODEL_TOKENS.findall(text)) for text in texts]}

    def encode(self, texts, **kwargs):
        return np.array([self._vector(text) for text in texts], dtype="float32")

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype="float32")
        body = text.split(": ", 1)[-1]
        for token in set(word.lower() for word in MODEL_TOKENS.findall(body)):
            vector[abs(hash(token)) % self.dimension] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


def summary_text(topic: str = "Excel 入库方式", text: str = "Excel 用模型提供的 summary 入库",
                 turn_ids=("abc123:0",)) -> str:
    payload = {"topic": topic,
               "user_requirements": [{"text": text, "turn_ids": list(turn_ids)}],
               "confirmed_decisions": [{"text": text, "turn_ids": list(turn_ids)}],
               "assistant_proposals": [], "open_questions": [], "superseded_decisions": []}
    return json.dumps(payload, ensure_ascii=False)


class FakeLLM:
    """桩摘要模型（同时充当 llm_factory）：记录提示词，按脚本返回内容。"""

    def __init__(self, reply=None, delay: float = 0.0):
        self.reply = summary_text() if reply is None else reply
        self.delay = delay
        self.prompts: list[str] = []
        self.timeouts: list[float] = []
        self.entered = threading.Event()
        self.gate: threading.Event | None = None

    def __call__(self, settings, timeout=None):
        self.timeouts.append(timeout)
        return self

    def invoke(self, prompt):
        self.entered.set()
        if self.gate is not None and not self.gate.wait(timeout=30):
            raise RuntimeError("测试闸门未放行")
        if self.delay:
            time.sleep(self.delay)
        self.prompts.append(prompt)
        if isinstance(self.reply, BaseException):
            raise self.reply
        if callable(self.reply):
            return SimpleNamespace(content=self.reply(prompt, len(self.prompts)))
        return SimpleNamespace(content=self.reply)


class TempRoot:
    """把资料目录和会话目录都挪到项目外的临时目录。"""

    def setUp(self):
        self.root = SCRATCH / f"t_{uuid_hex()}"
        self.sessions = self.root / "sessions"
        self.patch = patch.object(rag_state, "DATA_ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        sessions_patch = patch.object(session_store, "SESSIONS_DIR", self.sessions)
        sessions_patch.start()
        self.addCleanup(sessions_patch.stop)
        self.addCleanup(self.wipe)
        rag_state.ensure_dirs()
        self.sessions.mkdir(parents=True, exist_ok=True)

    def wipe(self) -> None:
        rag_store._collection = rag_store._client = rag_store._collection_root = None
        if conversations_module is not None:
            conversations_module._collection = None
            conversations_module._collection_root = None
        shutil.rmtree(self.root, ignore_errors=True)

    # ---------- 会话原文 ----------

    def write_session(self, session_id: str = SESSION_ID, turns=(("Excel 怎么入库？",
                                                                "Excel 用模型给的 summary 入库"),),
                      title: str = "RAG 设计", updated_at: str = "2026-02-02T10:00:00+08:00") -> dict:
        session = {"id": session_id, "title": title, "created_at": updated_at, "updated_at": updated_at,
                   "turns": [{"user": user, "final_answer": answer} for user, answer in turns]}
        session_store.SessionStore(self.sessions).write(session)
        return session

    def read_session(self, session_id: str = SESSION_ID) -> dict:
        return session_store.SessionStore(self.sessions).read(session_id)

    def revision(self, session_id: str = SESSION_ID) -> str:
        return session_store.revision_of(self.read_session(session_id))

    # ---------- 摘要 ----------

    def summarize(self, llm: FakeLLM, session_id: str = SESSION_ID, **kwargs) -> dict:
        return conversation_summary.use_or_create_summary(
            session_id, api_settings=API_SETTINGS, llm_factory=llm,
            store=session_store.SessionStore(self.sessions), **kwargs)

    def summary_rows(self) -> list[dict]:
        conn = rag_state.connect()
        try:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM conversation_summaries WHERE session_id=?", (SESSION_ID,)).fetchall()]
        finally:
            conn.close()


class ModelCase(TempRoot, unittest.TestCase):
    """假 embedding：摘要的 token 计数与对话切块都不加载真实模型。"""

    def setUp(self):
        super().setUp()
        self.model = FakeModel()
        for target, name, value in ((rag_store, "get_model", lambda: self.model),
                                    (rag_store, "model_ready", lambda: True)):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)


class ServiceCase(ModelCase):
    """真实 ASGI 应用 + 真实后台执行器，只把 embedding 换成假模型。"""

    def setUp(self):
        super().setUp()
        self._tester = None
        self.addCleanup(self.close_client)

    def client(self) -> TestClient:
        if self._tester is None:
            tester = TestClient(file_service.app)
            tester.__enter__()
            self._tester = tester
        return self._tester

    def close_client(self) -> None:
        tester = getattr(self, "_tester", None)
        if tester is not None:
            self._tester = None
            tester.__exit__(None, None, None)

    def post(self, path: str, payload: dict) -> tuple[int, dict]:
        reply = self.client().post(path, json=payload)
        return reply.status_code, reply.json()

    def get(self, path: str, params: dict | None = None) -> tuple[int, dict]:
        reply = self.client().get(path, params=params or {})
        return reply.status_code, reply.json()

    def put(self, path: str, payload: dict) -> tuple[int, dict]:
        reply = self.client().put(path, json=payload)
        return reply.status_code, reply.json()

    def wait_for(self, predicate, timeout: float = 30.0, message: str = "条件未满足") -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        self.fail(message)

    def fast_retries(self) -> None:
        patcher = patch.object(rag_state, "RETRY_DELAYS", (0.05, 0.1))
        patcher.start()
        self.addCleanup(patcher.stop)

    def conversation_job(self, job_id: str) -> dict | None:
        conn = rag_state.connect()
        try:
            return rag_state.get_conversation_job(conn, job_id)
        finally:
            conn.close()

    def detail(self, session_id: str) -> dict:
        status, body = self.get(f"/v1/conversations/{session_id}")
        self.assertEqual(status, 200, body)
        return body


# ---------------------------------------------------------------- 摘要

class TestConversationSummary(ModelCase):
    def test_generates_then_reuses_cache(self):
        turns = (("Excel 怎么入库？", "Excel 用模型给的 summary 入库"),
                 ("确认：按这个做", "好的，按这个方案执行"))
        self.write_session(turns=turns)
        llm = FakeLLM()
        first = self.summarize(llm)
        self.assertEqual(first["status"], "generated")
        self.assertEqual(first["source_revision"], self.revision())
        self.assertEqual(first["current_revision"], first["source_revision"])
        self.assertFalse(first["stale"])
        self.assertEqual(set(first["summary"]), set(conversation_summary.SUMMARY_KEYS))
        self.assertEqual(first["summary"]["confirmed_decisions"][0]["text"], "Excel 用模型提供的 summary 入库")
        self.assertTrue(first["summary"]["confirmed_decisions"][0]["turn_ids"])
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("abc123", llm.prompts[0], "提示词必须带上轮次标识")
        rows = self.summary_rows()
        self.assertEqual(len(rows), 1, "成功后才写一次缓存")

        second = self.summarize(llm)
        self.assertEqual(second["status"], "cached")
        self.assertEqual(second["summary"], first["summary"])
        self.assertEqual(second["generated_at"], first["generated_at"])
        self.assertEqual(len(llm.prompts), 1, "缓存命中不能再调用模型")
        self.assertEqual(len(self.summary_rows()), 1)

    def test_session_update_regenerates_for_new_revision(self):
        self.write_session(turns=(("第一轮问题", "第一轮回答"),))
        llm = FakeLLM()
        first = self.summarize(llm)
        self.write_session(turns=(("第一轮问题", "第一轮回答"), ("第二轮问题", "第二轮回答")))
        second = self.summarize(llm)
        self.assertEqual(second["status"], "generated")
        self.assertNotEqual(second["source_revision"], first["source_revision"])
        self.assertEqual(second["source_revision"], self.revision())
        self.assertEqual(len(llm.prompts), 2)
        revisions = {row["source_revision"] for row in self.summary_rows()}
        self.assertEqual(revisions, {first["source_revision"], second["source_revision"]},
                         "两个版本各留一条缓存，互不冒充")

    def test_update_during_generation_is_reported_stale(self):
        turns = (("第一轮问题", "第一轮回答"),)
        self.write_session(turns=turns)
        stale_revision = self.revision()

        def reply(prompt, count):
            if count == 1:  # 模型还在算的时候，会话又加了一轮
                self.write_session(turns=(("第一轮问题", "第一轮回答"), ("生成期间的新问题", "新回答")))
            return summary_text(turn_ids=(f"{SESSION_ID}:0",))

        llm = FakeLLM(reply=reply)
        result = self.summarize(llm)
        self.assertTrue(result["stale"])
        self.assertEqual(result["status"], "stale")
        self.assertEqual(result["source_revision"], stale_revision)
        self.assertEqual(result["current_revision"], self.revision())
        self.assertNotEqual(result["source_revision"], result["current_revision"])
        self.assertIn(stale_revision[:12], result["note"])
        self.assertIn(result["current_revision"][:12], result["note"])
        self.assertEqual({row["source_revision"] for row in self.summary_rows()}, {stale_revision},
                         "缓存只能记在它真正覆盖的那个版本上")

        again = self.summarize(llm)
        self.assertEqual(again["status"], "generated", "会话已更新，下次回忆要按新版本重新生成")
        self.assertEqual(len(llm.prompts), 2)

    def test_model_failure_returns_failed_and_writes_nothing(self):
        self.write_session()
        llm = FakeLLM(reply=RuntimeError("model down"))
        result = self.summarize(llm)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["summary"])
        self.assertTrue(result["note"])
        self.assertEqual(self.summary_rows(), [], "失败不能留下任何缓存")

    def test_invalid_json_is_a_failure_not_a_partial_cache(self):
        self.write_session()
        llm = FakeLLM(reply="这不是 JSON")
        result = self.summarize(llm)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["summary"])
        self.assertEqual(self.summary_rows(), [])

    def test_concurrent_requests_share_one_generation(self):
        self.write_session(turns=(("Excel 怎么入库？", "Excel 用 summary 入库"),))
        llm = FakeLLM()
        llm.gate = threading.Event()
        results: list[dict] = []
        errors: list[BaseException] = []

        def run():
            try:
                results.append(self.summarize(llm))
            except BaseException as exc:  # noqa: BLE001 - 测试里如实记录
                errors.append(exc)

        first = threading.Thread(target=run, name="summary-leader")
        first.start()
        self.assertTrue(llm.entered.wait(10), "第一次生成没有进入模型调用")
        second = threading.Thread(target=run, name="summary-waiter")
        second.start()
        time.sleep(0.3)  # 给等待方时间登记到同一个 in-flight 条目
        self.assertEqual(len(llm.prompts), 0, "模型还没返回，任何一方都不该提前调用第二次")
        llm.gate.set()
        first.join(timeout=30)
        second.join(timeout=30)
        self.assertEqual(errors, [])
        self.assertEqual(len(llm.prompts), 1, "同一 (会话, 版本) 的并发请求只能生成一次")
        self.assertEqual(len(results), 2)
        self.assertEqual({item["status"] for item in results}, {"generated"})
        self.assertEqual(results[0]["summary"], results[1]["summary"])
        self.assertEqual(len(self.summary_rows()), 1)

    def test_long_session_is_segmented_in_order_within_request_cap(self):
        turns = tuple((f"第 {index} 个问题：关于入库方式 {index} 的决定",
                       f"第 {index} 个回答：先按方案 {index} 执行，" + "细节很多。" * 20)
                      for index in range(8))
        session = self.write_session(turns=turns)
        turn_cost = conversation_summary._count_tokens(
            conversation_summary._render_turn(SESSION_ID, 0, session["turns"][0]))
        # 预算正好装两轮：8 轮 → 4 次请求，既触发分段又刚好在硬上限内
        with patch.object(conversation_summary, "SUMMARY_INPUT_TOKEN_BUDGET", turn_cost * 2):
            llm = FakeLLM()
            result = self.summarize(llm)
        self.assertEqual(result["status"], "generated")
        self.assertGreaterEqual(len(llm.prompts), 2, "超预算的长会话必须分段提炼")
        self.assertLessEqual(len(llm.prompts), conversation_summary.MAX_SUMMARY_REQUESTS,
                             "单次调用不能超过请求硬上限")
        for prompt in llm.prompts:
            self.assertLessEqual(len(re.findall(r"个问题：关于入库方式", prompt)), 2,
                                 "单次请求塞进的轮次不能超过 token 预算")
        joined = "\n".join(llm.prompts)
        for index in (0, 3, 7):
            self.assertIn(f"第 {index} 个问题", joined,
                          f"第 {index} 轮必须进模型，不能只总结最前或最后")

    def test_oversized_session_fails_without_giant_requests_or_partial_cache(self):
        """明显超过 4 个分段的会话：不把剩余分段并进最后一次请求，也不缓存半成品。"""
        turns = tuple((f"第 {index} 个问题：关于入库方式 {index} 的决定",
                       f"第 {index} 个回答：先按方案 {index} 执行，" + "细节很多。" * 20)
                      for index in range(8))
        self.write_session(turns=turns)
        with patch.object(conversation_summary, "SUMMARY_INPUT_TOKEN_BUDGET", 40):
            llm = FakeLLM()
            result = self.summarize(llm)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["summary"])
        self.assertIn("容量", result["note"], "必须说明会话超过本次摘要容量")
        self.assertEqual(llm.prompts, [], "容量不足时一次请求都不该发（更不该发超预算的请求）")
        self.assertEqual(self.summary_rows(), [], "会话没处理完就绝不能缓存半成品")

    def test_deadline_stops_before_caching_a_partial_summary(self):
        turns = tuple((f"问题 {index}", "回答 " + "内容。" * 40) for index in range(4))
        self.write_session(turns=turns)
        llm = FakeLLM(delay=0.3)
        with patch.object(conversation_summary, "SUMMARY_INPUT_TOKEN_BUDGET", 30), \
                patch.object(conversation_summary, "SUMMARY_DEADLINE_SECONDS", 0.1):
            result = self.summarize(llm)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["summary"])
        self.assertIn("超时", result["note"])
        self.assertEqual(len(self.summary_rows()), 0, "没做完就绝不能缓存半成品")
        self.assertLess(len(llm.prompts), conversation_summary.MAX_SUMMARY_REQUESTS)

    def test_settings_and_summary_never_carry_credentials(self):
        self.write_session()
        llm = FakeLLM()
        result = self.summarize(llm)
        self.assertNotIn(API_SETTINGS["API_KEY"], json.dumps(result, ensure_ascii=False))
        for row in self.summary_rows():
            self.assertNotIn(API_SETTINGS["API_KEY"], row["summary_json"])
            self.assertNotIn(API_SETTINGS["API_KEY"], row["source_revision"])

    def test_missing_session_fails_without_crashing(self):
        result = self.summarize(FakeLLM(), session_id="nosuch")
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["summary"])
        self.assertIn("找不到", result["note"])


# ---------------------------------------------------------------- 接口

class TestConversationEndpoints(ServiceCase):
    def test_search_rejects_empty_query(self):
        for query in ("", "   "):
            status, body = self.post("/v1/conversations/search", {"query": query})
            self.assertEqual(status, 400, query)
            self.assertEqual(body["error"], "bad_request")

    def test_detail_unknown_session_is_404(self):
        status, body = self.get("/v1/conversations/nosuch")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "not_found")

    def test_detail_reports_turns_and_index_state(self):
        self.client()
        self.write_session(turns=(("Excel 怎么入库？", "Excel 用 summary 入库"),
                                  ("确认：按这个做", "好的")))
        body = self.detail(SESSION_ID)
        self.assertEqual(body["session"]["id"], SESSION_ID)
        self.assertEqual(body["session"]["turn_count"], 2)
        self.assertEqual([turn["turn_id"] for turn in body["turns"]], ["abc123:0", "abc123:1"])
        self.assertEqual(body["turns"][0]["user"], "Excel 怎么入库？")
        self.assertIsNone(body["index"], "还没同步过就没有版本记录")
        self.assertIsNone(body["job"])
        self.assertIsNone(body["summary"])

    def test_put_and_get_summary_round_trip(self):
        self.client()
        self.write_session()
        revision = self.revision()
        payload = {"source_revision": revision, "summary_schema_version": 1,
                   "summary_json": summary_text(), "generated_at": "2026-02-02T11:00:00+08:00"}
        status, body = self.put(f"/v1/conversations/{SESSION_ID}/summary", payload)
        self.assertEqual(status, 200)
        self.assertEqual(body, {"session_id": SESSION_ID, "source_revision": revision,
                                "summary_schema_version": 1, "stored": True})
        status, record = self.get(f"/v1/conversations/{SESSION_ID}/summary")
        self.assertEqual(status, 200)
        self.assertEqual(record["current_revision"], revision)
        self.assertEqual(record["record"]["source_revision"], revision)
        self.assertEqual(record["record"]["generated_at"], "2026-02-02T11:00:00+08:00")
        self.assertEqual(json.loads(record["record"]["summary_json"])["topic"], "Excel 入库方式")
        self.assertIsNotNone(self.detail(SESSION_ID)["summary"])

    def test_put_summary_rejects_broken_json(self):
        self.client()
        self.write_session()
        for broken in ("{ not json", "[]"):
            status, body = self.put(f"/v1/conversations/{SESSION_ID}/summary",
                                    {"source_revision": self.revision(), "summary_schema_version": 1,
                                     "summary_json": broken, "generated_at": ""})
            self.assertEqual(status, 400, broken)
            self.assertEqual(body["error"], "bad_request")
        status, _ = self.put(f"/v1/conversations/{SESSION_ID}/summary",
                             {"source_revision": "", "summary_schema_version": 1,
                              "summary_json": summary_text(), "generated_at": ""})
        self.assertEqual(status, 400, "缺源版本必须拒绝")

    def test_put_summary_rejects_unknown_revision(self):
        self.client()
        self.write_session()
        status, body = self.put(f"/v1/conversations/{SESSION_ID}/summary",
                                {"source_revision": "f" * 64, "summary_schema_version": 1,
                                 "summary_json": summary_text(), "generated_at": ""})
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "revision_mismatch")

    def test_health_and_state_expose_conversation_stats(self):
        self.client()
        status, health = self.get("/health")
        self.assertEqual(status, 200)
        for field in ("documents", "vectors", "jobs_pending", "jobs_failed", "model_ready"):
            self.assertIn(field, health, "原有字段必须保留")
        for field in ("conversations", "conversation_vectors", "conversation_jobs_pending",
                      "conversation_jobs_failed"):
            self.assertIn(field, health)
            self.assertGreaterEqual(health[field], 0)
        status, state = self.get("/v1/state")
        self.assertEqual(status, 200)
        self.assertIn("events", state)
        self.assertIn("jobs", state)
        self.assertIn("model_ready", state)
        self.assertIsInstance(state["conversation_jobs"], list)
        self.assertEqual(state["conversations_indexed"], 0)
        self.assertEqual(state["conversations_pending"], 0)
        self.assertEqual(state["conversations_broken"], [])


@needs_conversations
class TestConversationSearchAndRecall(ServiceCase):
    """检索/回忆的响应整形：用桩检索结果驱动，不依赖真实向量与阈值。"""

    def canned_found(self, session_id: str, hits: list[dict] | None = None) -> dict:
        hits = hits or [{"session_id": session_id, "session_title": "RAG 设计", "session_updated_at": None,
                         "turn_id": f"{session_id}:0", "turn_index": 0, "chunk_index": 0,
                         "primary_role": "answer", "primary_hit": True, "score": 0.93,
                         "start_char": 0, "end_char": 12,
                         "text": "本轮最终回答：Excel 用模型给的 summary 入库"}]
        candidate = {"session_id": session_id, "title": "RAG 设计", "updated_at": "2026-02-02T10:00:00+08:00",
                     "score": 0.93, "best_turn_id": f"{session_id}:0", "primary_hit": True, "hits": hits}
        return {"query": "Excel 怎么入库", "hits": hits, "candidates": [candidate], "status": "found"}

    def test_recall_not_found_returns_candidates_and_no_summary(self):
        self.client()
        canned = {"query": "无关问题", "hits": [], "candidates": [], "status": "not_found"}
        with patch.object(conversations_module, "search_conversations", return_value=canned), \
                patch.object(conversation_summary, "use_or_create_summary") as summary:
            status, body = self.post("/v1/conversations/recall", {"query": "无关问题"})
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "not_found", "query": "无关问题", "candidates": []})
        summary.assert_not_called()

    def test_recall_ambiguous_lists_evidence_without_summary(self):
        self.client()
        self.write_session(turns=(("Excel 怎么入库？", "Excel 用 summary 入库"),), title="RAG 设计")
        self.write_session(session_id="def456", title="另一个会话",
                           turns=(("Excel 用什么？", "用文件里的 summary"),))
        candidates = [{"session_id": SESSION_ID, "title": "RAG 设计", "updated_at": "2026-02-02T10:00:00+08:00",
                       "score": 0.9, "best_turn_id": "abc123:0", "primary_hit": True,
                       "hits": [{"session_id": SESSION_ID, "turn_id": "abc123:0", "turn_index": 0,
                                 "primary_hit": True, "score": 0.9}]},
                      {"session_id": "def456", "title": "另一个会话",
                       "updated_at": "2026-02-01T10:00:00+08:00", "score": 0.898,
                       "best_turn_id": "def456:0", "primary_hit": True,
                       "hits": [{"session_id": "def456", "turn_id": "def456:0", "turn_index": 0,
                                 "primary_hit": True, "score": 0.898}]}]
        canned = {"query": "Excel 怎么入库", "hits": [], "candidates": candidates, "status": "ambiguous"}
        with patch.object(conversations_module, "search_conversations", return_value=canned), \
                patch.object(conversation_summary, "use_or_create_summary") as summary:
            status, body = self.post("/v1/conversations/recall", {"query": "Excel 怎么入库"})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ambiguous")
        self.assertEqual([item["session_id"] for item in body["candidates"]], [SESSION_ID, "def456"])
        self.assertEqual(body["candidates"][0]["evidence"][0]["turn_id"], "abc123:0")
        self.assertEqual(body["candidates"][0]["evidence"][0]["user"], "Excel 怎么入库？")
        self.assertEqual(body["candidates"][0]["evidence"][0]["final_answer"], "Excel 用 summary 入库")
        summary.assert_not_called()

    def test_recall_found_returns_summary_and_original_evidence(self):
        self.client()
        self.write_session()
        canned = self.canned_found(SESSION_ID)
        outcome = {"status": "generated", "summary": json.loads(summary_text()),
                   "source_revision": "r" * 64, "current_revision": "r" * 64, "stale": False,
                   "generated_at": "2026-02-02T11:00:00+08:00", "note": None, "elapsed": 0.1}
        with patch.object(conversations_module, "search_conversations", return_value=canned), \
                patch.object(conversation_summary, "use_or_create_summary", return_value=outcome) as summary:
            status, body = self.post("/v1/conversations/recall",
                                     {"query": "Excel 怎么入库", "api_settings": API_SETTINGS})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "found")
        self.assertEqual(body["session"]["id"], SESSION_ID)
        self.assertEqual(body["summary_status"], "generated")
        self.assertEqual(body["summary"]["topic"], "Excel 入库方式")
        self.assertFalse(body["summary_stale"])
        self.assertEqual(body["summary_source_revision"], "r" * 64)
        self.assertEqual(body["candidates"], [{"session_id": SESSION_ID, "title": "RAG 设计",
                                                "updated_at": "2026-02-02T10:00:00+08:00",
                                                "score": 0.93, "primary_hit": True}])
        evidence = body["evidence"][0]
        self.assertEqual(evidence["turn_id"], "abc123:0")
        self.assertEqual(evidence["user"], "Excel 怎么入库？")
        self.assertEqual(evidence["final_answer"], "Excel 用模型给的 summary 入库")
        self.assertEqual(evidence["score"], 0.93)
        self.assertEqual((evidence["start_char"], evidence["end_char"]), (0, 12))
        self.assertFalse(evidence["truncated"])
        self.assertTrue(evidence["excerpt"])
        summary.assert_called_once()
        passed = summary.call_args.kwargs["api_settings"]
        self.assertEqual(passed["API_KEY"], API_SETTINGS["API_KEY"],
                         "请求里的配置原样交给摘要层，只在本机内存里用")

    def test_recall_ignores_candidates_without_primary_hit(self):
        """只有上下文片段命中、正文没命中的候选不算找到，不生成摘要。"""
        self.client()
        candidate = {"session_id": SESSION_ID, "title": "RAG 设计", "updated_at": "2026-02-02T10:00:00+08:00",
                     "score": 0.9, "best_turn_id": "abc123:0", "primary_hit": False,
                     "hits": [{"session_id": SESSION_ID, "turn_id": "abc123:0", "turn_index": 0,
                               "primary_hit": False, "score": 0.9}]}
        canned = {"query": "Excel 怎么入库", "hits": [], "candidates": [candidate], "status": "found"}
        with patch.object(conversations_module, "search_conversations", return_value=canned), \
                patch.object(conversation_summary, "use_or_create_summary") as summary:
            status, body = self.post("/v1/conversations/recall", {"query": "Excel 怎么入库"})
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "not_found", "query": "Excel 怎么入库", "candidates": []})
        summary.assert_not_called()

    def test_recall_named_session_wins_even_if_not_best_candidate(self):
        """用户点名会话的定向回忆：以该会话为准，检索只为取它的证据。"""
        self.client()
        self.write_session(turns=(("Excel 怎么入库？", "Excel 用 summary 入库"),), title="目标会话")
        self.write_session(session_id="def456", title="更靠前的候选",
                           turns=(("Excel 用什么？", "用文件里的 summary"),))
        canned = {"query": "Excel 怎么入库", "hits": [], "status": "found",
                  "candidates": [{"session_id": "def456", "title": "更靠前的候选",
                                  "updated_at": "2026-02-03T10:00:00+08:00", "score": 0.95,
                                  "best_turn_id": "def456:0", "primary_hit": True, "hits": []},
                                 {"session_id": SESSION_ID, "title": "目标会话",
                                  "updated_at": "2026-02-02T10:00:00+08:00", "score": 0.8,
                                  "best_turn_id": "abc123:0", "primary_hit": True,
                                  "hits": [{"session_id": SESSION_ID, "turn_id": "abc123:0", "turn_index": 0,
                                            "primary_hit": True, "score": 0.8, "start_char": 0,
                                            "end_char": 15, "text": "本轮最终回答：Excel 用 summary 入库"}]}]}
        outcome = {"status": "cached", "summary": json.loads(summary_text()), "source_revision": "r" * 64,
                   "current_revision": "r" * 64, "stale": False, "generated_at": "now", "note": None,
                   "elapsed": 0.0}
        with patch.object(conversations_module, "search_conversations", return_value=canned) as search, \
                patch.object(conversation_summary, "use_or_create_summary", return_value=outcome) as summary:
            status, body = self.post("/v1/conversations/recall",
                                     {"query": "Excel 怎么入库", "session_id": SESSION_ID,
                                      "exclude_session": "cur999"})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "found")
        self.assertEqual(body["session"]["id"], SESSION_ID)
        self.assertEqual(body["session"]["title"], "目标会话")
        self.assertEqual(body["evidence"][0]["turn_id"], "abc123:0")
        self.assertEqual(body["evidence"][0]["final_answer"], "Excel 用 summary 入库")
        self.assertEqual(search.call_args.kwargs["exclude_sessions"], ["cur999"],
                         "定向回忆只排除显式要排除的会话，不排除目标会话本身")
        self.assertEqual(summary.call_args.args[0], SESSION_ID)

    def test_recall_named_session_without_hits_falls_back_to_recent_turns(self):
        self.client()
        self.write_session(turns=(("第一轮问题", "第一轮回答"), ("第二轮问题", "第二轮回答")),
                           title="目标会话")
        canned = {"query": "完全无关的检索词", "hits": [], "candidates": [], "status": "not_found"}
        with patch.object(conversations_module, "search_conversations", return_value=canned), \
                patch.object(conversation_summary, "use_or_create_summary",
                             return_value={"status": "failed", "summary": None, "note": "模型不可用",
                                           "source_revision": None, "current_revision": None,
                                           "stale": False, "generated_at": None, "elapsed": 0.0}):
            status, body = self.post("/v1/conversations/recall",
                                     {"query": "完全无关的检索词", "session_id": SESSION_ID})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "found")
        self.assertEqual([item["turn_id"] for item in body["evidence"]], ["abc123:0", "abc123:1"])
        self.assertEqual(body["evidence"][-1]["final_answer"], "第二轮回答")

    def test_recall_named_session_missing_is_not_found(self):
        self.client()
        canned = {"query": "问题", "hits": [], "candidates": [], "status": "not_found"}
        with patch.object(conversations_module, "search_conversations", return_value=canned):
            status, body = self.post("/v1/conversations/recall",
                                     {"query": "问题", "session_id": "nosuch"})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "not_found")

    def test_recall_excludes_current_session_when_not_targeted(self):
        self.client()
        canned = {"query": "问题", "hits": [], "candidates": [], "status": "not_found"}
        with patch.object(conversations_module, "search_conversations", return_value=canned) as search:
            self.post("/v1/conversations/recall",
                      {"query": "问题", "exclude_session": "cur123"})
        self.assertEqual(search.call_args.kwargs["exclude_sessions"], ["cur123"])

    def test_recall_summary_failure_still_returns_evidence(self):
        self.client()
        self.write_session()
        canned = self.canned_found(SESSION_ID)
        failed = {"status": "failed", "summary": None, "source_revision": "r" * 64,
                  "current_revision": "r" * 64, "stale": False, "generated_at": None,
                  "note": "摘要生成超时", "elapsed": 60.0}
        with patch.object(conversations_module, "search_conversations", return_value=canned), \
                patch.object(conversation_summary, "use_or_create_summary", return_value=failed):
            status, body = self.post("/v1/conversations/recall", {"query": "Excel 怎么入库"})
        self.assertEqual(status, 200)
        self.assertEqual(body["summary_status"], "failed")
        self.assertIsNone(body["summary"])
        self.assertEqual(body["summary_note"], "摘要生成超时")
        self.assertTrue(body["evidence"], "摘要失败也必须照常给原文证据")

    def test_recall_returns_evidence_when_the_session_exceeds_summary_capacity(self):
        """超长会话摘要按容量判失败时，原文 evidence 照常返回，且不落半成品缓存。"""
        self.client()
        turns = tuple((f"第 {index} 个问题：关于入库方式 {index} 的决定",
                       f"第 {index} 个回答：先按方案 {index} 执行，" + "细节很多。" * 20)
                      for index in range(8))
        self.write_session(turns=turns)
        canned = self.canned_found(SESSION_ID)
        with patch.object(conversations_module, "search_conversations", return_value=canned), \
                patch.object(conversation_summary, "SUMMARY_INPUT_TOKEN_BUDGET", 40):
            status, body = self.post("/v1/conversations/recall",
                                     {"query": "Excel 怎么入库", "api_settings": API_SETTINGS})
        self.assertEqual(status, 200)
        self.assertEqual(body["summary_status"], "failed")
        self.assertIsNone(body["summary"])
        self.assertIn("容量", body["summary_note"])
        self.assertTrue(body["evidence"], "摘要失败也必须照常给原文证据")
        self.assertEqual(self.summary_rows(), [])

    def test_recall_summary_exception_still_returns_evidence(self):
        self.client()
        self.write_session()
        canned = self.canned_found(SESSION_ID)
        with patch.object(conversations_module, "search_conversations", return_value=canned), \
                patch.object(conversation_summary, "use_or_create_summary", side_effect=RuntimeError("boom")):
            status, body = self.post("/v1/conversations/recall", {"query": "Excel 怎么入库"})
        self.assertEqual(status, 200)
        self.assertEqual(body["summary_status"], "failed")
        self.assertTrue(body["evidence"])

    def test_recall_can_skip_summary(self):
        self.client()
        self.write_session()
        canned = self.canned_found(SESSION_ID)
        with patch.object(conversations_module, "search_conversations", return_value=canned), \
                patch.object(conversation_summary, "use_or_create_summary") as summary:
            status, body = self.post("/v1/conversations/recall",
                                     {"query": "Excel 怎么入库", "include_summary": False})
        self.assertEqual(status, 200)
        self.assertEqual(body["summary_status"], "skipped")
        self.assertIsNone(body["summary"])
        self.assertTrue(body["evidence"])
        summary.assert_not_called()

    def test_recall_evidence_is_capped(self):
        self.client()
        self.write_session(turns=tuple((f"问题 {index}", f"回答 {index}") for index in range(8)))
        hits = [{"session_id": SESSION_ID, "turn_id": f"{SESSION_ID}:{index}", "turn_index": index,
                 "chunk_index": index, "primary_role": "answer", "score": 0.9 - index * 0.01,
                 "start_char": 0, "end_char": 10, "text": "本轮最终回答：" + "很长。" * 500}
                for index in range(8)]
        canned = self.canned_found(SESSION_ID, hits=hits)
        with patch.object(conversations_module, "search_conversations", return_value=canned), \
                patch.object(conversation_summary, "use_or_create_summary", return_value={
                    "status": "failed", "summary": None, "note": None, "source_revision": None,
                    "current_revision": None, "stale": False, "generated_at": None, "elapsed": 0.0}):
            status, body = self.post("/v1/conversations/recall", {"query": "Excel 怎么入库"})
        self.assertEqual(status, 200)
        self.assertLessEqual(len(body["evidence"]), file_service.RECALL_MAX_EVIDENCE)
        total = sum(len(item["user"]) + len(item["final_answer"]) + len(item["excerpt"])
                    for item in body["evidence"])
        self.assertLessEqual(total, file_service.RECALL_MAX_CHARS + len(body["evidence"]) * 2)
        self.assertTrue(any(item["truncated"] for item in body["evidence"]), "超长证据必须标出截断")

    def test_search_endpoint_reshapes_hits(self):
        self.client()
        canned = {"query": "Excel 怎么入库", "status": "found", "hits": [],
                  "candidates": [{"session_id": SESSION_ID, "title": "RAG 设计",
                                  "updated_at": "2026-02-02T10:00:00+08:00", "score": 0.93,
                                  "best_turn_id": "abc123:0", "primary_hit": True,
                                  "hits": [{"session_id": SESSION_ID, "turn_id": "abc123:0",
                                            "turn_index": 0, "chunk_index": 0, "primary_role": "answer",
                                            "primary_hit": True, "score": 0.93, "start_char": 3,
                                            "end_char": 20,
                                            "text": "本轮最终回答：Excel 用 summary 入库"}]}]}
        with patch.object(conversations_module, "search_conversations", return_value=canned):
            status, body = self.post("/v1/conversations/search",
                                     {"query": "Excel 怎么入库", "top_chunks": 5, "max_sessions": 2})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "found")
        self.assertTrue(body["candidates"][0]["primary_hit"], "primary_hit 必须透传给上层")
        hit = body["candidates"][0]["hits"][0]
        for field in ("session_id", "session_title", "session_updated_at", "turn_id", "turn_index",
                      "role", "chunk_index", "score", "primary_hit", "start_char", "end_char",
                      "text", "excerpt"):
            self.assertIn(field, hit)
        self.assertEqual(hit["role"], "answer")
        self.assertEqual(hit["session_title"], "RAG 设计")
        self.assertEqual(hit["session_updated_at"], "2026-02-02T10:00:00+08:00")


@needs_conversations
class TestConversationJobs(ServiceCase):
    """后台对话索引：入队 → 切块入库 → 切换 active，重复通知幂等，失败自动重试。"""

    def sync_session(self, session_id: str = SESSION_ID) -> dict:
        status, body = self.post("/v1/conversations/sync", {"session_id": session_id})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["sessions"], 1)
        return body["items"][0]

    def test_sync_builds_index_and_repeats_are_idempotent(self):
        self.client()
        self.write_session(turns=(("Excel 怎么入库？", "Excel 用模型给的 summary 入库"),
                                  ("PDF 呢？", "PDF 直接提取正文入库")))
        item = self.sync_session()
        self.assertTrue(item["queued"])
        self.assertFalse(item["skipped"])
        job_id, version_id = item["job_id"], item["version_id"]
        self.wait_for(lambda: (self.conversation_job(job_id) or {}).get("status") == "indexed",
                      message=f"对话任务没有完成：{self.conversation_job(job_id)}")
        detail = self.detail(SESSION_ID)
        self.assertTrue(detail["index"]["active"])
        self.assertEqual(detail["index"]["version_id"], version_id)
        self.assertEqual(detail["index"]["status"], "indexed")
        self.assertGreater(detail["index"]["chunk_count"], 0)
        count = conversations_module.vector_count()
        self.assertGreater(count, 0)

        again = self.sync_session()
        self.assertTrue(again["skipped"], "同一版本重复通知不能再建一遍")
        self.assertFalse(again["queued"])
        conn = rag_state.connect()
        try:
            self.assertEqual(rag_state.conversation_revision(conn, SESSION_ID), item["revision"])
            self.assertEqual(rag_state.active_conversation_version(conn, SESSION_ID)["id"], version_id)
        finally:
            conn.close()
        time.sleep(0.3)
        self.assertEqual(conversations_module.vector_count(), count, "重复 sync 不能新增向量")

        # 直接重复执行入库（模拟任务被重跑）：确定性 id 只覆盖，不追加
        conn = rag_state.connect()
        try:
            version = rag_state.get_conversation_version(conn, version_id)
        finally:
            conn.close()
        conversations_module.index_version(version, sessions_dir=self.sessions)
        self.assertEqual(conversations_module.vector_count(), count)

    def test_sync_all_queues_every_session(self):
        self.client()
        self.write_session(turns=(("问题一", "回答一"),))
        self.write_session(session_id="def456", title="另一个会话", turns=(("问题二", "回答二"),))
        status, body = self.post("/v1/conversations/sync", {"all": True})
        self.assertEqual(status, 200)
        self.assertGreaterEqual(body["sessions"], 2)
        self.assertEqual(body["broken"], [])
        self.assertGreaterEqual(body["queued"], 2)

    def test_startup_queues_existing_sessions(self):
        """首次启用：lifespan 会把已有会话全部入队补齐索引。"""
        self.write_session(turns=(("Excel 怎么入库？", "Excel 用 summary 入库"),))
        with patch.object(conversations_module, "sync_all",
                          wraps=conversations_module.sync_all) as sync_all:
            self.client()
        sync_all.assert_called_once()
        conn = rag_state.connect()
        try:
            self.assertEqual(rag_state.known_conversation_revisions(conn),
                             {f"{SESSION_ID}@{self.revision()}"})
        finally:
            conn.close()
        self.wait_for(lambda: conversations_module.vector_count() > 0, message="启动入队的会话没有建索引")
        self.wait_for(lambda: self.detail(SESSION_ID)["index"]["active"],
                      message="启动入队的会话没有切换成有效版本")

    def test_restart_after_sync_does_not_rebuild(self):
        self.client()
        self.write_session(turns=(("Excel 怎么入库？", "Excel 用 summary 入库"),))
        item = self.sync_session()
        self.wait_for(lambda: (self.conversation_job(item["job_id"]) or {}).get("status") == "indexed")
        count = conversations_module.vector_count()
        self.close_client()
        time.sleep(0.2)
        self.client()  # 重启：lifespan 会再走一遍补齐索引的流程
        time.sleep(0.5)
        self.assertEqual(conversations_module.vector_count(), count, "重启不能重复建向量")
        conn = rag_state.connect()
        try:
            self.assertEqual(rag_state.conversation_revision(conn, SESSION_ID), item["revision"])
        finally:
            conn.close()

    def test_index_failure_retries_then_fails_and_can_be_retried_manually(self):
        self.fast_retries()
        self.client()
        self.write_session()
        with patch.object(conversations_module, "index_version", side_effect=OSError("向量库暂时写不进去")):
            item = self.sync_session()
            job_id = item["job_id"]
            self.wait_for(lambda: (self.conversation_job(job_id) or {}).get("status") == "failed",
                          message=f"没有走到终态：{self.conversation_job(job_id)}")
        job = self.conversation_job(job_id)
        self.assertEqual(job["attempts"], 3, "首次执行 + 最多两次自动重试")
        self.assertEqual(job["round"], 1)
        self.assertIn("向量库暂时写不进去", job["last_error"])
        conn = rag_state.connect()
        try:
            snapshot = rag_state.conversation_job_snapshot(conn, 20)
        finally:
            conn.close()
        self.assertIn(job_id, [row["job_id"] for row in snapshot], "失败任务必须出现在快照里")

        conn = rag_state.connect()
        try:
            fresh = rag_state.retry_conversation_job(conn, job_id)
        finally:
            conn.close()
        self.assertEqual(fresh["round"], 2)
        self.assertNotEqual(fresh["job_id"], job_id)
        self.assertEqual(self.conversation_job(job_id)["status"], "failed", "旧一轮保持终态")
        file_service.worker.wake()
        self.wait_for(lambda: (self.conversation_job(fresh["job_id"]) or {}).get("status") == "indexed",
                      message=f"手动重试没有完成：{self.conversation_job(fresh['job_id'])}")
        self.assertTrue(self.detail(SESSION_ID)["index"]["active"])

    def test_retry_route_creates_the_next_round_for_a_failed_conversation_job(self):
        """真正走服务入口 POST /v1/jobs/{job_id}/retry：cjob_... 必须按对话任务重试。"""
        self.fast_retries()
        self.client()
        self.write_session()
        with patch.object(conversations_module, "index_version", side_effect=OSError("向量库暂时写不进去")):
            item = self.sync_session()
            job_id = item["job_id"]
            self.wait_for(lambda: (self.conversation_job(job_id) or {}).get("status") == "failed",
                          message=f"没有走到终态：{self.conversation_job(job_id)}")

        status, body = self.post(f"/v1/jobs/{job_id}/retry", {})
        self.assertEqual(status, 200, body)
        self.assertTrue(str(body["job_id"]).startswith("cjob_"), body)
        self.assertNotEqual(body["job_id"], job_id)
        self.assertEqual(body["round"], 2, "重试必须新建下一轮，而不是复用失败的那一轮")
        self.assertEqual(body["status"], "queued")
        fresh = self.conversation_job(body["job_id"])
        self.assertIsNotNone(fresh, "新建的对话任务必须真的落在 conversation_jobs 里")
        self.assertEqual(fresh["session_id"], SESSION_ID)
        self.assertEqual(fresh["version_id"], item["version_id"])
        self.assertEqual(self.conversation_job(job_id)["status"], "failed", "旧一轮保持终态")

        # 重试成功后要唤醒执行器：新任务应当自动执行完，不用等下一轮轮询
        self.wait_for(lambda: (self.conversation_job(body["job_id"]) or {}).get("status") == "indexed",
                      message=f"重试的新任务没有执行：{self.conversation_job(body['job_id'])}")
        self.assertTrue(self.detail(SESSION_ID)["index"]["active"])

    def test_file_jobs_are_claimed_before_conversation_jobs(self):
        self.write_session()
        conn = rag_state.connect()
        try:
            doc = rag_state.get_or_create_document(conn, "https://e.org/r.pdf", "pdf", "r.pdf")
            version, _ = rag_state.add_version(conn, doc_id=doc["doc_id"], sha256="c" * 64, kind="pdf",
                                               local_path="p", size=1, file_name="r.pdf", final_url="u",
                                               summary=None, page_count=1, empty_pages=[], sheet_names=None)
            file_job, _ = rag_state.ensure_job(conn, doc["doc_id"], version["id"], "pdf")
            rag_state.upsert_conversation(conn, session_id=SESSION_ID, title="RAG 设计", turn_count=1,
                                          created_at="2026-02-02T10:00:00+08:00",
                                          updated_at="2026-02-02T10:00:00+08:00")
            conversation_version, _ = rag_state.ensure_conversation_version(
                conn, SESSION_ID, self.revision(), 1, "RAG 设计")
            conversation_job, _ = rag_state.ensure_conversation_job(conn, SESSION_ID,
                                                                   conversation_version["id"])
            conn.commit()
        finally:
            conn.close()
        worker = file_service.IngestionWorker()  # 不启动：只验证取任务的优先级
        claimed = worker._claim()
        self.assertEqual(claimed["_queue"], "file")
        self.assertEqual(claimed["job_id"], file_job["job_id"])
        second = worker._claim()
        self.assertEqual(second["_queue"], "conversation")
        self.assertEqual(second["job_id"], conversation_job["job_id"])

    def test_conversation_task_switches_active_and_drops_old_vectors(self):
        self.client()
        self.write_session(turns=(("第一轮问题", "第一轮回答"),))
        first = self.sync_session()
        self.wait_for(lambda: (self.conversation_job(first["job_id"]) or {}).get("status") == "indexed")
        first_revision = first["revision"]

        self.write_session(turns=(("第一轮问题", "第一轮回答（已修改）"), ("第二轮问题", "第二轮回答")))
        second = self.sync_session()
        self.assertNotEqual(second["revision"], first_revision)
        self.wait_for(lambda: (self.conversation_job(second["job_id"]) or {}).get("status") == "indexed")
        detail = self.detail(SESSION_ID)
        self.assertEqual(detail["index"]["revision"], second["revision"])
        self.assertTrue(detail["index"]["active"])
        conn = rag_state.connect()
        try:
            self.assertEqual(rag_state.conversation_revision(conn, SESSION_ID), second["revision"])
            old = rag_state.version_by_revision(conn, SESSION_ID, first_revision)
        finally:
            conn.close()
        self.assertEqual(old["status"], "superseded")
        remaining = conversations_module._ids_of(SESSION_ID, first_revision)
        self.assertEqual(remaining, [], "被替换版本的旧向量必须删掉")


if __name__ == "__main__":
    unittest.main()
