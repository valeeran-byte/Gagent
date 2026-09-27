"""跨会话回忆的真实模型端到端用例（独立验收，不复用实现者的假 embedding）。

每个用例都用真实 multilingual-e5-small 建索引、真实 Chroma 检索，只把资料目录和
sessions 目录挪到临时目录（patch rag_state.DATA_ROOT / session_store.SESSIONS_DIR），
项目自己的 sessions/ 只读、rag_data/ 不写。模型缺失时整体 skip（不静默通过）；
模型存在时一个用例都不跳过。

摘要用例不联网：用桩 llm_factory 走真实 `conversation_summary.use_or_create_summary`
与真实 `rag.service.recall_conversation`，验证生成/复用/失效重生成与失败降级。
真实 API 的摘要质量在 scripts/acceptance_conversations.py 里单独验收。
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import conversation_summary
import session_store
from rag import conversations as conv
from rag import state as rag_state
from rag import vectors as rag_store

MODEL_READY = rag_store.model_ready()
MODEL_NOTE = f"真实 embedding 模型缺失或不全：{rag_store.model_dir()}"

SCRATCH = pathlib.Path(tempfile.gettempdir()) / f"gagent_recall_e2e_{os.getpid()}"
shutil.rmtree(SCRATCH, ignore_errors=True)

PROJECT = pathlib.Path(__file__).resolve().parents[1]
REAL_SESSIONS = PROJECT / "sessions"
STUB_SETTINGS = {"API_KEY": "sk-e2e-canary-not-a-real-key", "MODEL": "stub-model",
                 "BASE_URL": "http://127.0.0.1:1/v1"}
CANARY = STUB_SETTINGS["API_KEY"]


def turn(user: str, answer: str) -> dict:
    return {"user": user, "final_answer": answer}


def stub_summary(turn_ids=("x:0",), decision="Excel 用 summary 入库", superseded=None) -> str:
    payload = {"topic": "桩摘要",
               "user_requirements": [{"text": "用户要求", "turn_ids": list(turn_ids)}],
               "confirmed_decisions": [{"text": decision, "turn_ids": list(turn_ids)}],
               "assistant_proposals": [], "open_questions": [],
               "superseded_decisions": ([{"text": superseded, "turn_ids": list(turn_ids)}]
                                        if superseded else [])}
    return json.dumps(payload, ensure_ascii=False)


class StubLLM:
    """桩对话模型：记录提示词，返回固定 JSON；可让指定次数直接抛异常。"""

    def __init__(self, reply=None, fail_times: int = 0):
        self.reply = stub_summary() if reply is None else reply
        self.fail_times = fail_times
        self.prompts: list[str] = []

    def __call__(self, settings, timeout=None):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("桩模型不可用")
        return SimpleNamespace(content=self.reply)


@unittest.skipUnless(MODEL_READY, MODEL_NOTE)
class RealModelCase(unittest.TestCase):
    """真实模型 + 隔离目录；语料由各用例自己写。"""

    def setUp(self):
        self.root = SCRATCH / f"t_{uuid.uuid4().hex[:10]}"
        self.sessions = self.root / "sessions"
        self.sessions.mkdir(parents=True, exist_ok=True)
        self.store = session_store.SessionStore(self.sessions)
        self._patch(rag_state, "DATA_ROOT", self.root / "data")
        # 模型目录不跟随 DATA_ROOT：rag_store.MODELS_ROOT 是独立常量，真实模型仍在项目里。
        self._patch(session_store, "SESSIONS_DIR", self.sessions)
        self._reset_vectors()
        rag_state.ensure_dirs()
        self.addCleanup(self.wipe)

    def _patch(self, target, name, value):
        patcher = patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _reset_vectors(self):
        rag_store._collection = None
        rag_store._client = None
        rag_store._collection_root = None
        conv._collection = None
        conv._collection_root = None

    def wipe(self):
        self._reset_vectors()
        shutil.rmtree(self.root, ignore_errors=True)

    # ---------- 语料与索引 ----------

    def write_session(self, session_id, title, turns, updated_at="2026-03-01T09:00:00+08:00"):
        session = {"id": session_id, "title": title, "created_at": updated_at,
                   "updated_at": updated_at, "turns": [turn(user, answer) for user, answer in turns]}
        self.store.write(session)
        return session

    def run_jobs(self) -> int:
        """复用真实执行器的口径：claim → index → 切换 active → 删被替换版本的向量。"""
        processed = 0
        conn = rag_state.connect()
        try:
            while True:
                job = rag_state.claim_conversation_job(conn, float("inf"))
                if job is None:
                    break
                version = rag_state.get_conversation_version(conn, job["version_id"])
                count = conv.index_version(version, sessions_dir=self.sessions)
                conv.set_version_active(version["session_id"], version["revision"], True)
                result = rag_state.finish_conversation_job(conn, job["job_id"], ok=True)
                superseded = result.get("superseded")
                if superseded:
                    old = rag_state.get_conversation_version(conn, superseded)
                    if old is not None:
                        conv.delete_version_vectors(old["session_id"], old["revision"])
                processed += 1 if count else 0
        finally:
            conn.close()
        return processed

    def sync_and_index(self) -> dict:
        result = conv.sync_all(sessions_dir=self.sessions)
        self.run_jobs()
        return result

    def search(self, query, **options):
        options.setdefault("sessions_dir", self.sessions)
        return conv.search_conversations(query, **options)

    def call_recall(self, query, **payload):
        from rag.service import ConversationRecall, recall_conversation
        request = ConversationRecall(query=query, **payload)
        return recall_conversation(request)


class TestRealSessions(RealModelCase):
    """用项目真实 sessions/：旧会话自动建索引、能找回以前讨论。"""

    def copy_real_sessions(self) -> list[dict]:
        sessions = []
        for path in sorted(REAL_SESSIONS.glob("*.json")):
            if path.name == session_store.STATE_FILENAME:
                continue
            session = session_store.SessionStore(REAL_SESSIONS).read(path.stem)
            self.store.write(session)
            sessions.append(session)
        return sessions

    def test_real_sessions_are_indexed_and_recalled(self):
        sessions = self.copy_real_sessions()
        with_turns = [item for item in sessions if item["turns"]]
        self.assertGreaterEqual(len(with_turns), 3, "项目 sessions/ 里应有带轮次的真实会话")
        result = self.sync_and_index()
        self.assertEqual(result["broken"], [])
        conn = rag_state.connect()
        try:
            self.assertEqual(conv.pending_count(conn), 0)
            self.assertEqual(conv.failed_count(conn), 0)
            self.assertEqual(conv.indexed_count(conn), len(with_turns))
        finally:
            conn.close()
        self.assertGreater(conv.vector_count(), 0)

        found = self.search("2025 年代表 LPL 出征全球总决赛的四支队伍是哪几个")
        # 两个真实会话都是 LPL 话题，这条很可能被判 ambiguous：要求 top1 是正确会话，
        # 状态只能是 found 或 ambiguous（ambiguous 也要给候选标题和命中片段）。
        self.assertIn(found["status"], ("found", "ambiguous"))
        self.assertEqual(found["candidates"][0]["session_id"], "99c1e6")
        if found["status"] == "ambiguous":
            self.assertGreaterEqual(len(found["candidates"]), 2)
            for item in found["candidates"]:
                self.assertTrue(item["title"])
                self.assertTrue(item["hits"])

        found_zh = self.search("Transformer 论文里多头注意力有什么作用")
        self.assertEqual(found_zh["status"], "found")
        self.assertEqual(found_zh["candidates"][0]["session_id"], "be0fa5")

        evidence = conv.recall_candidates(found_zh["candidates"][0]["session_id"],
                                          turn_ids=[hit["turn_id"]
                                                    for hit in found_zh["candidates"][0]["hits"]],
                                          sessions_dir=self.sessions)
        self.assertTrue(evidence)
        for item in evidence:
            self.assertTrue(item["turn_id"].startswith("be0fa5:"))
            self.assertTrue(item["user"] or item["final_answer"])

    def test_real_sessions_unrelated_question_is_not_found(self):
        self.copy_real_sessions()
        self.sync_and_index()
        for query in ("红烧肉要炖多久才能软烂", "吉他换弦的步骤是什么"):
            result = self.search(query)
            self.assertEqual(result["status"], "not_found", query)
            # not_found 仍会报出高于噪声线的"最接近候选"，但都不能达到判定门槛
            for item in result["candidates"]:
                self.assertLess(item["score"], conv.CONVERSATION_MIN_SCORE, query)


class TestRetrievalQuality(RealModelCase):
    """真实模型的换说法、无关拒绝、近似重复判定。"""

    def setUp(self):
        super().setUp()
        self.excel = self.write_session("excel01", "文件入库方案", [
            ("我们要给文件做 RAG 入库，PDF 和 Excel 分别怎么处理？",
             "PDF 直接提取正文后切块入库；Excel 先让模型读网页介绍生成 summary，再按 summary 建向量。"),
            ("Excel 那块确定用 summary 了吗？", "确定：Excel 用模型提供的 summary 入库。")])
        self.lpl = self.write_session("lpl01", "LPL 世界赛名单", [
            ("代表 LPL 去打全球总决赛的战队有哪几个？",
             "四支：AL、BLG、TES、iG，其中 AL 是一号种子。")])
        self.refer = self.write_session("refer01", "备份策略确认", [
            ("备份策略有两种方案：第一种是每天全量备份，第二种是每周全量加每天增量。你建议哪种？",
             "建议第二种：每周全量 + 每天增量，恢复更快。"),
            ("用第二种", "好，按第二种执行：每周全量备份，工作日每天做增量备份。"),
            ("按这个做", "已按第二种方案落地。")])
        self.budgeta = self.write_session("budgeta", "季度预算方案 A",
                                          [("季度预算最后定哪一版？", "季度预算最后定 A 版：控制成本优先。")])
        self.budgetb = self.write_session("budgetb", "季度预算方案 B",
                                          [("季度预算最后定哪一版？", "季度预算最后定 A 版：控制成本优先。")])
        self.sync_and_index()

    def test_paraphrases_hit_the_right_session(self):
        cases = [("Excel 文件入库最后决定用哪种方式", "excel01"),
                 ("PDF 入库需要先让模型写 summary 吗", "excel01"),
                 ("代表 LPL 去全球总决赛的战队有哪几个", "lpl01"),
                 ("备份策略最后按哪个方案执行", "refer01")]
        for query, expected in cases:
            result = self.search(query)
            self.assertEqual(result["status"], "found", query)
            self.assertEqual(result["candidates"][0]["session_id"], expected, query)
            self.assertGreaterEqual(result["candidates"][0]["score"], conv.CONVERSATION_MIN_SCORE, query)

    def test_unrelated_questions_are_not_found(self):
        for query in ("红烧肉要炖多久才软烂", "怎么给吉他换弦", "猫咪多久洗一次澡"):
            result = self.search(query)
            self.assertEqual(result["status"], "not_found", query)
            for item in result["candidates"]:
                self.assertLess(item["score"], conv.CONVERSATION_MIN_SCORE, query)

    def test_near_duplicate_sessions_are_ambiguous(self):
        result = self.search("季度预算最后定的是哪一版")
        self.assertEqual(result["status"], "ambiguous")
        # 两个近重复会话必须都在候选里，且不混成一份历史；
        # 允许出现第三个沾边会话（refer01 也谈"最后定的哪个方案"），但不能取代这两个。
        ids = {item["session_id"] for item in result["candidates"]}
        self.assertTrue({"budgeta", "budgetb"} <= ids, ids)
        for item in result["candidates"]:
            self.assertTrue(item["title"])
            self.assertTrue(item["hits"])

    def test_referent_turn_carries_previous_turn_context(self):
        """指代能靠"紧邻上一轮原文"还原：片段只带一轮上下文，不累积更早轮次。"""
        session = self.store.read("refer01")
        records = conv.build_records(session)
        # 只带紧邻上一轮：按这个做（第 2 轮）应能看到上一轮的"第二种"，
        # 第 0 轮的那个"第一种"属于更早轮次，不该被塞进来（累积会挤出本轮主体）。
        for index, marker, referent in ((1, "用第二种", "第一种"), (2, "按这个做", "第二种")):
            chunks = [item for item in records if item["turn_index"] == index]
            text = "\n".join(item["text"] for item in chunks)
            self.assertIn(marker, text)
            self.assertIn("上一轮用户问题：", text)
            self.assertIn(referent, text, "指代必须能靠紧邻上一轮原文还原")
            self.assertIn("本轮用户问题：", text)
            self.assertEqual(text.count("上一轮用户问题："), len(chunks),
                             "每个片段只带一行上一轮上下文，不能累积多轮")
        result = self.search("备份策略最后按哪个方案执行")
        self.assertEqual((result["candidates"][0])["session_id"], "refer01")


class TestRealSummaryAndService(RealModelCase):
    """真实服务函数 + 桩摘要模型：缓存复用、版本失效、失败降级、凭据不外泄。"""

    def setUp(self):
        super().setUp()
        self.session = self.write_session("cache01", "缓存方案定稿", [
            ("缓存层用 Redis 还是进程内 LRU？", "建议先用进程内 LRU。"),
            ("不用 LRU 了，改成本地 SQLite 缓存", "好，缓存改用本地 SQLite，不再用进程内 LRU。")])
        self.sync_and_index()
        self.llm = StubLLM(reply=stub_summary(turn_ids=("cache01:1",),
                                              decision="缓存改用本地 SQLite",
                                              superseded="原先建议的进程内 LRU"))

    def use_stub(self):
        """只在需要桩模型时替换工厂：失败路径必须走真实 _default_llm_factory。"""
        from contextlib import contextmanager

        @contextmanager
        def factory_patch():
            with patch.object(conversation_summary, "_default_llm_factory",
                              lambda settings, timeout: self.llm):
                yield
        return factory_patch()

    def test_summary_generated_reused_then_regenerated_after_update(self):
        with self.use_stub():
            first = self.call_recall("缓存层最后确定用哪个方案", include_summary=True,
                                     api_settings=STUB_SETTINGS)
            self.assertEqual(first["status"], "found")
            self.assertEqual(first["session"]["id"], "cache01")
            self.assertEqual(first["summary_status"], "generated")
            self.assertEqual(len(self.llm.prompts), 1)
            evidence_ids = {item["turn_id"] for item in first["evidence"]}
            self.assertIn("cache01:1", evidence_ids)
            self.assertTrue(any(item.get("turn_ids") and set(item["turn_ids"]) <= evidence_ids
                                for item in first["summary"]["confirmed_decisions"]))

            second = self.call_recall("缓存层最后确定用哪个方案", include_summary=True,
                                      api_settings=STUB_SETTINGS)
            self.assertEqual(second["summary_status"], "cached")
            self.assertEqual(second["summary"], first["summary"])
            self.assertEqual(len(self.llm.prompts), 1, "缓存命中不能再次请求模型")

            data = self.store.read("cache01")
            data["turns"].append(turn("缓存失效策略怎么定？", "本地 SQLite 缓存加 TTL 和版本号双重失效。"))
            self.store.write(data)
            self.sync_and_index()
            third = self.call_recall("缓存层最后确定用哪个方案", include_summary=True,
                                     api_settings=STUB_SETTINGS)
            self.assertEqual(third["summary_status"], "generated")
            self.assertNotEqual(third["summary_source_revision"], first["summary_source_revision"])
            self.assertEqual(len(self.llm.prompts), 2)

    def test_summary_failure_still_returns_evidence(self):
        # 真实 _default_llm_factory + 死端口：走真实失败路径，不注入桩
        broken = self.call_recall("缓存层最后确定用哪个方案", include_summary=True,
                                  api_settings={"API_KEY": CANARY, "MODEL": "stub-model",
                                                "BASE_URL": "http://127.0.0.1:1/v1"})
        self.assertEqual(broken["status"], "found")
        self.assertEqual(broken["summary_status"], "failed")
        self.assertIsNone(broken["summary"])
        self.assertTrue(broken["evidence"])

        self.llm.fail_times = 1
        with self.use_stub():
            crashed = self.call_recall("缓存层最后确定用哪个方案", include_summary=True,
                                       api_settings=STUB_SETTINGS)
        self.assertEqual(crashed["status"], "found")
        self.assertEqual(crashed["summary_status"], "failed")
        self.assertTrue(crashed["evidence"])

        skipped = self.call_recall("缓存层最后确定用哪个方案", include_summary=False)
        self.assertEqual(skipped["summary_status"], "skipped")
        self.assertTrue(skipped["evidence"])

    def test_exclude_current_session_and_explicit_target(self):
        found = self.call_recall("缓存层最后确定用哪个方案", include_summary=False)
        self.assertEqual(found["session"]["id"], "cache01")

        excluded = self.search("缓存层最后确定用哪个方案", exclude_sessions=["cache01"])
        self.assertNotIn("cache01", [item["session_id"] for item in excluded["candidates"]])

        # 显式指定会话：即使它与查询主题无关，也以指定会话为准并给出原文
        explicit = self.call_recall("LPL 全球总决赛的战队有哪些", session_id="cache01",
                                    include_summary=False)
        self.assertEqual(explicit["status"], "found")
        self.assertEqual(explicit["session"]["id"], "cache01")
        self.assertTrue(explicit["evidence"])

    def test_credentials_never_reach_vectors_cache_logs_or_response(self):
        response = self.call_recall("缓存层最后确定用哪个方案", include_summary=True,
                                    api_settings={"API_KEY": CANARY, "MODEL": "stub-model",
                                                  "BASE_URL": "http://127.0.0.1:1/v1"})
        self.assertNotIn(CANARY, json.dumps(response, ensure_ascii=False))
        root = pathlib.Path(rag_state.DATA_ROOT)
        leaked = []
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                path = pathlib.Path(dirpath) / name
                try:
                    if CANARY.encode("utf-8") in path.read_bytes():
                        leaked.append(str(path.relative_to(root)))
                except OSError:
                    continue
        self.assertEqual(leaked, [], "凭据不得写进任何落盘文件")
        conn = rag_state.connect()
        try:
            rows = conn.execute("SELECT summary_json FROM conversation_summaries").fetchall()
            events = conn.execute("SELECT * FROM events").fetchall()
        finally:
            conn.close()
        self.assertEqual([row for row in rows if CANARY in row["summary_json"]], [])
        self.assertEqual([dict(row) for row in events
                          if CANARY in json.dumps(dict(row), ensure_ascii=False)], [])


class TestChunkingAndVersions(RealModelCase):
    """真实 tokenizer 下的切块预算、无静默截断、旧版本向量失效。"""

    def test_long_turn_keeps_body_and_respects_limit(self):
        long_user = "背景很长：" + ("这一段描述业务约束和验收要求。" * 60)
        long_answer = "结论开头。" + ("中段分析内容，包含具体参数与取舍。" * 120) + "最后一段的收尾结论。"
        session = {"id": "chunkcheck", "title": "很长的标题" + "标" * 40,
                   "created_at": "", "updated_at": "",
                   "turns": [turn("上一轮问题" + "问" * 80, "上一轮回答" + "答" * 200),
                             turn(long_user, long_answer)]}
        records = conv.build_records(session)
        this_turn = [item for item in records if item["turn_index"] == 1]
        self.assertGreater(len(this_turn), 1, "长回答必须继续分块")
        for item in records:
            self.assertLessEqual(conv._tokens(item["text"]), conv.CHUNK_LIMIT_TOKENS)
        normalize = lambda text: "".join(str(text).split())  # noqa: E731
        joined = normalize("".join(item["text"] for item in this_turn))
        # 切块会给每个句子加一行角色标签，所以按"整句出现次数"核对有没有静默丢内容：
        # 用户那句重复 60 次、回答中段那句重复 120 次，一个都不能少。
        self.assertIn(normalize("背景很长："), joined)
        self.assertEqual(joined.count("这一段描述业务约束和验收要求"), 60, "长 user 被静默截断")
        self.assertEqual(joined.count("中段分析内容，包含具体参数与取舍"), 120, "长回答被静默截断")
        self.assertIn("结论开头。", joined)
        self.assertIn("最后一段的收尾结论。", joined)
        for item in this_turn:
            self.assertTrue("本轮用户问题：" in item["text"] or "本轮最终回答：" in item["text"],
                            "上下文不能把本轮主体挤出片段")
            context = [line for line in item["text"].splitlines()
                       if line.startswith(("会话标题：", "上一轮用户问题：", "上一轮回答中的邻近内容："))]
            self.assertLessEqual(sum(conv._tokens(line.split("：", 1)[1]) for line in context),
                                 conv.TITLE_BUDGET_TOKENS + conv.CONTEXT_TOTAL_BUDGET_TOKENS)

    def test_editing_an_old_turn_invalidates_old_vectors(self):
        # 旧轮次是"缓存"，改完只剩"部署"：旧话题在新版本里一个字都不剩，
        # 这样"还能不能检索到旧内容"就没有语义歧义。
        session = self.write_session("deploy01", "部署环境",
                                     [("缓存层用 Redis 还是进程内 LRU？",
                                       "缓存层最后定为进程内 LRU，够用。")])
        self.sync_and_index()
        old_revision = session_store.revision_of(session)
        before = self.search("缓存层用进程内 LRU 还是 Redis")
        self.assertEqual(before["status"], "found")
        self.assertEqual(before["candidates"][0]["session_id"], "deploy01")

        changed = dict(session)
        changed["turns"] = [turn("部署环境最后用什么编排？",
                                 "部署环境最后改为 Kubernetes 托管集群。")]
        self.store.write(changed)
        conv.sync_session(changed, sessions_dir=self.sessions)
        self.run_jobs()
        self.assertEqual(conv._ids_of("deploy01", old_revision), [], "旧版本向量必须被删掉")
        after = self.search("缓存层用进程内 LRU 还是 Redis")
        self.assertEqual(after["status"], "not_found", "旧轮次内容已不存在，不能还能回忆出来")
        self.assertEqual(after["hits"], [], "not_found 时不能给出命中片段")
        # 没到门槛时仍允许报出高于噪声线的近邻候选（规范要求），但都不能达到判定门槛
        for item in after["candidates"]:
            self.assertLess(item["score"], conv.CONVERSATION_MIN_SCORE)
        fresh = self.search("部署环境改用 Kubernetes 了吗")
        self.assertEqual(fresh["status"], "found")
        self.assertEqual(fresh["candidates"][0]["session_id"], "deploy01")

    def test_repeated_sync_does_not_duplicate_vectors(self):
        self.write_session("ingest01", "文件入库方案",
                           [("Excel 怎么入库？", "Excel 用模型给的 summary 入库。")])
        self.sync_and_index()
        before = conv.vector_count()
        for _ in range(3):
            result = conv.sync_session("ingest01", sessions_dir=self.sessions)
            self.assertTrue(result["skipped"], "重复通知应被幂等跳过")
        self.run_jobs()
        self.assertEqual(conv.vector_count(), before)
        conn = rag_state.connect()
        try:
            versions = conn.execute("SELECT COUNT(*) AS n FROM conversation_versions"
                                    " WHERE session_id='ingest01'").fetchone()["n"]
            jobs = conn.execute("SELECT COUNT(*) AS n FROM conversation_jobs").fetchone()["n"]
        finally:
            conn.close()
        self.assertEqual((versions, jobs), (1, 1))


if __name__ == "__main__":
    unittest.main()
