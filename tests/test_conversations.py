"""对话记忆的切块、索引、版本重建、检索聚合与阈值判定。

用可复现的假 embedding（字符二元组哈希）验证逻辑；真实 e5 模型的检索质量在
scripts/acceptance_conversations.py 里用项目真实会话和标注查询集验收。
不联网、不读写项目自己的 rag_data/ 与 sessions/。
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import tempfile
import unittest
import uuid
import zlib
from unittest.mock import patch

import numpy as np

from rag import conversations as conv
from rag import state as rag_state
from rag import vectors as rag_store
from session_store import SessionStore, revision_of

# 带上进程号：多人/多进程同时跑测试时不会互删临时目录（Chroma 的目录删不干净也要能重来）。
SCRATCH = pathlib.Path(tempfile.gettempdir()) / f"gagent_conversation_tests_{os.getpid()}"
shutil.rmtree(SCRATCH, ignore_errors=True)

CHARS = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")


class CharModel:
    """字符三元组哈希的假 embedding：共享汉字越多越相似，分数尺度与真实 e5 不同，
    因此本文件里的阈值断言都显式传 min_score，不依赖默认阈值。"""

    def __init__(self, dimension: int = 2048):
        self.dimension = dimension
        self.max_seq_length = 512
        self.passage_calls: list[list[str]] = []
        self.query_calls: list[list[str]] = []

    def tokenizer(self, texts, add_special_tokens=True):
        return {"input_ids": [[0] + [1] * len(text) for text in texts]}

    def encode(self, texts, **kwargs):
        if texts and str(texts[0]).startswith("query: "):
            self.query_calls.append(list(texts))
        else:
            self.passage_calls.append(list(texts))
        return np.array([self._vector(text) for text in texts], dtype="float32")

    def _vector(self, text: str) -> np.ndarray:
        body = "".join(str(text).split())
        body = body.split(": ", 1)[-1]
        vector = np.zeros(self.dimension, dtype="float32")
        for index in range(max(1, len(body) - 2)):
            # 用 crc32 而不是内置 hash()：Python 的字符串 hash 每次进程都加盐，
            # 会让同一个用例在不同次运行里得到不同分数，阈值断言就变成随机失败。
            vector[zlib.crc32(body[index:index + 3].encode("utf-8")) % self.dimension] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


class ConversationCase(unittest.TestCase):
    """隔离的 rag_data 与 sessions 目录 + 假 embedding。"""

    def setUp(self):
        self.root = SCRATCH / f"t_{uuid.uuid4().hex[:10]}"
        self.sessions = self.root / "sessions"
        self.store = SessionStore(self.sessions)
        self.model = CharModel()
        self._patch(rag_state, "DATA_ROOT", self.root / "data")
        self._patch(rag_store, "MODELS_ROOT", self.root / "models")
        self._patch(rag_store, "_model", self.model)
        self._patch(rag_store, "_collection", None)
        self._patch(rag_store, "_collection_root", None)
        self._patch(conv, "_collection", None)
        self._patch(conv, "_collection_root", None)
        self.root.mkdir(parents=True, exist_ok=True)
        rag_state.ensure_dirs()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _patch(self, module, name, value):
        patcher = patch.object(module, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    # ---------- 工具 ----------

    def session(self, title, turns, session_id=None):
        data = {"id": session_id or f"s{uuid.uuid4().hex[:6]}", "title": title,
                "created_at": "2026-01-01T00:00:00+08:00",
                "updated_at": "2026-01-01T00:00:00+08:00",
                "turns": [{"user": user, "final_answer": answer} for user, answer in turns]}
        self.store.write(data)
        return data

    def run_jobs(self) -> int:
        """模拟后台执行器：把到期任务跑完，返回处理的任务数。"""
        processed = 0
        conn = rag_state.connect()
        try:
            while True:
                job = rag_state.claim_conversation_job(conn, float("inf"))
                if job is None:
                    break
                version = rag_state.get_conversation_version(conn, job["version_id"])
                try:
                    count = conv.index_version(version, sessions_dir=self.sessions)
                    conv.set_version_active(version["session_id"], version["revision"], True)
                    result = rag_state.finish_conversation_job(conn, job["job_id"], ok=True)
                    superseded = result.get("superseded")
                    old = rag_state.get_conversation_version(conn, superseded) if superseded else None
                    if old is not None:
                        conv.delete_version_vectors(old["session_id"], old["revision"])
                except Exception as exc:
                    rag_state.finish_conversation_job(conn, job["job_id"], ok=False, error=str(exc))
                    raise
                processed += 1
        finally:
            conn.close()
        return processed

    def sync_and_index(self) -> None:
        conv.sync_all(sessions_dir=self.sessions)
        self.run_jobs()

    def search(self, query, **options):
        # 假 embedding 的分数尺度与真实 e5 完全不同（满分大约 0.4，不是 0.9+），
        # 所以这些用例显式传一个"够宽松"的门槛来验证流程；真实阈值由
        # scripts/acceptance_conversations.py 用真模型和真实会话校准。
        options.setdefault("min_score", 0.15)
        options.setdefault("sessions_dir", self.sessions)
        return conv.search_conversations(query, **options)


def records_for_context(session):
    """切块结果里还需要看正文/上下文比例的用例共用。"""
    return conv.build_records(session)


class TestChunking(ConversationCase):
    def test_turn_keeps_user_and_answer_with_context(self):
        session = self.session("RAG 设计", [
            ("我们要做文件入库，PDF 怎么处理？", "PDF 直接提取正文入库。" * 30),
            ("不是，是 Excel 需要 summary，PDF 直接用内容", "好的，Excel 用 summary 入库。"),
        ])
        records = conv.build_records(session)
        second = [item for item in records if item["turn_index"] == 1]
        self.assertTrue(second)
        # 本轮 user 与 final_answer 在同一条记录里，且带着会话标题与上一轮原文
        first = second[0]
        self.assertIn("会话标题：RAG 设计", first["text"])
        self.assertIn("上一轮用户问题：我们要做文件入库", first["text"])
        self.assertIn("本轮用户问题：不是，是 Excel 需要 summary", first["text"])
        self.assertIn("本轮最终回答", first["text"])
        self.assertEqual(first["turn_id"], f"{session['id']}:1")
        self.assertEqual(first["chunk_id"], f"{session['id']}:1:{first['chunk_index']}")
        # 角色与原文位置都标了出来
        self.assertIn("answer", first["roles"])
        self.assertEqual(first["primary_role"], "answer")
        self.assertLessEqual(first["end_char"], len("好的，Excel 用 summary 入库。"))

    def test_every_chunk_within_model_limit(self):
        session = self.session("长会话", [("很长的提问。" * 60, "很长的回答。" * 400)])
        records = conv.build_records(session)
        self.assertGreater(len(records), 2)
        for item in records:
            self.assertLessEqual(conv._tokens(item["text"]), conv.CHUNK_LIMIT_TOKENS)

    def test_long_answer_is_not_silently_truncated(self):
        body = "第一段结论。" + "中段内容。" * 300 + "最后一段的收尾句子。"
        session = self.session("长回答", [("问题", body)])
        records = conv.build_records(session)
        joined = "".join(item["text"] for item in records)
        for fragment in ("第一段结论。", "最后一段的收尾句子。"):
            self.assertIn(fragment, joined)

    def test_different_turns_do_not_share_one_chunk(self):
        session = self.session("多轮", [("第一轮问题", "第一轮回答" * 40),
                                       ("第二轮问题", "第二轮回答" * 40)])
        records = conv.build_records(session)
        first = [item for item in records if item["turn_index"] == 0]
        second = [item for item in records if item["turn_index"] == 1]
        self.assertTrue(first and second)
        # 上一轮的原文可以作为上下文出现在下一轮的片段里，但两轮的主体不会挤进同一条记录
        for item in first:
            self.assertNotIn("本轮用户问题：第二轮问题", item["text"])
        for item in second:
            self.assertNotIn("本轮用户问题：第一轮问题", item["text"])
        self.assertTrue(all(item["turn_id"].endswith(":0") for item in first))
        self.assertTrue(all(item["turn_id"].endswith(":1") for item in second))

    def test_rewritten_text_is_never_injected(self):
        """上下文只能来自真实历史：片段里出现的内容必须能在原文里找到。"""
        session = self.session("原文检查", [("原始提问甲", "原始回答乙"), ("追问丙", "回答丁")])
        records = conv.build_records(session)
        haystack = "".join(turn["user"] + turn["final_answer"] for turn in session["turns"]) + session["title"]
        for item in records:
            for line in item["text"].splitlines():
                body = line.split("：", 1)[1] if "：" in line else line
                self.assertIn(body[:12], haystack)

    def test_context_is_only_the_immediately_previous_turn(self):
        """每个片段只带紧邻上一轮的上下文，不能把更早轮次的上下文累积进来。

        回归：曾经把每一轮的上一轮上下文追加到同一个列表，导致第 5 轮的片段里
        出现 5 行"上一轮用户问题"，上下文占到片段 88%，把本轮正文挤成配角，
        同领域无关问题也能靠这些上下文拿到 0.86+。
        """
        turns = [(f"第{index}轮的问题内容", f"第{index}轮的回答内容") for index in range(6)]
        session = self.session("六轮会话", turns)
        records = conv.build_records(session)
        self.assertEqual(len([item for item in records if item["turn_index"] == 5]), 1)
        for item in records:
            lines = item["text"].splitlines()
            prev_users = [line for line in lines if line.startswith("上一轮用户问题：")]
            prev_answers = [line for line in lines if line.startswith("上一轮回答中的邻近内容：")]
            self.assertLessEqual(len(prev_users), 1, item["text"][:200])
            self.assertLessEqual(len(prev_answers), 1, item["text"][:200])
            if item["turn_index"] == 0:
                self.assertEqual(prev_users, [], "第一轮没有上一轮，不该带上下文")
                continue
            # 只带紧邻上一轮：能对上第 turn_index-1 轮，且不含更早轮次
            self.assertIn(f"第{item['turn_index'] - 1}轮", prev_users[0])
            for index in range(item["turn_index"] - 1):
                self.assertNotIn(f"第{index}轮", item["text"], "更早轮次的上下文不该出现")

    def test_context_never_dominates_the_chunk(self):
        """标题 + 上一轮上下文的总长度必须受限，长会话的每一片段都要以本轮为主体。"""
        turns = [(f"第{index}轮的问题" + "补充说明。" * 30, f"第{index}轮的回答" + "细节。" * 60)
                 for index in range(6)]
        session = self.session("长上下文会话", turns)
        for item in records_for_context(session):
            lines = item["text"].splitlines()
            context = [line for line in lines
                       if line.startswith(("会话标题：", "上一轮用户问题：", "上一轮回答中的邻近内容："))]
            context_tokens = sum(conv._tokens(line.split("：", 1)[1]) for line in context)
            self.assertLessEqual(context_tokens,
                                 conv.TITLE_BUDGET_TOKENS + conv.CONTEXT_TOTAL_BUDGET_TOKENS,
                                 item["text"][:200])
            body_tokens = sum(conv._tokens(line.split("：", 1)[1]) for line in lines
                              if line.startswith(("本轮用户问题：", "本轮最终回答：")))
            self.assertGreater(body_tokens, context_tokens, "上下文不能超过本轮主体")


class ScriptedModel(CharModel):
    """按"片段里出现的词"给固定向量：用来精确控制相似度，验证判定分支而不靠分数尺度。

    rules 形如 [(["关键词"], "标记"), ...]：命中第一条规则的文本用该标记向量，
    其余文本用 default 标记。相同标记 → 余弦 1，不同标记 → 余弦 0（正交向量）。
    """

    def __init__(self, rules, default="other", dimension: int = 2048):
        super().__init__(dimension=dimension)
        self.rules = [(tuple(word.lower() for word in words), marker) for words, marker in rules]
        self.default = default

    def _vector(self, text: str) -> np.ndarray:
        body = str(text).lower()
        body = body.split(": ", 1)[-1]
        marker = self.default
        for words, name in self.rules:
            if any(word in body for word in words):
                marker = name
                break
        return self._marker(marker)

    def _marker(self, marker: str) -> np.ndarray:
        if marker == "zero":
            return np.zeros(self.dimension, dtype="float32")
        rng = np.random.default_rng(abs(zlib.crc32(str(marker).encode("utf-8"))) or 1)
        base = rng.standard_normal(2)
        vector = np.zeros(self.dimension, dtype="float32")
        vector[0], vector[1] = base[0], base[1]
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


class TestChunkAndSearch(ConversationCase):
    def setUp(self):
        super().setUp()
        self.excel = self.session("RAG 设计", [(
            "之前那个对话里，我们最后怎么决定 Excel 的入库方式？",
            "Excel 使用模型根据网页介绍提供的 summary 入库，PDF 则直接提取正文。")])
        self.lpl = self.session("LPL 讨论", [("今年 LPL 的 S 赛是哪几个队去？", "BLG、TES、JDG 四支队伍。")])
        self.dinner = self.session("晚饭", [("今晚吃什么？", "吃火锅。")])
        self.sync_and_index()

    # 假模型按"本片段里有没有关键词"给固定向量，因此可以用 TURN_STRONG / TURN_WEAK
    # 精确安排哪个会话该命中、哪个只是沾边，不依赖随机分数尺度。
    def fake_ranked(self, scores):
        """构造 decide_candidates 需要的输入：只验证阈值与分差口径，不经过向量库。"""
        return [{"session_id": f"s{index}", "title": f"会话 {index}", "updated_at": "2026-01-01T00:00:00",
                 "score": score, "best_turn_id": f"s{index}:0", "primary_hit": True,
                 "hits": [{"turn_id": f"s{index}:0", "score": score, "primary_hit": True}]}
                for index, score in enumerate(scores)]

    def fixture_search(self, query, model, **options):
        """用指定模型重建索引再检索：验证"片段命中安排 → 判定结果"这条完整链路。"""
        options.setdefault("sessions_dir", self.sessions)
        options.setdefault("min_score", 0.8)
        with patch.object(rag_store, "_model", model), \
                patch.object(rag_store, "_collection", None), \
                patch.object(rag_store, "_collection_root", None), \
                patch.object(conv, "_collection", None), \
                patch.object(conv, "_collection_root", None):
            self.sync_and_index()
            return conv.search_conversations(query, **options)

    def test_search_returns_the_right_session(self):
        result = self.search("Excel 的入库方式是怎么定的", min_score=0.15)
        self.assertEqual(result["candidates"][0]["session_id"], self.excel["id"])
        self.assertEqual(result["candidates"][0]["title"], "RAG 设计")

    def test_unrelated_question_does_not_return_history(self):
        # 门槛取 0.9（远高于假模型任何正样本的分），验证"没到门槛就不冒充命中"
        result = conv.search_conversations("量子纠缠的贝尔不等式怎么推导", sessions_dir=self.sessions,
                                           min_score=0.9)
        self.assertEqual(result["status"], "not_found")
        # 明显不相干的片段连候选都不报，不能把噪声当成历史交给模型
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["hits"], [])

    def test_near_miss_still_reports_candidates_but_not_found(self):
        """有候选但没到门槛：不冒充命中，仍把高于噪声线的候选报出来供调用方判断。"""
        decision = conv.decide_candidates(self.fake_ranked([0.55, 0.30]), threshold=0.86,
                                          noise=0.2, margin=0.006, max_sessions=3)
        self.assertEqual(decision["status"], "not_found")
        self.assertEqual([item["score"] for item in decision["candidates"]], [0.55, 0.30])

    def test_candidates_below_noise_line_are_hidden(self):
        """低于噪声线的候选不报：避免把明显不相干的片段当成历史。"""
        decision = conv.decide_candidates(self.fake_ranked([0.55, 0.30]), threshold=0.86,
                                          noise=0.7, margin=0.006, max_sessions=3)
        self.assertEqual(decision["status"], "not_found")
        self.assertEqual(decision["candidates"], [])

    def test_current_session_can_be_excluded(self):
        result = self.search("Excel 的入库方式是怎么定的", exclude_sessions=[self.excel["id"]])
        self.assertNotIn(self.excel["id"], [item["session_id"] for item in result["candidates"]])

    def test_at_most_three_candidates(self):
        for index in range(5):
            self.session(f"相似主题 {index}", [("Excel 的入库方式", "用 summary 入库。")])
        self.sync_and_index()
        result = self.search("Excel 的入库方式")
        self.assertLessEqual(len(result["candidates"]), conv.MAX_CANDIDATES)

    def test_ranking_uses_best_chunk_not_sum(self):
        """会话排序只看最佳片段：片段多的会话不该靠"数量多"赢过只有一个强命中的会话。"""
        long_session = self.session("长会话", [("无关话题", f"无关回答 {index}。") for index in range(12)])
        short_session = self.session("短会话", [("Excel 的入库方式", "用网页 summary 入库。")])
        model = ScriptedModel(rules=[(["excel", "入库方式"], "hit")], default="flat")
        result = self.fixture_search("Excel 的入库方式", model, min_score=0.9, ambiguous_margin=0.0)
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["candidates"][0]["session_id"], short_session["id"])
        self.assertNotEqual(result["candidates"][0]["session_id"], long_session["id"])

    def test_close_candidates_are_ambiguous(self):
        """两个会话分数几乎相同 → ambiguous，不悄悄选一个当确定来源。"""
        first = self.session("方案 A 讨论", [("方案 A 的缓存怎么选？", "按方案 A 定。")])
        second = self.session("方案 B 讨论", [("方案 B 的缓存怎么选？", "按方案 B 定。")])
        model = ScriptedModel(rules=[(["方案"], "tie")], default="tie")
        result = self.fixture_search("方案 A 和方案 B 的缓存怎么选", model,
                                     min_score=0.9, ambiguous_margin=0.01)
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual({item["session_id"] for item in result["candidates"]},
                         {first["id"], second["id"]})
        # 模糊结果也带命中片段，便于主模型按原文判断
        self.assertTrue(all(item["hits"] for item in result["candidates"]))

    def test_clear_winner_is_found_not_ambiguous(self):
        self.session("完全无关的会话", [("怎么养绿萝？", "少浇水多通风。")])
        self.sync_and_index()
        result = self.search("Excel 的入库方式")
        self.assertEqual(result["status"], "found")

    def test_extra_hits_support_the_chosen_session(self):
        session = self.session("多轮讨论", [
            ("Excel 的入库方式怎么定？", "用 summary 入库。"),
            ("那 PDF 呢？", "PDF 直接提取正文。"),
        ])
        self.sync_and_index()
        result = self.search("Excel 的入库方式和 PDF 有什么区别")
        self.assertEqual(result["candidates"][0]["session_id"], session["id"])
        self.assertTrue(len(result["candidates"][0]["hits"]) >= 1)

    def test_empty_query_is_not_found(self):
        result = self.search("   ")
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["hits"], [])


class TestSyncAndVersions(ConversationCase):
    def test_sync_is_idempotent(self):
        session = self.session("幂等", [("问题", "回答")])
        first = conv.sync_session(session, sessions_dir=self.sessions)
        self.assertTrue(first["queued"])
        self.run_jobs()
        again = conv.sync_session(session, sessions_dir=self.sessions)
        self.assertTrue(again["skipped"])
        self.assertFalse(again["queued"])
        # 重复通知不重复产生向量
        before = conv.vector_count()
        conv.sync_session(session, sessions_dir=self.sessions)
        conv.sync_session(session, sessions_dir=self.sessions)
        self.assertEqual(conv.vector_count(), before)
        conn = rag_state.connect()
        try:
            rows = conn.execute("SELECT COUNT(*) AS n FROM conversation_versions").fetchone()["n"]
            jobs = conn.execute("SELECT COUNT(*) AS n FROM conversation_jobs").fetchone()["n"]
        finally:
            conn.close()
        self.assertEqual((rows, jobs), (1, 1))

    def test_all_existing_sessions_are_indexed_on_first_sync(self):
        for index in range(3):
            self.session(f"旧会话 {index}", [(f"问题 {index}", f"回答 {index}")])
        result = conv.sync_all(sessions_dir=self.sessions)
        self.assertEqual(result["sessions"], 3)
        self.assertEqual(result["queued"], 3)
        self.run_jobs()
        conn = rag_state.connect()
        try:
            self.assertEqual(conv.indexed_count(conn), 3)
            self.assertEqual(conv.pending_count(conn), 0)
        finally:
            conn.close()

    def test_editing_an_old_turn_rebuilds_the_session(self):
        session = self.session("会变的会话", [("第一轮问题", "第一轮回答")])
        self.sync_and_index()
        old_revision = conv.session_version(rag_state.connect(), session["id"])
        changed = dict(session)
        changed["turns"] = [{"user": "第一轮问题", "final_answer": "第一轮回答（已修改）"},
                            {"user": "第二轮问题", "final_answer": "第二轮回答"}]
        self.store.write(changed)
        result = conv.sync_session(changed, sessions_dir=self.sessions)
        self.assertFalse(result["skipped"])
        self.assertNotEqual(result["revision"], old_revision)
        self.run_jobs()
        # 旧版本的向量被删除，检索不到旧内容
        self.assertEqual(conv.delete_version_vectors(session["id"], old_revision), 0)
        conn = rag_state.connect()
        try:
            active = rag_state.active_conversation_version(conn, session["id"])
        finally:
            conn.close()
        self.assertEqual(active["revision"], revision_of(changed))
        found = self.search("第一轮回答（已修改）")
        self.assertEqual(found["candidates"][0]["session_id"], session["id"])

    def test_deleting_a_turn_rebuilds_and_drops_old_vectors(self):
        session = self.session("删轮次", [("第一轮问题", "第一轮回答"), ("第二轮问题", "第二轮回答")])
        self.sync_and_index()
        before = conv.vector_count()
        trimmed = dict(session)
        trimmed["turns"] = [{"user": "第一轮问题", "final_answer": "第一轮回答"}]
        self.store.write(trimmed)
        conv.sync_session(trimmed, sessions_dir=self.sessions)
        self.run_jobs()
        self.assertLess(conv.vector_count(), before)
        result = self.search("第二轮回答")
        candidates = [item["session_id"] for item in result["candidates"]]
        self.assertNotIn(session["id"], candidates)

    def test_failed_index_retries_then_can_be_retried_by_user(self):
        session = self.session("会失败的会话", [("问题", "回答")])
        conv.sync_session(session, sessions_dir=self.sessions)
        conn = rag_state.connect()
        try:
            job = rag_state.claim_conversation_job(conn, float("inf"))
            for expected in ("retry_wait", "retry_wait", "failed"):
                result = rag_state.finish_conversation_job(conn, job["job_id"], ok=False,
                                                           error="模拟 embedding 失败")
                self.assertEqual(result["status"], expected)
                if expected != "failed":
                    job = rag_state.claim_conversation_job(conn, float("inf"))
            snapshot = rag_state.conversation_job_snapshot(conn)
            self.assertEqual(len(snapshot), 1)
            self.assertEqual(snapshot[0]["status"], "failed")
            self.assertIn("模拟 embedding 失败", snapshot[0]["last_error"])
            fresh = rag_state.retry_conversation_job(conn, snapshot[0]["job_id"])
            self.assertEqual(fresh["round"], 2)
            self.assertEqual(fresh["status"], "queued")
        finally:
            conn.close()
        # 用户重试后能真正完成；被替代的旧失败记录不再算作待办或失败
        self.run_jobs()
        conn = rag_state.connect()
        try:
            self.assertEqual(conv.pending_count(conn), 0)
            self.assertEqual(conv.failed_count(conn), 0)
            self.assertEqual(rag_state.conversation_job_snapshot(conn), [])
            active = rag_state.active_conversation_version(conn, session["id"])
            self.assertEqual(active["status"], "indexed")
        finally:
            conn.close()

    def test_broken_session_file_is_reported_not_indexed(self):
        self.sessions.mkdir(parents=True, exist_ok=True)
        path = self.sessions / "broken1.json"
        path.write_text("{不是合法 JSON", encoding="utf-8")
        self.session("正常会话", [("问题", "回答")])
        result = conv.sync_all(sessions_dir=self.sessions)
        self.assertEqual(result["sessions"], 1)
        self.assertTrue(result["broken"])

    def test_reconcile_and_orphan_cleanup(self):
        session = self.session("校对", [("问题", "回答")])
        self.sync_and_index()
        store = conv._store()
        found = store.get(include=["metadatas"])
        store.update(ids=found["ids"],
                     metadatas=[{**(meta or {}), "active": 0} for meta in found["metadatas"]])
        changed = conv.reconcile_active()
        self.assertGreater(changed, 0)
        self.assertEqual({int(meta.get("active") or 0)
                          for meta in conv._store().get(include=["metadatas"])["metadatas"]}, {1})
        # 版本记录删掉后，残留向量会被清理
        conn = rag_state.connect()
        try:
            conn.execute("DELETE FROM conversation_jobs")
            conn.execute("DELETE FROM conversation_versions")
            conn.execute("DELETE FROM conversations")
            conn.commit()
        finally:
            conn.close()
        self.assertGreater(conv.drop_orphans(), 0)
        self.assertEqual(conv.vector_count(), 0)

    def test_version_mismatch_is_a_permanent_error(self):
        session = self.session("版本不符", [("问题", "回答")])
        conv.sync_session(session, sessions_dir=self.sessions)
        conn = rag_state.connect()
        try:
            version = rag_state.latest_conversation_version(conn, session["id"])
        finally:
            conn.close()
        session["turns"].append({"user": "新问题", "final_answer": "新回答"})
        self.store.write(session)
        with self.assertRaises(rag_state.PermanentError):
            conv.index_version(version, sessions_dir=self.sessions)


class TestEvidence(ConversationCase):
    def test_evidence_comes_from_saved_turns(self):
        session = self.session("证据", [("第一轮问题", "第一轮回答"), ("本轮问题", "本轮回答")])
        evidence = conv.recall_candidates(session["id"], turn_ids=[f"{session['id']}:1"],
                                         sessions_dir=self.sessions)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["turn_id"], f"{session['id']}:1")
        self.assertEqual(evidence[0]["user"], "本轮问题")
        self.assertEqual(evidence[0]["final_answer"], "本轮回答")
        self.assertFalse(evidence[0]["final_answer_truncated"])

    def test_long_evidence_reports_truncation_range(self):
        answer = "很长的回答。" * 500
        session = self.session("长证据", [("问题", answer)])
        evidence = conv.recall_candidates(session["id"], sessions_dir=self.sessions)
        self.assertTrue(evidence[0]["final_answer_truncated"])
        self.assertEqual(evidence[0]["answer_chars"][1], len(evidence[0]["final_answer"]) - 1)

    def test_missing_session_returns_empty(self):
        self.assertEqual(conv.recall_candidates("nope99", sessions_dir=self.sessions), [])

    def test_key_decisions_can_be_traced_to_turns(self):
        session = self.session("决定溯源", [
            ("Excel 需要 summary，PDF 直接用内容", "好的，Excel 用 summary 入库。"),
            ("那就按这个做", "已经按这个方案实现。")])
        self.sync_and_index()
        result = conv.search_conversations("Excel 入库方案", sessions_dir=self.sessions, min_score=0.15)
        turn_ids = [hit["turn_id"] for hit in result["candidates"][0]["hits"]]
        evidence = conv.recall_candidates(session["id"], turn_ids=turn_ids,
                                         sessions_dir=self.sessions)
        self.assertTrue(evidence)
        for item in evidence:
            self.assertTrue(item["turn_id"].startswith(session["id"]))


if __name__ == "__main__":
    unittest.main()
