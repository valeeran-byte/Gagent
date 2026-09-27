"""recall_conversation 工具与当前会话上下文的验收测试。

不联网、不拉服务：客户端调用用桩替换，全部验证"工具怎么把当前会话排除掉、
怎么把三种结果如实交给模型"。
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("GAGENT_RAG_DISABLE", "1")

import agent_tools
import G_agent as agent


class FakeRecall:
    """记录每次调用参数，按脚本返回服务侧结果。"""

    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"status": "not_found", "query": "x", "candidates": []}

    def __call__(self, query, **options):
        self.calls.append({"query": query, **options})
        return self.result


class RecallToolCase(unittest.TestCase):
    def setUp(self):
        self.fake = FakeRecall()

    def call(self, query="Excel 的入库方式", session_id=None, result=None,
             current="abc123"):
        if result is not None:
            self.fake.result = result
        with patch.object(agent_tools.rag_client, "recall_conversation", self.fake), \
                agent_tools.conversation_scope(current):
            return agent_tools.recall_conversation.invoke({"query": query, "session_id": session_id})

    # ---------- 当前会话上下文 ----------

    def test_current_session_is_excluded_by_default(self):
        self.call()
        self.assertEqual(self.fake.calls[0]["exclude_session"], "abc123")
        self.assertIsNone(self.fake.calls[0]["session_id"])

    def test_explicit_session_id_is_passed_and_not_excluded(self):
        self.call(session_id="other99")
        self.assertEqual(self.fake.calls[0]["session_id"], "other99")
        self.assertIsNone(self.fake.calls[0]["exclude_session"])

    def test_no_session_context_means_no_exclusion(self):
        self.call(current=None)
        self.assertIsNone(self.fake.calls[0]["exclude_session"])
        self.assertIsNone(self.fake.calls[0]["session_id"])

    def test_asking_about_the_current_session_is_refused(self):
        result = self.call(session_id="abc123")
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "current_session")
        self.assertIn("当前会话", result["reason"])
        self.assertEqual(self.fake.calls, [], "拒绝时不该再去检索")

    def test_empty_query_is_invalid(self):
        result = self.call(query="   ")
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(self.fake.calls, [])

    # ---------- 三种结果如实转交 ----------

    def test_found_returns_summary_and_evidence(self):
        service = {"status": "found", "query": "Excel 的入库方式",
                   "session": {"id": "abc999", "title": "RAG 设计", "updated_at": "2026-01-01T00:00:00"},
                   "summary_status": "cached",
                   "summary": {"topic": "文件 RAG 方案讨论",
                               "user_requirements": [{"text": "Excel 用 summary 入库", "turn_ids": ["abc999:1"]}],
                               "confirmed_decisions": [], "assistant_proposals": [],
                               "open_questions": [], "superseded_decisions": []},
                   "summary_stale": False,
                   "evidence": [{"turn_id": "abc999:1", "user": "不是，是 Excel 需要 summary",
                                 "final_answer": "好的，Excel 用 summary 入库。"}],
                   "candidates": [{"session_id": "abc999", "title": "RAG 设计", "score": 0.91}]}
        result = self.call(result=service)
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["session"]["id"], "abc999")
        self.assertEqual(result["summary"]["topic"], "文件 RAG 方案讨论")
        self.assertEqual(result["evidence"][0]["turn_id"], "abc999:1")
        # 摘要里引用的轮次必须在返回的 evidence 里能对应上
        cited = result["summary"]["user_requirements"][0]["turn_ids"]
        self.assertTrue(set(cited) & {item["turn_id"] for item in result["evidence"]})

    def test_ambiguous_returns_candidates_without_pretending(self):
        service = {"status": "ambiguous", "query": "方案怎么定",
                   "candidates": [{"session_id": "a1", "title": "方案 A 讨论", "score": 0.9,
                                   "evidence": [{"turn_id": "a1:0", "user": "方案 A", "final_answer": "用内存缓存"}]},
                                  {"session_id": "b2", "title": "方案 B 讨论", "score": 0.895,
                                   "evidence": [{"turn_id": "b2:0", "user": "方案 B", "final_answer": "用磁盘缓存"}]}]}
        result = self.call(result=service)
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual({item["session_id"] for item in result["candidates"]}, {"a1", "b2"})
        self.assertNotIn("summary", result)

    def test_not_found_is_reported_as_is(self):
        result = self.call(result={"status": "not_found", "query": "量子纠缠", "candidates": []})
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["candidates"], [])

    def test_error_is_reported_without_faking_history(self):
        result = self.call(result={"status": "error", "reason": "ServiceUnavailable", "evidence": []})
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "error")
        self.assertIn("ServiceUnavailable", result["reason"])
        self.assertEqual(result["evidence"], [])

    def test_summary_failure_still_carries_evidence(self):
        service = {"status": "found", "query": "Excel 的入库方式",
                   "session": {"id": "abc999", "title": "RAG 设计"},
                   "summary_status": "failed", "summary": None,
                   "summary_note": "摘要模型超时", "summary_stale": False,
                   "evidence": [{"turn_id": "abc999:1", "user": "问题", "final_answer": "回答"}]}
        result = self.call(result=service)
        self.assertTrue(result["success"])
        self.assertEqual(result["summary_status"], "failed")
        self.assertIsNone(result["summary"])
        self.assertEqual(result["evidence"][0]["turn_id"], "abc999:1")


class ToolRegistration(unittest.TestCase):
    def test_tool_is_registered_and_documented(self):
        names = [item.name for item in agent_tools.TOOLS]
        self.assertIn("recall_conversation", names)

    def test_system_prompt_has_recall_rule(self):
        self.assertIn("recall_conversation", agent.SYSTEM_PROMPT)
        self.assertIn("其他会话", agent.SYSTEM_PROMPT)
        self.assertIn("参考资料", agent.SYSTEM_PROMPT)
        self.assertIn("没有找到就如实说明", agent.SYSTEM_PROMPT)
        # 普通问题不强制调用，且不抢当前会话历史
        self.assertIn("普通问题不要求调用", agent.SYSTEM_PROMPT)

    def test_scope_is_reset_after_each_run(self):
        with agent_tools.conversation_scope("abc123"):
            self.assertEqual(agent_tools.current_session_id(), "abc123")
        self.assertIsNone(agent_tools.current_session_id())


if __name__ == "__main__":
    unittest.main()
