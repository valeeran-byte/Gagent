"""Gagent CLI 与会话存储的验收测试。

只使用临时 sessions/logs 目录，不读写用户真实会话；除 TestModelMessages 里用假模型驱动
真实的 BasicAgent 消息构造外，其余都不联网。
"""
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# 单测不为文件服务拉起后台进程；服务侧行为在 test_rag_service.py 里用隔离目录单独验证。
os.environ.setdefault("GAGENT_RAG_DISABLE", "1")

import cli
import G_agent as agent
import session_store
from session_store import SessionCorrupted, SessionNotFound, SessionStore


class FakeAgent:
    """替代 BasicAgent：记录每轮入参，按脚本返回结果，绝不联网。"""

    def __init__(self, script=None, events=()):
        self.script = dict(script or {})
        self.calls = []
        self.events = list(events)

    def run(self, question, *, file_url=None, file_name=None, history=None,
            progress=None, session_id=None, echo=True):
        self.calls.append(SimpleNamespace(question=question, history=[dict(turn) for turn in history or []],
                                          session_id=session_id, progress=progress, echo=echo))
        case = self.script.get(question, {})
        if case.get("raises") is KeyboardInterrupt:
            raise KeyboardInterrupt
        if case.get("raises"):
            raise case["raises"]
        if progress:
            for event in self.events:
                progress(*event)
        return agent.AgentResult(user_prompt=case.get("user_prompt", question),
                                 answer=case.get("answer", f"答：{question}"),
                                 ok=case.get("ok", True))

    @property
    def questions(self):
        return [call.question for call in self.calls]


class CliHarness(unittest.TestCase):
    """把 Chat 接到脚本化输入上，返回它写出的文本；active 指定磁盘上的上次活动会话。"""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gagent-cli-test-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.store = SessionStore(self.root / "sessions")
        self.logs = self.root / "logs"
        patcher = patch.object(agent, "LOGS_DIR", self.logs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def drive(self, inputs, fake=None, active=None, notifier=None):
        fake = FakeAgent() if fake is None else fake
        if active is not None:
            self.store.set_active(active["id"])
        output = io.StringIO()
        pending = iter(inputs)

        def input_fn(_prompt):
            try:
                return next(pending)
            except StopIteration:
                raise EOFError

        # 默认用静音通知器：单测不该为提示去读真实服务状态
        silent = notifier if notifier is not None else cli.RagNotifier(
            lambda text: None, poll_seconds=0, client=StubRagClient())
        chat = cli.Chat(self.store, fake, out=output, input_fn=input_fn, notifier=silent)
        return chat.start(), output.getvalue(), chat, fake


class TestStartup(CliHarness):
    def test_first_start_creates_and_enters_one_empty_session(self):
        code, text, chat, fake = self.drive(["/exit"])
        self.assertEqual(code, 0)
        self.assertIn("Gagent", text)
        self.assertIn("输入 /help 查看命令。", text)
        self.assertRegex(text, r"当前会话：未命名会话 \[[0-9a-f]{6}\]")
        self.assertEqual(chat.session["turns"], [])
        metas, _broken = self.store.scan()
        self.assertEqual([meta["id"] for meta in metas], [chat.session["id"]])
        self.assertEqual(self.store.active_id(), chat.session["id"])
        self.assertEqual(fake.calls, [])

    def test_restart_resumes_last_active_session_with_its_turns(self):
        first = self.store.create("幸福报告调查")
        self.store.save_turn(first, "美国幸福排名下降了吗？", "根据报告……")
        code, text, chat, _fake = self.drive(["/exit"], active=first)
        self.assertEqual(code, 0)
        self.assertIn("当前会话：幸福报告调查", text)
        self.assertIn(first["id"], text)
        self.assertEqual(chat.session["turns"],
                         [{"user": "美国幸福排名下降了吗？", "final_answer": "根据报告……"}])

    def test_missing_active_marker_lists_sessions_and_keeps_data(self):
        kept = self.store.create("旧会话")
        self.store.save_turn(kept, "问题", "回答")
        _code, text, chat, fake = self.drive(["/exit"])
        self.assertIn("没有找到上次会话标记", text)
        self.assertIn("旧会话", text)
        self.assertIn("/switch", text)
        self.assertIsNone(chat.session)
        self.assertEqual(fake.calls, [])
        self.assertIsNone(self.store.active_id())
        self.assertEqual(self.store.read(kept["id"])["turns"], [{"user": "问题", "final_answer": "回答"}])

    def test_active_marker_pointing_at_missing_session_is_not_overwritten(self):
        self.store.create("存在的会话")
        self.store.set_active("abcdef")
        _code, text, chat, _fake = self.drive(["/exit"])
        self.assertIn("上次会话 abcdef 的文件不存在", text)
        self.assertIsNone(chat.session)
        self.assertEqual(self.store.active_id(), "abcdef")

    def test_corrupted_active_session_file_is_reported_not_reset(self):
        session = self.store.create("会损坏的")
        self.store.set_active(session["id"])
        path = self.store.path_of(session["id"])
        path.write_text("{ 这不是 JSON", encoding="utf-8")
        _code, text, chat, _fake = self.drive(["/exit"])
        self.assertIn("读不出来", text)
        self.assertIsNone(chat.session)
        self.assertEqual(path.read_text(encoding="utf-8"), "{ 这不是 JSON")

    def test_state_file_only_holds_the_active_id(self):
        session = self.store.create("只记 ID")
        self.store.set_active(session["id"])
        raw = json.loads(self.store.state_path.read_text(encoding="utf-8"))
        self.assertEqual(raw, {"active_session": session["id"]})


class TestCommands(CliHarness):
    def test_help_lists_every_command_and_the_memory_rules(self):
        _code, text, _chat, _fake = self.drive(["/help", "/exit"])
        for token in ("/help", "/api", "/new [标题]", "/sessions", "/switch <id>", "/exit"):
            self.assertIn(token, text)
        self.assertIn("同一个会话保留上下文", text)
        self.assertIn("/new 创建独立会话", text)
        self.assertIn("/switch 恢复已有会话", text)
        self.assertIn("不进入会话历史", text)

    def test_local_commands_blank_input_and_bad_ids_never_reach_the_agent(self):
        _code, _text, _chat, fake = self.drive(
            ["", "   ", "/help", "/sessions", "/nope", "/switch", "abc", "/switch 123456",
             "/switch a b", "/exit"])
        self.assertEqual(fake.calls, [])
        self.assertEqual(list(self.logs.glob("*.txt")), [])

    def test_switch_lists_sessions_and_selects_by_number(self):
        first = self.store.create("幸福报告调查")
        self.store.save_turn(first, "q", "a")
        second = self.store.create("LangGraph学习")
        _code, text, chat, fake = self.drive(["/switch", "2", "继续刚才的话题", "/exit"],
                                             fake=FakeAgent(), active=second)
        self.assertIn("选择会话（输入编号，直接回车取消）：", text)
        self.assertIn("* 1) LangGraph学习", text)
        self.assertIn("  2) 幸福报告调查", text)
        self.assertIn(f"已切换到：幸福报告调查 [{first['id']}]", text)
        self.assertEqual(chat.session["id"], first["id"])
        self.assertEqual(self.store.active_id(), first["id"])
        self.assertEqual(fake.calls[0].history, [{"user": "q", "final_answer": "a"}])
        self.assertEqual(fake.calls[0].session_id, first["id"])

    def test_switch_picker_cancel_keeps_current_session(self):
        keep = self.store.create("保持这个")
        self.store.create("别的")
        for answer in ("", "9", "x"):
            _code, text, chat, fake = self.drive(["/switch", answer, "/exit"],
                                                 fake=FakeAgent(), active=keep)
            self.assertIn("已取消", text)
            self.assertEqual(chat.session["id"], keep["id"])
            self.assertEqual(fake.calls, [])

    def test_switch_picker_on_current_session_says_so(self):
        only = self.store.create("只有一个")
        _code, text, chat, _fake = self.drive(["/switch", "1", "/exit"], active=only)
        self.assertIn(f"已经在：只有一个 [{only['id']}]", text)
        self.assertEqual(chat.session["id"], only["id"])

    def test_switch_picker_without_sessions(self):
        output = io.StringIO()
        chat = cli.Chat(self.store, FakeAgent(), out=output)
        chat.pick_session()
        self.assertIn("还没有已保存的会话", output.getvalue())

    def test_unknown_command_points_at_help(self):
        _code, text, _chat, _fake = self.drive(["/frobnicate", "/exit"])
        self.assertIn("未知命令：/frobnicate", text)
        self.assertIn("/help", text)

    def test_switch_with_too_many_arguments_shows_usage(self):
        _code, text, _chat, _fake = self.drive(["/switch a b", "/exit"])
        self.assertIn("用法：/switch 选择会话，或 /switch <id>", text)

    def test_sessions_lists_id_title_time_and_marks_current(self):
        first = self.store.create("幸福报告调查")
        self.store.save_turn(first, "q1", "a1")
        second = self.store.create("LangGraph学习")
        _code, text, _chat, _fake = self.drive(["/sessions", "/exit"], active=second)
        self.assertIn("会话列表：", text)
        starred = [line for line in text.splitlines() if line.startswith("*")]
        self.assertEqual(len(starred), 1)
        self.assertIn(second["id"], starred[0])
        self.assertIn("LangGraph学习", starred[0])
        self.assertIn(first["id"], text)
        self.assertIn("幸福报告调查", text)
        self.assertIn("1 轮", text)
        self.assertIn(datetime.now().strftime("%Y-%m-%d"), text)

    def test_new_with_explicit_title(self):
        _code, text, chat, _fake = self.drive(["/new LangGraph学习", "/exit"])
        self.assertIn("已创建并切换到：LangGraph学习", text)
        self.assertEqual(chat.session["title"], "LangGraph学习")
        self.assertEqual(self.store.active_id(), chat.session["id"])

    def test_untitled_session_is_named_by_first_question(self):
        fake = FakeAgent()
        _code, text, chat, _f = self.drive(["ToolNode 是什么？", "第二个问题", "/exit"], fake)
        self.assertIn("当前会话：未命名会话", text)
        session_id = chat.session["id"]
        self.assertEqual(self.store.read(session_id)["title"], "ToolNode 是什么？")
        self.assertEqual(fake.questions, ["ToolNode 是什么？", "第二个问题"])
        self.assertEqual(fake.calls[1].history,
                         [{"user": "ToolNode 是什么？", "final_answer": "答：ToolNode 是什么？"}])

    def test_new_without_title_hints_at_the_first_question(self):
        _code, text, _chat, _fake = self.drive(["/new", "/exit"])
        self.assertIn("已创建并切换到：未命名会话", text)
        self.assertIn("第一个提问会成为会话名", text)

    def test_explicit_title_is_not_replaced_by_the_first_question(self):
        fake = FakeAgent()
        _code, text, chat, _f = self.drive(["/new 我的会话", "第一个问题", "/exit"], fake)
        self.assertEqual(chat.session["title"], "我的会话")
        self.assertEqual(self.store.read(chat.session["id"])["title"], "我的会话")

    def test_new_keeps_existing_sessions(self):
        first = self.store.create("第一个")
        self.store.save_turn(first, "q", "a")
        _code, _text, chat, _fake = self.drive(["/new 第二个", "/exit"], active=first)
        self.assertNotEqual(chat.session["id"], first["id"])
        metas, _broken = self.store.scan()
        self.assertEqual({meta["title"] for meta in metas}, {"第一个", "第二个"})
        self.assertEqual(self.store.read(first["id"])["turns"], [{"user": "q", "final_answer": "a"}])

    def test_switch_restores_the_other_sessions_history(self):
        first = self.store.create("A 会话")
        self.store.save_turn(first, "A 的问题", "A 的回答")
        second = self.store.create("B 会话")
        fake = FakeAgent()
        _code, text, chat, _fake = self.drive([f"/switch {first['id']}", "回到 A", "/exit"],
                                              fake, active=second)
        self.assertIn(f"已切换到：A 会话 [{first['id']}]", text)
        self.assertEqual(chat.session["id"], first["id"])
        self.assertEqual(fake.calls[0].history, [{"user": "A 的问题", "final_answer": "A 的回答"}])
        self.assertEqual(fake.calls[0].session_id, first["id"])
        self.assertEqual(fake.questions, ["回到 A"])

    def test_switch_missing_id_explains_and_keeps_current(self):
        current = self.store.create("留下的")
        _code, text, chat, _fake = self.drive(["/switch ffffff", "/exit"], active=current)
        self.assertIn("找不到会话 ffffff", text)
        self.assertIn("当前会话不变", text)
        self.assertEqual(chat.session["id"], current["id"])

    def test_corrupted_file_on_switch_keeps_current_session_and_bytes(self):
        current = self.store.create("好的")
        broken = self.store.create("坏的")
        path = self.store.path_of(broken["id"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["turns"] = "不是列表"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        _code, text, chat, _fake = self.drive([f"/switch {broken['id']}", "/exit"], active=current)
        self.assertIn("读不出来", text)
        self.assertEqual(chat.session["id"], current["id"])
        self.assertIn("不是列表", path.read_text(encoding="utf-8"))

    def test_switch_argument_cannot_escape_the_sessions_dir(self):
        current = self.store.create("当前")
        _code, text, chat, fake = self.drive(["/switch ../../etc/passwd", "/exit"], active=current)
        self.assertIn("找不到会话", text)
        self.assertEqual(fake.calls, [])
        self.assertEqual(chat.session["id"], current["id"])

    def test_exit_and_eof_stop_the_loop(self):
        code, text, _chat, _fake = self.drive(["/exit", "不会被读到"])
        self.assertEqual(code, 0)
        self.assertNotIn("答：不会被读到", text)
        code, text, _chat, _fake = self.drive([])
        self.assertEqual(code, 0)
        self.assertIn("再见", text)

    def test_ctrl_c_at_the_prompt_exits_without_a_traceback(self):
        fake = FakeAgent()
        output = io.StringIO()

        def input_fn(_prompt):
            raise KeyboardInterrupt

        chat = cli.Chat(self.store, fake, out=output, input_fn=input_fn)
        self.assertEqual(chat.start(), 0)
        self.assertIn("再见", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())


class TestMemory(CliHarness):
    def test_second_turn_history_carries_previous_user_and_answer(self):
        fake = FakeAgent()
        session = self.store.create("记忆")
        _code, _text, chat, _f = self.drive(["第一个问题", "第二个问题", "/exit"], fake, active=session)
        self.assertEqual(fake.questions, ["第一个问题", "第二个问题"])
        self.assertEqual(fake.calls[0].history, [])
        self.assertEqual(fake.calls[1].history,
                         [{"user": "第一个问题", "final_answer": "答：第一个问题"}])
        self.assertEqual(chat.session["turns"], fake.calls[1].history +
                         [{"user": "第二个问题", "final_answer": "答：第二个问题"}])
        self.assertEqual(self.store.read(session["id"])["turns"], chat.session["turns"])

    def test_each_turn_is_saved_immediately_not_only_at_exit(self):
        fake = FakeAgent()
        session = self.store.create("即时保存")
        _code, _text, _chat, _f = self.drive(["第一问", "第二问", "第三问", "/exit"], fake, active=session)
        self.assertEqual(len(self.store.read(session["id"])["turns"]), 3)

    def test_attachment_text_stays_in_the_saved_turn(self):
        fake = FakeAgent(script={"带附件": {"user_prompt": "带附件\n\n【资料】报告.pdf：https://e/x.pdf"}})
        session = self.store.create("附件")
        self.drive(["带附件", "/exit"], fake, active=session)
        self.assertIn("https://e/x.pdf", self.store.read(session["id"])["turns"][0]["user"])

    def test_sessions_do_not_share_history(self):
        fake = FakeAgent()
        first = self.store.create("A 会话")
        self.store.save_turn(first, "A 的问题", "A 的回答")
        second = self.store.create("B 会话")
        _code, _text, chat, _f = self.drive(
            ["B 的问题", f"/switch {first['id']}", "回到 A", "/exit"], fake, active=second)
        self.assertEqual(fake.calls[0].history, [])
        self.assertEqual(fake.calls[0].session_id, second["id"])
        self.assertEqual(fake.calls[1].history, [{"user": "A 的问题", "final_answer": "A 的回答"}])
        self.assertEqual(fake.calls[1].session_id, first["id"])
        self.assertEqual(chat.session["turns"],
                         [{"user": "A 的问题", "final_answer": "A 的回答"},
                          {"user": "回到 A", "final_answer": "答：回到 A"}])
        self.assertEqual(self.store.read(second["id"])["turns"],
                         [{"user": "B 的问题", "final_answer": "答：B 的问题"}])

    def test_progress_and_answer_come_from_the_run_not_from_history(self):
        fake = FakeAgent(events=[("tool_call", "web_search", "")])
        session = self.store.create("进度")
        _code, text, _chat, _f = self.drive(["提问", "/exit"], fake, active=session)
        self.assertIn("● 正在搜索……", text)
        self.assertEqual(fake.calls[0].echo, False)
        self.assertEqual(fake.calls[0].session_id, session["id"])

    def test_no_session_selected_means_no_model_call(self):
        self.store.create("已有一个会话")
        fake = FakeAgent()
        _code, text, _chat, _f = self.drive(["随便问一句", "/exit"], fake)
        self.assertIn("尚未选定会话", text)
        self.assertEqual(fake.calls, [])

    def test_incomplete_rounds_are_not_saved(self):
        fake = FakeAgent(script={
            "模型失败": {"raises": RuntimeError("boom")},
            "被中止": {"raises": KeyboardInterrupt},
            "没完成": {"ok": False, "answer": "模型请求失败：Timeout"},
        })
        session = self.store.create("不保存")
        _code, text, chat, _f = self.drive(
            ["模型失败", "被中止", "没完成", "好的", "/exit"], fake, active=session)
        self.assertIn("本轮未完成：RuntimeError", text)
        self.assertIn("本轮已中止", text)
        self.assertIn("模型请求失败：Timeout", text)
        self.assertEqual(chat.session["turns"], [{"user": "好的", "final_answer": "答：好的"}])
        self.assertEqual(self.store.read(session["id"])["turns"], chat.session["turns"])

    def test_failure_with_history_suggests_a_new_session(self):
        fake = FakeAgent(script={"第二个": {"ok": False, "answer": "模型请求失败：BadRequest"}})
        session = self.store.create("过长")
        self.store.save_turn(session, "第一个", "第一段很长的回答")
        _code, text, _chat, _f = self.drive(["第二个", "/exit"], fake, active=session)
        self.assertIn("/new", text)
        self.assertIn("上下文过长", text)

    def test_final_answer_printed_once_and_no_internals_dumped(self):
        answer = "根据报告，美国排名下降了。"
        fake = FakeAgent(script={"问题": {"answer": answer}},
                         events=[("tool_call", "web_search", ""),
                                 ("tool_call", "visit_webpage", ""),
                                 ("tool_error", "visit_webpage", "HTTP 404: 资源不存在")])
        session = self.store.create("显示")
        _code, text, _chat, _f = self.drive(["问题", "/exit"], fake, active=session)
        self.assertEqual(text.count(answer), 1)
        self.assertIn("● 正在搜索……", text)
        self.assertIn("● 正在读取网页……", text)
        self.assertIn("● 工具执行失败：HTTP 404: 资源不存在", text)
        for token in ("[final]", "AgentResult", "'messages'", "Traceback", "tool_call"):
            self.assertNotIn(token, text)
        self.assertEqual(list(self.logs.glob("*.txt")), [])

    def test_save_failure_is_reported_and_retried(self):
        fake = FakeAgent()
        session = self.store.create("写入失败")
        real_write = SessionStore.write
        state = {"fail": True}

        def flaky_write(self, item):
            if state["fail"]:
                state["fail"] = False
                raise OSError(13, "Permission denied")
            real_write(self, item)

        with patch.object(SessionStore, "write", flaky_write):
            code, text, chat, _f = self.drive(["这一轮要保住", "/sessions", "/exit"], fake, active=session)
        self.assertEqual(code, 0)
        self.assertIn("本轮未保存", text)
        self.assertIn("已补存", text)
        self.assertEqual([turn["user"] for turn in chat.session["turns"]], ["这一轮要保住"])
        self.assertEqual(self.store.read(session["id"])["turns"],
                         [{"user": "这一轮要保住", "final_answer": "答：这一轮要保住"}])

    def test_unsaved_turn_survives_a_switch(self):
        fake = FakeAgent()
        first = self.store.create("A")
        second = self.store.create("B")
        real_write = SessionStore.write
        state = {"fail": True}

        def flaky_write(self, item):
            if state["fail"] and item["id"] == first["id"]:
                state["fail"] = False
                raise OSError(13, "Permission denied")
            real_write(self, item)

        with patch.object(SessionStore, "write", flaky_write):
            _code, text, _chat, _f = self.drive(
                ["A 的问题", f"/switch {second['id']}", "/exit"], fake, active=first)
        self.assertIn("本轮未保存", text)
        self.assertEqual(self.store.read(first["id"])["turns"],
                         [{"user": "A 的问题", "final_answer": "答：A 的问题"}])


class TestSessionStore(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gagent-store-test-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.store = SessionStore(self.root / "sessions")

    def test_sessions_dir_is_project_relative(self):
        self.assertEqual(session_store.SESSIONS_DIR,
                         Path(session_store.__file__).resolve().parent / "sessions")
        self.assertEqual(SessionStore().root, session_store.SESSIONS_DIR)

    def test_ids_are_unique_and_same_title_is_allowed(self):
        first = self.store.create("同名")
        second = self.store.create("同名")
        self.assertNotEqual(first["id"], second["id"])
        self.assertNotIn("同名", self.store.path_of(first["id"]).name)
        metas, _broken = self.store.scan()
        self.assertEqual(sorted(meta["id"] for meta in metas), sorted([first["id"], second["id"]]))

    def test_session_file_shape_only_keeps_user_and_final_answer(self):
        session = self.store.create("形状")
        self.store.save_turn(session, "问题", "回答")
        raw = json.loads(self.store.path_of(session["id"]).read_text(encoding="utf-8"))
        self.assertEqual(set(raw), {"id", "title", "created_at", "updated_at", "turns"})
        self.assertEqual(raw["turns"], [{"user": "问题", "final_answer": "回答"}])
        text = self.store.path_of(session["id"]).read_text(encoding="utf-8")
        for token in ("tool_calls", "ToolMessage", "args"):
            self.assertNotIn(token, text)

    def test_writing_leaves_no_temp_files(self):
        session = self.store.create("原子")
        for index in range(3):
            self.store.save_turn(session, f"q{index}", f"a{index}")
        names = [path.name for path in self.store.root.iterdir()]
        self.assertEqual([name for name in names if name.startswith(".tmp-")], [])
        self.assertEqual(len(self.store.read(session["id"])["turns"]), 3)

    def test_corrupted_json_is_not_treated_as_empty(self):
        session = self.store.create("损坏")
        path = self.store.path_of(session["id"])
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with self.assertRaises(SessionCorrupted):
            self.store.read(session["id"])
        metas, broken = self.store.scan()
        self.assertEqual(metas, [])
        self.assertIn(path.name, broken[0])

    def test_turn_entries_must_be_objects(self):
        session = self.store.create("轮次坏了")
        path = self.store.path_of(session["id"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["turns"] = ["一个字符串"]
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(SessionCorrupted):
            self.store.read(session["id"])

    def test_id_must_match_the_file_name(self):
        session = self.store.create("串号")
        path = self.store.path_of(session["id"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["id"] = "ffffff"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(SessionCorrupted):
            self.store.read(session["id"])

    def test_read_missing_session_raises_not_found(self):
        with self.assertRaises(SessionNotFound):
            self.store.read("abcdef")

    def test_rejects_id_that_cannot_be_a_filename(self):
        for candidate in ("../escape", "a/b", "", "x" * 40):
            with self.assertRaises(SessionNotFound):
                self.store.read(candidate)

    def test_scan_sorts_by_recent_update(self):
        older = self.store.create("先建")
        newer = self.store.create("后建")
        self.store.save_turn(newer, "q", "a")
        metas, _broken = self.store.scan()
        self.assertEqual([meta["id"] for meta in metas], [newer["id"], older["id"]])

    def test_titles_may_stay_empty_and_are_cleaned(self):
        self.assertEqual(self.store.create()["title"], "")
        self.assertEqual(self.store.create("  ")["title"], "")
        self.assertEqual(self.store.create("两\n行\t标题")["title"], "两 行 标题")
        self.assertEqual(self.store.create("很长" * 40)["title"], ("很长" * 40)[:40])


class ModelStub:
    """像 ChatOpenAI 一样提供 bind_tools/invoke，并记录模型真正看到的消息。"""

    def __init__(self, make_call=None, final="证据不足，无法确认。", fail=None):
        self.make_call, self.final, self.fail = make_call, final, fail
        self.rounds = self.final_calls = 0
        self.seen = []

    def bind_tools(self, registered):
        def reply(messages):
            self.seen.append(list(messages))
            self.rounds += 1
            if self.fail:
                raise self.fail
            call = self.make_call(self.rounds, messages) if self.make_call else None
            return AIMessage(content="" if call else "12", tool_calls=[] if call is None else [call])
        return SimpleNamespace(invoke=reply)

    def invoke(self, messages):
        self.final_calls += 1
        return AIMessage(content=self.final)


class TestModelMessages(unittest.TestCase):
    """真实的 BasicAgent 消息构造：只注入已完成问答，不把历史当本轮工具证据。"""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gagent-model-test-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        patcher = patch.object(agent, "LOGS_DIR", self.root / "logs")
        patcher.start()
        self.addCleanup(patcher.stop)

    def quiet_run(self, model, question="问题", **kwargs):
        with patch.object(agent, "ChatOpenAI", return_value=model):
            instance = agent.BasicAgent()
            console = io.StringIO()
            with redirect_stdout(console):
                result = instance.run(question, echo=False, **kwargs)
        return SimpleNamespace(result=result, console=console.getvalue(), instance=instance)

    def test_run_returns_the_answer_and_execution_status(self):
        history = [{"user": "美国幸福排名下降了吗？", "final_answer": "根据报告下降了。"},
                   {"user": "那芬兰呢？", "final_answer": "芬兰仍排第一。", "多余字段": "忽略"}]
        run = self.quiet_run(ModelStub(), "两国的差距怎么解释", history=history)
        self.assertEqual(run.result.answer, "12")
        self.assertTrue(run.result.ok)
        self.assertEqual(run.result.user_prompt, "两国的差距怎么解释")
        self.assertEqual(run.console, "")

    def test_message_order_and_types(self):
        from langchain_core.messages import SystemMessage
        model = ModelStub()
        history = [{"user": "美国幸福排名下降了吗？", "final_answer": "根据报告下降了。"},
                   {"user": "那芬兰呢？", "final_answer": "芬兰仍排第一。"}]
        self.quiet_run(model, "两国的差距怎么解释", history=history)
        sent = model.seen[0]
        self.assertIsInstance(sent[0], SystemMessage)
        self.assertEqual([type(message) for message in sent[1:]],
                         [HumanMessage, AIMessage, HumanMessage, AIMessage, HumanMessage])
        self.assertEqual([sent[1].content, sent[2].content],
                         ["美国幸福排名下降了吗？", "根据报告下降了。"])
        self.assertEqual(sent[-1].content, "两国的差距怎么解释")
        self.assertFalse([m for m in sent if isinstance(m, ToolMessage)])
        self.assertFalse([m for m in sent if getattr(m, "tool_calls", None)])

    def test_history_is_not_counted_as_current_round_tool_evidence(self):
        history = [{"user": "查过了吗",
                    "final_answer": "我看过 https://en.wikipedia.org/wiki/X 和 https://a.example/x"}]
        seeded = agent._history_messages(history)
        self.assertFalse(agent._research_complete(seeded + [HumanMessage(content="再查")]))
        self.assertIsNone(agent._stop_reason(seeded))
        self.assertEqual(agent._required_visit_calls(seeded, 1), [])
        self.assertEqual(seeded[1].tool_calls, [])

    def test_last_transcript_only_covers_the_current_round(self):
        model = ModelStub()
        history = [{"user": "旧问题", "final_answer": "旧回答"}]
        run = self.quiet_run(model, "新问题", history=history)
        self.assertNotIn("旧问题", run.instance.last_transcript)
        self.assertNotIn("旧回答", run.instance.last_transcript)
        self.assertIn("新问题", run.instance.last_transcript)

    def test_injected_history_does_not_change_round_counting(self):
        model = ModelStub(make_call=lambda n, _m: {"name": "web_search", "args": {"query": f"q{n}"},
                                                   "id": str(n)} if n <= 2 else None)
        history = [{"user": f"旧{i}", "final_answer": f"答{i}"} for i in range(6)]
        run = self.quiet_run(model, "新问题", history=history)
        self.assertEqual(run.result.answer, "12")
        self.assertTrue(run.result.ok)

    def test_model_request_failure_is_reported_as_not_ok(self):
        class Boom(Exception):
            pass

        run = self.quiet_run(ModelStub(fail=Boom()), "问题")
        self.assertFalse(run.result.ok)
        self.assertIn("模型请求失败", run.result.answer)
        self.assertEqual(run.console, "")

    def test_quiet_run_writes_the_full_log_with_the_session_id(self):
        model = ModelStub(make_call=lambda n, _m: {"name": "visit_webpage",
                                                  "args": {"url": "https://example.org/x"},
                                                  "id": str(n)} if n == 1 else None)
        failed = ToolMessage(content='{"url": "https://example.org/x"}', name="visit_webpage",
                             tool_call_id="1", status="error")
        events = []
        with patch.object(agent, "ToolNode", return_value=SimpleNamespace(invoke=lambda _state: {"messages": [failed]})):
            with patch.object(agent, "ChatOpenAI", return_value=model):
                instance = agent.BasicAgent()
                console = io.StringIO()
                with redirect_stdout(console):
                    result = instance.run("读网页", session_id="abc123", echo=False,
                                          progress=lambda *event: events.append(event))
        log_files = sorted((self.root / "logs").glob("*.txt"))
        self.assertEqual(len(log_files), 1)
        content = log_files[0].read_text(encoding="utf-8")
        self.assertIn("[session] abc123", content)
        self.assertIn("[tool call 1/12] visit_webpage", content)
        self.assertIn("[final] " + result.answer, content)
        self.assertEqual(events[0], ("tool_call", "visit_webpage", ""))
        self.assertEqual(events[-1][:2], ("tool_error", "visit_webpage"))
        self.assertEqual(console.getvalue(), "")
        self.assertEqual(result.user_prompt, "读网页")

    def test_default_call_keeps_the_old_single_round_console_output(self):
        model = ModelStub()
        with patch.object(agent, "ChatOpenAI", return_value=model):
            instance = agent.BasicAgent()
            console = io.StringIO()
            with redirect_stdout(console):
                answer = instance("只有一轮")
        text = console.getvalue()
        self.assertEqual(answer, "12")
        self.assertIn("[final] 12", text)
        self.assertIn("本次运行日志", text)
        self.assertEqual(model.seen[0][-1].content, "只有一轮")
        self.assertEqual(len(model.seen), 1)


class TestExecuteToolsProgress(unittest.TestCase):
    def tool_call(self, name, args, call_id):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])

    def test_cached_calls_emit_no_progress(self):
        requested = self.tool_call("web_search", {"query": "芬兰 排名"}, "c1")
        done = ToolMessage(content='{"results": []}', name="web_search", tool_call_id="c1",
                           status="success")
        executed = []

        def node(state):
            executed.append([call["id"] for call in state["messages"][-1].tool_calls])
            return {"messages": [done]}

        events = []
        state = {"messages": [requested, done, self.tool_call("web_search", {"query": "芬兰 排名"}, "c2")]}
        out = agent._execute_tools(node, state, lambda *event: events.append(event))
        self.assertEqual(executed, [])
        self.assertEqual(events, [])
        self.assertEqual(out["messages"][0].tool_call_id, "c2")
        self.assertEqual(out["messages"][0].content, done.content)

    def test_error_progress_keeps_one_short_line(self):
        failed = ToolMessage(content="第一行 HTTP 404: 资源不存在\n" + "长" * 500,
                             name="visit_webpage", tool_call_id="c3", status="error")

        def node(_state):
            return {"messages": [failed]}

        events = []
        agent._execute_tools(SimpleNamespace(invoke=node),
                             {"messages": [self.tool_call("visit_webpage", {"url": "https://x/1"}, "c3")]},
                             lambda *event: events.append(event))
        self.assertEqual([event[0] for event in events], ["tool_call", "tool_error"])
        self.assertEqual(events[1][1], "visit_webpage")
        self.assertTrue(events[1][2].startswith("第一行 HTTP 404"))
        self.assertLessEqual(len(events[1][2]), 91)

    def test_success_emits_only_the_call(self):
        ok = ToolMessage(content='{"ok": true}', name="run_python", tool_call_id="c4", status="success")
        events = []
        agent._execute_tools(SimpleNamespace(invoke=lambda _state: {"messages": [ok]}),
                             {"messages": [self.tool_call("run_python", {"code": "print(1)"}, "c4")]},
                             lambda *event: events.append(event))
        self.assertEqual(events, [("tool_call", "run_python", "")])


class StubRagClient:
    """给提示线程用的假文件服务客户端：不联网、不起进程。"""

    def __init__(self, batches=(), cursor=0, failure=None):
        self.batches = [list(batch) for batch in batches]
        self.saved = []
        self.start_cursor = cursor
        self.calls = []
        self.failure = failure

    def read_cursor(self):
        return self.start_cursor

    def write_cursor(self, value):
        self.saved.append(value)

    def events_after(self, cursor=0, limit=50, start=False):
        self.calls.append((cursor, start))
        if self.failure:
            raise self.failure
        batch = self.batches.pop(0) if self.batches else []
        latest = batch[-1]["id"] if batch else cursor
        return {"events": batch, "cursor": latest, "latest": latest}


def event(event_id, kind, message):
    return {"id": event_id, "kind": kind, "message": message, "detail": None,
            "created_at": "2026-09-25T10:00:00+08:00", "job_id": f"job_{event_id}"}


class TestRagNotices(unittest.TestCase):
    """入库完成/失败提示：属于界面，不进会话，也不打断输入。"""

    def notifier(self, lines, poll_seconds=0.01, **kwargs):
        client = StubRagClient(**kwargs)
        notice = cli.RagNotifier(lines.append, poll_seconds=poll_seconds, client=client)
        notice.cursor = client.read_cursor()
        return notice, client, lines

    def test_completion_shown_once_and_cursor_saved(self):
        lines = []
        batches = [[event(1, "indexed", "WHR25.pdf 已加入长期记忆")], [], [event(2, "indexed", "x.xlsx 已加入长期记忆")]]
        notice, client, _ = self.notifier(lines, batches=batches)
        notice.poll_once()
        notice.poll_once()
        self.assertEqual(lines, ["WHR25.pdf 已加入长期记忆"])
        self.assertEqual(client.saved, [1], "消费位置要落盘，否则下次启动会重复弹")
        notice.poll_once()
        self.assertEqual(lines, ["WHR25.pdf 已加入长期记忆", "x.xlsx 已加入长期记忆"])
        self.assertEqual(client.calls[1], (1, False), "轮询不该顺带把服务拉起来")

    def test_failure_notice_points_at_memory_command(self):
        lines = []
        notice, _, _ = self.notifier(lines, batches=[[event(3, "failed", "scan.pdf 加入长期记忆失败，已停止自动重试")]])
        notice.poll_once()
        self.assertEqual(len(lines), 1)
        self.assertIn("已停止自动重试", lines[0])
        self.assertIn("/memory", lines[0])

    def test_notice_holds_while_user_is_typing(self):
        lines = []
        notice, _, _ = self.notifier(lines)
        notice.begin_input()
        notice.deliver(["WHR25.pdf 已加入长期记忆"])
        self.assertEqual(lines, [], "正在编辑的输入不该被插队")
        notice.end_input()
        self.assertEqual(lines, ["WHR25.pdf 已加入长期记忆"])

    def test_service_outage_is_silent_and_keeps_loop_alive(self):
        lines = []
        notice, client, _ = self.notifier(lines, failure=RuntimeError("服务没起来"))
        self.assertEqual(notice.poll, 0.01)
        notice.start()
        time.sleep(0.05)
        notice.stop()
        self.assertEqual(lines, [])
        self.assertEqual(client.saved, [], "取不到事件时不能推进游标")

    def test_polling_thread_does_not_start_the_service(self):
        lines = []
        notice, client, _ = self.notifier(lines)
        notice.start()
        time.sleep(0.05)
        notice.stop()
        self.assertTrue(client.calls)
        self.assertFalse(any(start for _, start in client.calls))


class TestRagCommands(CliHarness):
    """/memory 与 /retry：未完成不提示成功，失败原因在这里可查、可手动重试。"""

    def state(self, jobs=(), events=()):
        return patch.object(cli.rag_client, "state", return_value={"jobs": list(jobs), "events": list(events),
                                                            "model_ready": True}), \
            patch.object(cli.rag_client, "health", return_value={"documents": 2, "vectors": 9,
                                                          "jobs_pending": 1, "jobs_failed": 1})

    def test_memory_lists_pending_and_failed_with_reason(self):
        jobs = [{"job_id": "job_1", "status": "queued", "attempts": 0, "max_attempts": 3,
                 "file_name": "WHR25.pdf", "last_error": None},
                {"job_id": "job_2", "status": "failed", "attempts": 3, "max_attempts": 3,
                 "file_name": "scan.pdf", "last_error": "扫描版 PDF 无可提取文字（共 12 页），本版本不支持 OCR"}]
        state, health = self.state(jobs=jobs)
        with state, health:
            _code, text, _chat, _fake = self.drive(["/memory", "/exit"])
        self.assertIn("WHR25.pdf", text)
        self.assertIn("job_1", text)
        self.assertIn("不支持 OCR", text)
        self.assertIn("/retry job_2", text)
        self.assertNotIn("已加入长期记忆", text.split("最近事件")[0])

    def test_memory_says_so_when_nothing_is_running(self):
        state, health = self.state()
        with state, patch.object(cli.rag_client, "health", return_value=None):
            _code, text, _chat, _fake = self.drive(["/memory", "/exit"])
        self.assertIn("未运行", text)
        self.assertIn("没有待办或失败的任务", text)

    def test_memory_reports_unavailable_service(self):
        with patch.object(cli.rag_client, "state", side_effect=RuntimeError("文件服务未能启动")), \
                patch.object(cli.rag_client, "health", return_value=None):
            _code, text, _chat, _fake = self.drive(["/memory", "/exit"])
        self.assertIn("读不到长期记忆状态", text)

    def test_retry_submits_new_round_and_reports_it(self):
        with patch.object(cli.rag_client, "retry_job", return_value={"job_id": "job_9", "status": "queued",
                                                              "round": 2, "attempts": 0}) as retry:
            _code, text, _chat, _fake = self.drive(["/retry job_2", "/exit"])
        retry.assert_called_once_with("job_2")
        self.assertIn("已提交重试", text)
        self.assertIn("job_9", text)

    def test_retry_usage_and_rejection(self):
        with patch.object(cli.rag_client, "retry_job", side_effect=RuntimeError("该文件版本已经入库，无需重试")) as retry:
            _code, text, _chat, _fake = self.drive(["/retry", "/retry a b", "/retry job_3", "/exit"])
        retry.assert_called_once_with("job_3")
        self.assertEqual(text.count("用法：/retry <job_id>"), 2)
        self.assertIn("无需重试", text)

    def test_notice_shows_during_a_turn_without_entering_history(self):
        lines = []
        notice = cli.RagNotifier(lambda text: lines.append(text), poll_seconds=0, client=StubRagClient())
        code, text, chat, fake = self.drive(["这个文件讲了什么", "/exit"], notifier=notice,
                                            fake=FakeAgent(events=[("tool_call", "read_pdf", "")]))
        # 服务在轮次之间给出完成事件
        notice.deliver(["WHR25.pdf 已加入长期记忆"])
        notice.flush()
        self.assertIn("正在读取 PDF", text)
        self.assertNotIn("已加入长期记忆", "".join(turn["user"] + turn["final_answer"]
                                                 for turn in chat.session["turns"]),
                         "完成提示是对话界面的事，不能写进会话历史")

    def test_help_covers_the_memory_commands(self):
        _code, text, _chat, _fake = self.drive(["/help", "/exit"])
        self.assertIn("/memory", text)
        self.assertIn("/retry", text)


class TestRagInput(unittest.TestCase):
    """每轮提问前的检索结果如何进消息：只包装本轮输入，不污染会话历史。"""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gagent-rag-input-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        patcher = patch.object(agent, "LOGS_DIR", self.root / "logs")
        patcher.start()
        self.addCleanup(patcher.stop)

    def hit(self, index, kind, **extra):
        base = {"source_id": f"R{index}", "kind": kind, "file_name": "WHR24.pdf" if kind == "pdf_chunk"
                else "financial.xlsx", "document_id": "doc_1", "version": "v" * 12,
                "local_path": "/tmp/x", "source_url": "https://e.org/f", "score": 0.9,
                "text": "Finland ranks first with 7.741" if kind == "pdf_chunk" else "微软示例财务工作簿"}
        return {**base, **extra}

    def run_with(self, hits, reason=None, question="美国与芬兰的差距有多大？", **kwargs):
        payload = {"results": hits} if reason is None else {"results": [], "reason": reason}
        seen = {}

        def spy(query, **options):
            seen["query"] = query
            seen["options"] = options
            return payload

        model = ModelStub()
        with patch.object(agent.rag_client, "search", spy), patch.object(agent, "ChatOpenAI", return_value=model):
            with redirect_stdout(io.StringIO()):
                result = agent.BasicAgent().run(question, echo=False, **kwargs)
        return SimpleNamespace(result=result, model=model, seen=seen,
                               message=model.seen[0][-1].content)

    def test_hits_are_wrapped_as_json_in_the_user_message(self):
        out = self.run_with([self.hit(1, "pdf_chunk", page=12, start_char=0, end_char=25),
                             self.hit(2, "excel_summary")])
        body = json.loads(out.message)
        self.assertEqual(sorted(body), ["retrieved_context", "user_input"])
        self.assertEqual(body["user_input"], "美国与芬兰的差距有多大？")
        pdf, excel = body["retrieved_context"]
        self.assertEqual([pdf["source_id"], excel["source_id"]], ["R1", "R2"])
        self.assertEqual(pdf["page"], 12)
        self.assertIn("start_char", pdf)
        self.assertNotIn("page", excel)
        self.assertNotIn("score", pdf)

    def test_session_history_still_gets_the_plain_question(self):
        out = self.run_with([self.hit(1, "pdf_chunk", page=3)])
        self.assertEqual(out.result.user_prompt, "美国与芬兰的差距有多大？")
        self.assertNotIn("retrieved_context", out.result.user_prompt)

    def test_no_hits_keep_the_plain_message(self):
        out = self.run_with([])
        self.assertEqual(out.message, "美国与芬兰的差距有多大？")
        self.assertNotIn("{", out.message)

    def test_query_is_this_round_text_without_history_or_attachment(self):
        history = [{"user": "上一轮问题", "final_answer": "上一轮回答 " * 30}]
        out = self.run_with([], reason="没有达到门槛的结果", history=history,
                            file_url="https://e.org/whr24.pdf", file_name="WHR24.pdf")
        self.assertEqual(out.seen["query"], "美国与芬兰的差距有多大？")
        self.assertNotIn("资料", out.seen["query"])
        self.assertIn("【资料】WHR24.pdf", out.message)   # 附件说明仍随问题一起给模型

    def test_retrieval_failure_does_not_break_the_round(self):
        out = self.run_with([], reason="文件服务未能启动")
        self.assertTrue(out.result.ok)
        self.assertEqual(out.result.answer, "12")
        self.assertEqual(out.message, "美国与芬兰的差距有多大？")

    def test_stale_version_flag_is_passed_through(self):
        hits = [self.hit(1, "pdf_chunk", page=4, start_char=0, newer_version_pending=True,
                         note="该文件存在更新版本尚未完成入库")]
        body = json.loads(self.run_with(hits).message)
        item = body["retrieved_context"][0]
        self.assertTrue(item["newer_version_pending"])
        self.assertIn("更新版本", item["note"])

    def test_system_prompt_states_the_file_answer_rules(self):
        prompt = agent.SYSTEM_PROMPT
        for token in ("user_input", "retrieved_context", "pdf_chunk", "excel_summary",
                      "不能拿摘要", "页码", "本机文件已经足够"):
            self.assertIn(token, prompt)



if __name__ == "__main__":
    unittest.main()
