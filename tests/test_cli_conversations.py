"""跨会话回忆的客户端与 CLI 接入测试。

全部用假 client 与假同步对象，不联网、不拉起本机文件服务；服务端接口还没落地时也能跑。
覆盖：保存/补存后的后台同步通知、启动时的全量同步开关、/memory 的对话段、
/retry 的对话任务、入库提示不刷屏、回忆进度文字，以及客户端请求的形状与降级。
"""
from types import SimpleNamespace
import io
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

# 与 tests/test_cli.py 一致：单测不为文件服务拉起后台进程。
os.environ.setdefault("GAGENT_RAG_DISABLE", "1")

import api_setup
import cli
import G_agent as agent
from rag import client as rag_client
from rag import state as rag_state
import session_store
from session_store import SessionStore


def wait_until(predicate, timeout: float = 3.0) -> bool:
    """等后台线程做出一件事；超时也返回最后一次判断结果，不 sleep 到天荒地老。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class FakeAgent:
    """替代 BasicAgent：记录每轮入参，按脚本返回结果，绝不联网。"""

    def __init__(self, script=None, events=()):
        self.script = dict(script or {})
        self.calls = []
        self.events = list(events)

    def run(self, question, *, file_url=None, file_name=None, history=None,
            progress=None, session_id=None, echo=True):
        self.calls.append(SimpleNamespace(question=question, session_id=session_id, echo=echo))
        case = self.script.get(question, {})
        if case.get("raises"):
            raise case["raises"]
        if progress:
            for event in self.events:
                progress(*event)
        return agent.AgentResult(user_prompt=case.get("user_prompt", question),
                                 answer=case.get("answer", f"答：{question}"),
                                 ok=case.get("ok", True))


class FakeSync:
    """替代后台同步：只记录通知，不联网；可按需在 start/notify 上抛异常。"""

    def __init__(self, fail=False):
        self.fail = fail
        self.started = 0
        self.stopped = 0
        self.notified: list[str] = []

    def start(self):
        self.started += 1
        if self.fail:
            raise RuntimeError("同步线程没起来")

    def notify(self, session_id):
        if self.fail:
            raise RuntimeError("通知失败")
        self.notified.append(session_id)

    def stop(self):
        self.stopped += 1


class StubSyncClient:
    """给 ConversationSync 用的假文件服务客户端：记录同步请求，可脚本化失败。"""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls: list[tuple] = []

    def conversations_sync(self, session_id=None, *, all=False, start=True):
        self.calls.append((session_id, bool(all)))
        if self.fail:
            raise RuntimeError("文件服务未能启动")
        return {"sessions": 1, "queued": 1}


class StubNoticeClient:
    """给提示线程用的假客户端：按批返回事件，不联网。"""

    def __init__(self, batches=()):
        self.batches = [list(batch) for batch in batches]
        self.saved: list[int] = []

    def read_cursor(self):
        return 0

    def write_cursor(self, value):
        self.saved.append(value)

    def events_after(self, cursor=0, limit=50, start=False):
        batch = self.batches.pop(0) if self.batches else []
        latest = batch[-1]["id"] if batch else cursor
        return {"events": batch, "cursor": latest, "latest": latest}


class ConversationHarness(unittest.TestCase):
    """把 Chat 接到脚本化输入上；默认注入假同步对象，单测里不会真的联网。"""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gagent-conv-test-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.store = SessionStore(self.root / "sessions")
        patcher = patch.object(agent, "LOGS_DIR", self.root / "logs")
        patcher.start()
        self.addCleanup(patcher.stop)

    def drive(self, inputs, fake=None, active=None, sync=None):
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

        silent = cli.RagNotifier(lambda text: None, poll_seconds=0, client=StubNoticeClient())
        chat = cli.Chat(self.store, fake, out=output, input_fn=input_fn, notifier=silent,
                        sync=sync if sync is not None else FakeSync())
        return chat.start(), output.getvalue(), chat, fake


class TestSaveTurnNotifiesSync(ConversationHarness):
    """保存成功才通知后台补索引；失败和补存路径都要走对。"""

    def test_successful_save_notifies_the_session(self):
        session = self.store.create("记忆")
        sync = FakeSync()
        _code, _text, _chat, _fake = self.drive(["第一问", "/exit"], active=session, sync=sync)
        self.assertEqual(sync.started, 1, "启动时该发起一次后台同步")
        self.assertEqual(sync.notified, [session["id"]])
        self.assertEqual(sync.stopped, 1)
        self.assertEqual(len(self.store.read(session["id"])["turns"]), 1)

    def test_save_oserror_does_not_notify(self):
        session = self.store.create("写入失败")
        sync = FakeSync()
        output = io.StringIO()
        chat = cli.Chat(self.store, FakeAgent(), out=output, input_fn=lambda _prompt: "/exit",
                        sync=sync)
        chat.session = session
        with patch.object(SessionStore, "save_turn", side_effect=OSError(13, "Permission denied")):
            chat.save_turn("问题", "回答")
        self.assertEqual(sync.notified, [])
        self.assertEqual([item["id"] for item in chat.unsaved], [session["id"]])
        self.assertIn("本轮未保存", output.getvalue())

    def test_flush_unsaved_notifies_after_the_retry_succeeds(self):
        """首次保存失败的轮次补存成功后也要通知，否则它会永远漏出索引。"""
        session = self.store.create("补存")
        sync = FakeSync()
        real_write = SessionStore.write
        state = {"fail": True}

        def flaky_write(self, item):
            if state["fail"]:
                state["fail"] = False
                raise OSError(13, "Permission denied")
            real_write(self, item)

        with patch.object(SessionStore, "write", flaky_write):
            code, text, chat, _fake = self.drive(["这一轮要保住", "/sessions", "/exit"],
                                                 active=session, sync=sync)
        self.assertEqual(code, 0)
        self.assertIn("本轮未保存", text)
        self.assertIn("已补存", text)
        self.assertEqual(sync.notified, [session["id"]], "只有补存成功这一条通知，保存失败时不该通知")
        self.assertEqual([turn["user"] for turn in chat.session["turns"]], ["这一轮要保住"])


class TestConversationSyncSwitch(ConversationHarness):
    """启动时的全量同步：默认开关、禁用时不发生、失败不影响启动。"""

    def test_enabled_flag_follows_the_environment(self):
        with patch.dict(os.environ, {"GAGENT_RAG_DISABLE": "1"}):
            self.assertFalse(cli.ConversationSync(client=StubSyncClient()).enabled)
        with patch.dict(os.environ, {"GAGENT_RAG_DISABLE": "", "GAGENT_CONVERSATION_SYNC": "0"}):
            self.assertFalse(cli.ConversationSync(client=StubSyncClient()).enabled)
        with patch.dict(os.environ, {"GAGENT_RAG_DISABLE": "", "GAGENT_CONVERSATION_SYNC": "1"}):
            self.assertTrue(cli.ConversationSync(client=StubSyncClient()).enabled)

    def test_startup_syncs_all_sessions_once_and_survives_failure(self):
        client = StubSyncClient(fail=True)
        sync = cli.ConversationSync(client=client, enabled=True, idle_seconds=0.01)
        code, text, _chat, _fake = self.drive(["/exit"], sync=sync)
        self.assertTrue(wait_until(lambda: bool(client.calls)))
        time.sleep(0.02)
        self.assertEqual(client.calls, [(None, True)], "启动时只该投一次全量同步，且不带 session_id")
        self.assertEqual(code, 0, "后台同步失败不能影响启动")
        self.assertIn("输入 /help 查看命令。", text)
        sync.stop()

    def test_disabled_never_touches_the_service(self):
        client = StubSyncClient()
        sync = cli.ConversationSync(client=client, idle_seconds=0.01)  # enabled 由环境推导
        self.assertFalse(sync.enabled)
        active = self.store.create("禁用同步")
        _code, _text, _chat, _fake = self.drive(["提问", "/exit"], active=active, sync=sync)
        time.sleep(0.05)
        self.assertEqual(client.calls, [], "GAGENT_RAG_DISABLE=1 时一个请求都不该发")

    def test_notify_syncs_every_saved_session(self):
        client = StubSyncClient()
        sync = cli.ConversationSync(client=client, enabled=True, idle_seconds=0.01)
        sync.notify("abc123")
        sync.notify("def456")
        self.assertTrue(wait_until(lambda: {"abc123", "def456"} <= {sid for sid, _ in client.calls}))
        sync.stop()
        self.assertTrue(all(not all_sessions for _sid, all_sessions in client.calls),
                        "单会话通知不该升级成全量同步")

    def test_sync_failure_is_silent(self):
        client = StubSyncClient(fail=True)
        sync = cli.ConversationSync(client=client, enabled=True, idle_seconds=0.01)
        sync.start()
        sync.notify("abc123")
        self.assertTrue(wait_until(lambda: len(client.calls) >= 2))
        sync.stop()
        self.assertIn((None, True), client.calls)
        self.assertIn(("abc123", False), client.calls)


class TestMemoryConversations(ConversationHarness):
    """/memory 的对话记忆段与 /retry：失败原因可查、可重试。"""

    def memory(self, state, health):
        with patch.object(cli.rag_client, "state", return_value=state), \
                patch.object(cli.rag_client, "health", return_value=health):
            return self.drive(["/memory", "/exit"])

    def test_memory_shows_conversation_counts_and_failure_reason(self):
        state = {
            "jobs": [], "events": [], "model_ready": True,
            "conversations_indexed": 4, "conversations_pending": 2,
            "conversation_jobs": [
                {"job_id": "cjob_1", "session_id": "abc123", "status": "failed", "attempts": 3,
                 "max_attempts": 3, "round": 2, "last_error": "摘要生成失败：模型超时",
                 "version_id": "v" * 12, "version_status": "failed", "turn_count": 7}],
        }
        health = {"documents": 1, "vectors": 2, "jobs_pending": 0, "jobs_failed": 0,
                  "conversations": 4, "conversation_vectors": 21,
                  "conversation_jobs_pending": 2, "conversation_jobs_failed": 1}
        _code, text, _chat, _fake = self.memory(state, health)
        self.assertIn("对话记忆：已索引 4 个会话  向量 21 条  待处理 2  失败 1", text)
        self.assertIn("对话索引失败任务：", text)
        self.assertIn("cjob_1", text)
        self.assertIn("abc123", text)
        self.assertIn("模型超时", text)
        self.assertIn("重试用 /retry cjob_1", text)

    def test_memory_without_conversation_failures_prints_only_counts(self):
        state = {"jobs": [], "events": [], "model_ready": True, "conversation_jobs": [
            {"job_id": "cjob_2", "session_id": "abc123", "status": "queued", "attempts": 0,
             "max_attempts": 3, "round": 1, "last_error": None}]}
        health = {"documents": 0, "vectors": 0, "jobs_pending": 0, "jobs_failed": 0,
                  "conversations": 1, "conversation_vectors": 3,
                  "conversation_jobs_pending": 1, "conversation_jobs_failed": 0}
        _code, text, _chat, _fake = self.memory(state, health)
        self.assertIn("对话记忆：已索引 1 个会话  向量 3 条  待处理 1  失败 0", text)
        self.assertNotIn("对话索引失败任务：", text)
        self.assertNotIn("/retry", text)

    def test_memory_reports_unavailable_conversation_vectors(self):
        state = {"jobs": [], "events": [], "model_ready": True}
        health = {"documents": 0, "vectors": 0, "jobs_pending": 0, "jobs_failed": 0,
                  "conversations": 0, "conversation_vectors": -1,
                  "conversation_jobs_pending": 0, "conversation_jobs_failed": 0}
        _code, text, _chat, _fake = self.memory(state, health)
        self.assertIn("对话记忆：已索引 0 个会话  向量 0 条  待处理 0  失败 0", text)
        self.assertIn("未就绪", text)

    def test_retry_routes_a_conversation_job(self):
        with patch.object(cli.rag_client, "retry_job",
                          return_value={"job_id": "cjob_9", "status": "queued", "round": 1,
                                        "attempts": 0}) as retry:
            _code, text, _chat, _fake = self.drive(["/retry cjob_7", "/exit"])
        retry.assert_called_once_with("cjob_7")
        self.assertIn("已提交重试", text)
        self.assertIn("cjob_9", text)
        self.assertIn("对话索引", text)


class TestSessionsDirFlag(unittest.TestCase):
    """--sessions-dir X：CLI 保存、后台服务启动、服务侧同步/索引/检索/摘要都只能用 X。"""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="gagent-sessions-dir-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.custom = self.root / "custom-sessions"
        self.saved_dir = session_store.SESSIONS_DIR
        self.saved_env = os.environ.get("GAGENT_SESSIONS_DIR")
        self.addCleanup(self._restore)

    def _restore(self):
        session_store.SESSIONS_DIR = self.saved_dir
        if self.saved_env is None:
            os.environ.pop("GAGENT_SESSIONS_DIR", None)
        else:
            os.environ["GAGENT_SESSIONS_DIR"] = self.saved_env

    def run_cli(self, *, custom_dir=True) -> int:
        argv = ["--sessions-dir", str(self.custom)] if custom_dir else []
        output = io.StringIO()
        pending = iter(["/exit"])

        def input_fn(_prompt):
            return next(pending)

        with patch.dict(os.environ, {"GAGENT_RAG_DISABLE": "1", "GAGENT_CONVERSATION_SYNC": "0"}):
            return cli.main(argv, agent=FakeAgent(), out=output, input_fn=input_fn)

    def test_the_flag_is_the_only_sessions_dir_for_the_whole_run(self):
        self.assertEqual(self.run_cli(), 0)
        # CLI 真的把会话存在 X 里（不是存到默认 sessions/）
        metas, broken = SessionStore(self.custom).scan()
        self.assertEqual(broken, [])
        self.assertEqual(len(metas), 1, "启动时新建的会话必须落在 --sessions-dir 指定的目录")
        # 本次运行唯一的会话目录：rag/client.py 启动服务时读它，服务侧
        # service._sessions_dir()/conversations._sessions_dir() 也读它
        self.assertEqual(session_store.SESSIONS_DIR, self.custom)
        self.assertEqual(session_store.default_sessions_dir(), self.custom)

    def test_background_service_starts_with_the_same_sessions_dir(self):
        self.assertEqual(self.run_cli(), 0)
        captured = {}

        class DummyProcess:
            pid = 4242

        def fake_popen(command, **kwargs):
            captured["command"] = list(command)
            captured["env"] = dict(kwargs.get("env") or {})
            return DummyProcess()

        with patch.object(rag_state, "ensure_dirs", lambda: None), \
                patch.object(rag_state, "logs_dir", lambda: self.root), \
                patch.object(rag_client.subprocess, "Popen", fake_popen):
            rag_client._spawn()
        self.assertEqual(captured["env"]["GAGENT_SESSIONS_DIR"], str(self.custom),
                         "后台服务必须继承 CLI 实际使用的会话目录，否则它会去索引默认 sessions/")

    def test_without_the_flag_the_default_dir_is_used(self):
        default = self.root / "default-sessions"
        with patch.object(cli, "SESSIONS_DIR", default):
            self.assertEqual(self.run_cli(custom_dir=False), 0)
        metas, _broken = SessionStore(default).scan()
        self.assertEqual(len(metas), 1, "不传参数时仍用原来的默认目录")
        self.assertEqual(session_store.SESSIONS_DIR, default)


class TestNoticeText(unittest.TestCase):
    """入库提示节流：对话索引成功不刷屏，失败要指向 /memory。"""

    def test_conversation_indexed_is_quiet(self):
        self.assertEqual(cli._notice_text({"kind": "conversation_indexed",
                                           "message": "会话 abc123 已加入对话记忆"}), "")

    def test_conversation_failed_points_at_memory(self):
        line = cli._notice_text({"kind": "conversation_failed", "message": "会话 abc123 索引失败"})
        self.assertIn("索引失败", line)
        self.assertIn("/memory", line)

    def test_conversation_failed_without_message_still_notices(self):
        self.assertIn("/memory", cli._notice_text({"kind": "conversation_failed"}))

    def test_file_events_keep_their_behaviour(self):
        self.assertEqual(cli._notice_text({"kind": "indexed", "message": "WHR25.pdf 已加入长期记忆"}),
                         "WHR25.pdf 已加入长期记忆")
        self.assertIn("/memory", cli._notice_text({"kind": "failed", "message": "scan.pdf 失败"}))

    def test_notifier_drops_indexed_and_shows_failed(self):
        lines = []
        client = StubNoticeClient(batches=[[
            {"id": 1, "kind": "conversation_indexed", "message": "会话已索引"},
            {"id": 2, "kind": "conversation_failed", "message": "会话索引失败"}]])
        notice = cli.RagNotifier(lines.append, poll_seconds=0, client=client)
        notice.poll_once()
        self.assertEqual(len(lines), 1)
        self.assertIn("会话索引失败", lines[0])
        self.assertIn("/memory", lines[0])
        self.assertEqual(client.saved, [2])


class TestRecallProgress(ConversationHarness):
    """回忆工具的进度文字：主标签与细分阶段都不该露出内部名字。"""

    def test_recall_tool_label(self):
        session = self.store.create("回忆")
        fake = FakeAgent(events=[("tool_call", "recall_conversation", "")])
        _code, text, _chat, _fake = self.drive(["上次那个结论是什么", "/exit"],
                                               fake=fake, active=session)
        self.assertIn("● 正在查找历史对话……", text)

    def test_recall_subphase_shows_a_readable_line(self):
        session = self.store.create("回忆")
        fake = FakeAgent(events=[("tool_call", "recall_conversation", ""),
                                 ("recall_candidates", "recall_conversation", ""),
                                 ("recall_summary", "recall_conversation", "")])
        _code, text, _chat, _fake = self.drive(["上次那个结论是什么", "/exit"],
                                               fake=fake, active=session)
        self.assertIn("● 正在整理相关会话……", text)
        self.assertNotIn("recall_candidates", text)
        self.assertNotIn("recall_summary", text)

    def test_help_mentions_cross_session_recall(self):
        _code, text, _chat, _fake = self.drive(["/help", "/exit"])
        for token in ("/help", "/memory", "/retry", "/exit"):
            self.assertIn(token, text)
        self.assertIn("跨会话回忆", text)


class TestConversationClient(unittest.TestCase):
    """rag/client.py 的请求形状与降级：服务端未落地时用 mock 验证契约。"""

    def call(self, func, *args, **kwargs):
        with patch.object(rag_client, "_request", return_value={"ok": True}) as request:
            result = func(*args, **kwargs)
        return result, request

    def test_sync_one_session(self):
        _result, request = self.call(rag_client.conversations_sync, "abc123", start=False)
        self.assertEqual(request.call_args.args[:2], ("POST", "/v1/conversations/sync"))
        self.assertEqual(request.call_args.kwargs["payload"], {"session_id": "abc123", "all": False})
        self.assertFalse(request.call_args.kwargs["start"])

    def test_sync_all_drops_the_session_id(self):
        _result, request = self.call(rag_client.conversations_sync, "abc123", all=True)
        self.assertEqual(request.call_args.kwargs["payload"], {"session_id": None, "all": True})

    def test_search_payload(self):
        _result, request = self.call(rag_client.conversations_search, "芬兰排名",
                                     exclude_sessions=["abc123"], top_chunks=5, max_sessions=2,
                                     start=False)
        self.assertEqual(request.call_args.args[:2], ("POST", "/v1/conversations/search"))
        self.assertEqual(request.call_args.kwargs["payload"],
                         {"query": "芬兰排名", "exclude_sessions": ["abc123"],
                          "top_chunks": 5, "max_sessions": 2})

    def test_search_empty_query_is_invalid_without_a_request(self):
        with patch.object(rag_client, "_request") as request:
            result = rag_client.conversations_search("   ")
        request.assert_not_called()
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["hits"], [])

    def test_search_service_failure_degrades(self):
        with patch.object(rag_client, "_request", side_effect=RuntimeError("文件服务未能启动")):
            result = rag_client.conversations_search("芬兰排名")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["candidates"], [])
        self.assertIn("文件服务未能启动", result["reason"])

    def test_history_path_and_summary_params(self):
        _result, request = self.call(rag_client.conversation_history, "abc123", start=False)
        self.assertEqual(request.call_args.args[:2], ("GET", "/v1/conversations/abc123"))
        _result, request = self.call(rag_client.conversation_summary, "abc123", revision="rev1")
        self.assertEqual(request.call_args.args[:2], ("GET", "/v1/conversations/abc123/summary"))
        self.assertEqual(request.call_args.kwargs["params"], {"revision": "rev1"})
        _result, request = self.call(rag_client.conversation_summary, "abc123")
        self.assertIsNone(request.call_args.kwargs["params"])

    def test_put_summary_payload(self):
        _result, request = self.call(rag_client.put_conversation_summary, "abc123",
                                     source_revision="rev1", summary_schema_version=2,
                                     summary_json={"topic": "幸福报告"}, generated_at="2026-09-25T10:00:00+08:00",
                                     start=False)
        self.assertEqual(request.call_args.args[:2], ("PUT", "/v1/conversations/abc123/summary"))
        self.assertEqual(request.call_args.kwargs["payload"],
                         {"source_revision": "rev1", "summary_schema_version": 2,
                          "summary_json": {"topic": "幸福报告"},
                          "generated_at": "2026-09-25T10:00:00+08:00"})

    def test_recall_passes_settings_and_uses_the_recall_timeout(self):
        settings = {"API_KEY": "k", "MODEL": "m", "BASE_URL": "u"}
        _result, request = self.call(rag_client.recall_conversation, "上次那个结论",
                                     session_id="abc123", exclude_session="def456",
                                     include_summary=False, api_settings=settings)
        kwargs = request.call_args.kwargs
        self.assertEqual(request.call_args.args[:2], ("POST", "/v1/conversations/recall"))
        self.assertEqual(kwargs["timeout"], rag_client.RECALL_TIMEOUT)
        self.assertEqual(kwargs["attempts"], 1, "回忆可能很慢，不该再重试一遍")
        self.assertEqual(kwargs["payload"]["session_id"], "abc123")
        self.assertEqual(kwargs["payload"]["exclude_session"], "def456")
        self.assertFalse(kwargs["payload"]["include_summary"])
        self.assertEqual(kwargs["payload"]["api_settings"], settings)

    def test_recall_reads_the_active_api_settings_when_absent(self):
        settings = {"API_KEY": "k", "MODEL": "m", "BASE_URL": "u"}
        with patch.object(api_setup, "active_api_settings", return_value=settings), \
                patch.object(rag_client, "_request", return_value={"status": "found"}) as request:
            result = rag_client.recall_conversation("上次那个结论")
        self.assertEqual(result, {"status": "found"})
        self.assertEqual(request.call_args.kwargs["payload"]["api_settings"], settings)

    def test_recall_without_usable_settings_degrades(self):
        with patch.object(api_setup, "active_api_settings", side_effect=ValueError("bad config")), \
                patch.object(rag_client, "_request") as request:
            result = rag_client.recall_conversation("上次那个结论")
        request.assert_not_called()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["evidence"], [])
        self.assertIn("bad config", result["reason"])

    def test_recall_service_failure_degrades_instead_of_raising(self):
        with patch.object(rag_client, "_request", side_effect=RuntimeError("文件服务未能启动")):
            result = rag_client.recall_conversation("上次那个结论", api_settings={"API_KEY": "k"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["evidence"], [])
        self.assertIn("文件服务未能启动", result["reason"])

    def test_recall_empty_query_is_invalid_without_a_request(self):
        with patch.object(rag_client, "_request") as request:
            result = rag_client.recall_conversation("   ")
        request.assert_not_called()
        self.assertEqual(result, {"status": "invalid", "reason": "空问题", "evidence": []})

    def test_recall_timeout_constant_is_longer_than_a_plain_search(self):
        self.assertIsInstance(rag_client.RECALL_TIMEOUT, float)
        self.assertGreater(rag_client.RECALL_TIMEOUT, rag_client.SEARCH_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
