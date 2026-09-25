"""复杂调用链、失败收尾和文件解析测试，不调用真实模型或评分接口。"""
from contextlib import redirect_stdout
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import hashlib
import os
from pathlib import Path
import shutil
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import uuid

import httpx
import pandas as pd
from pypdf import PdfWriter
from pypdf.errors import PdfReadError
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from langchain_core.messages import AIMessage, ToolMessage

# 单测不为文件服务拉起后台进程；服务侧的调度与检索在集成脚本里单独验证。
os.environ.setdefault("GAGENT_RAG_DISABLE", "1")

import G_agent as agent
import agent_tools as tools
from rag import storage          # PDF/Excel 的下载、落盘与解析（已从 agent_tools 迁出）
from rag import state as rag_state   # 资料库状态：文档/版本/入库任务，替代旧的 downloads/index.json


SCRIPT_OK = "print(3 * 4)"


def response(status=200, url="https://example.org/data", **kwargs):
    return httpx.Response(status, request=httpx.Request("GET", url), **kwargs)


def invoke(tool, args):
    return tool.invoke({"type": "tool_call", "id": "test", "name": tool.name, "args": args})


def pdf_bytes(text="Hello PDF world"):
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'),
                             NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
    stream = DecodedStreamObject()
    stream.set_data(b'BT /F1 12 Tf 20 100 Td (' + text.encode('ascii') + b') Tj ET')
    page[NameObject('/Contents')] = stream
    writer.add_blank_page(width=200, height=200)
    data = io.BytesIO()
    writer.write(data)
    return data.getvalue()


class FakeModel:
    def __init__(self, make_call, final="证据不足，无法确认。"):
        self.make_call, self.final = make_call, final
        self.rounds, self.final_calls = 0, 0

    def bind_tools(self, registered):
        self.registered = {t.name for t in registered}
        def reply(messages):
            self.history = messages
            self.rounds += 1
            call = self.make_call(self.rounds, messages)
            return AIMessage(content="12" if call is None else "", tool_calls=[] if call is None else [call])
        return SimpleNamespace(invoke=reply)

    def invoke(self, messages):
        self.final_calls += 1
        self.final_messages = messages
        return AIMessage(content=self.final)


class ArtifactIsolation:
    """把资料库、网页表格目录和运行日志改到临时目录，测试不写项目的 downloads/ 和 rag_data/。

    不用 tempfile.TemporaryDirectory：它的 0700 目录在受限环境下无法再创建子目录。
    """

    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / f".artifacts-{uuid.uuid4().hex[:8]}"
        self.root.mkdir()
        self.downloads = self.root / "downloads"
        self.logs = self.root / "logs"
        self.rag_data = self.root / "rag_data"
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        # 资料库路径是模块常量，connect() 在 DATA_ROOT 变化时会重建 schema，所以可以整体替换；
        # downloads/ 仍然要换掉，因为网页表格的 CSV 还是 agent_tools 自己写的。
        for module, name, value in ((tools, "DOWNLOADS_DIR", self.downloads),
                                    (agent, "LOGS_DIR", self.logs),
                                    (rag_state, "DATA_ROOT", self.rag_data)):
            isolation = patch.object(module, name, value)
            isolation.start()
            self.addCleanup(isolation.stop)

    def stored(self, local_path):
        """读取结果给的路径：文件服务返回绝对路径，网页表格仍是项目内的相对路径。"""
        path = Path(local_path)
        return path if path.is_absolute() else self.root / path

    def path_of(self, kind, file_name):
        """资料库里该来源文件的落盘绝对路径（rag_data/files/<kind>/<name>）。"""
        return str(rag_state.files_dir() / kind / file_name)

    def query(self, sql, *args):
        """直接读资料库的 SQLite 行：索引 JSON 已经不存在，去重与版本都要在库里断言。"""
        conn = rag_state.connect()
        try:
            return [dict(row) for row in conn.execute(sql, args).fetchall()]
        finally:
            conn.close()

    def execute(self, sql, *args):
        """测试自己写资料库，用于构造"版本已入库""登记缺元数据"这类前置状态。"""
        conn = rag_state.connect()
        try:
            conn.execute(sql, args)
        finally:
            conn.close()

    def doc(self, source_url, kind):
        """按规范地址取唯一文档行；请求里带的 #锚点 不该分出第二份资料。"""
        rows = self.query("SELECT * FROM documents WHERE source_url=? AND kind=?",
                          rag_state.canonical_url(source_url), kind)
        self.assertEqual(len(rows), 1, f"文档登记数量不对：{source_url}")
        return rows[0]

    def version(self, doc_id, sha256):
        rows = self.query("SELECT * FROM versions WHERE doc_id=? AND sha256=?", doc_id, sha256)
        self.assertEqual(len(rows), 1, f"版本登记数量不对：{sha256[:12]}")
        return rows[0]

    def job(self, doc_id, version_id):
        conn = rag_state.connect()
        try:
            current = rag_state.current_job(conn, doc_id, version_id)
        finally:
            conn.close()
        self.assertIsNotNone(current, "读取内容时要同时确保入库任务存在")
        return current

    def ingestion_state(self, doc_id):
        conn = rag_state.connect()
        try:
            return rag_state.ingestion_state(conn, [doc_id])[doc_id]
        finally:
            conn.close()

    def counts(self):
        """(文档, 版本, 任务) 行数：断言"没有多登记一份资料/版本/任务"。"""
        return tuple(len(self.query(f"SELECT * FROM {table}"))
                     for table in ("documents", "versions", "jobs"))

    def files(self, kind=None):
        """资料目录里实际存在的文件名，替代原来对 downloads/ 目录树的检查。"""
        folder = rag_state.files_dir() / kind if kind else rag_state.files_dir()
        return sorted(p.name for p in folder.rglob("*") if p.is_file()) if folder.exists() else []

class RuntimeTests(ArtifactIsolation, unittest.TestCase):
    def test_external_research_prompt_requires_two_sources(self):
        self.assertIn("至少 2 个不同来源", agent.SYSTEM_PROMPT)
        self.assertIn("只出现在搜索摘要里不算已经核实", agent.SYSTEM_PROMPT)
        self.assertNotIn("必须是 Wikipedia", agent.SYSTEM_PROMPT)

    def test_system_prompt_routes_calculations_to_run_python(self):
        self.assertIn("run_python(code)", agent.SYSTEM_PROMPT)
        self.assertNotIn("python_calculator", agent.SYSTEM_PROMPT)
        self.assertIn("不是沙箱", agent.SYSTEM_PROMPT)
        self.assertIn("不要为了计算再次抓取相同来源", agent.SYSTEM_PROMPT)

    def test_system_prompt_injects_current_datetime(self):
        prompt = agent._system_prompt(agent.datetime(2026, 9, 24, 18, 30, 45))
        self.assertTrue(prompt.startswith("当前日期时间：2026-09-24 18:30:45\n\n"))
        self.assertIn("最近、当前、今年", prompt)
        self.assertNotIn("Asia/Shanghai", prompt)

    def run_agent(self, model, question="test"):
        with patch.object(agent, "ChatOpenAI", return_value=model), patch("builtins.print"):
            return agent.BasicAgent()(question)

    def test_failed_request_hits_network_once_and_stops(self):
        model = FakeModel(lambda n, _: {"name": "visit_webpage", "args": {"url": "https://example.org/data"}, "id": str(n)})
        with patch.object(tools.httpx, "get", return_value=response(404)) as get:
            self.assertEqual(self.run_agent(model, "Read https://example.org/data"), "证据不足，无法确认。")
        self.assertEqual(get.call_count, 1)
        self.assertEqual(model.rounds, 3)
        self.assertEqual(model.final_calls, 1)

    def test_three_failures_can_recover_by_reading_a_page(self):
        def call(n, _):
            if n <= 3:
                return {"name": "read_pdf", "args": {"url": f"https://example.org/missing{n}"}, "id": str(n)}
            if n == 4:
                return {"name": "visit_webpage", "args": {"url": "https://example.org/page"}, "id": str(n)}
            return None
        model = FakeModel(call)
        with patch.object(tools.httpx, "get", side_effect=[response(404), response(404), response(404), response(text="<html>12</html>")]):
            self.assertEqual(self.run_agent(model), "12")
        self.assertEqual(model.rounds, 5)
        self.assertEqual(model.final_calls, 0)

    def test_identical_search_results_do_not_prevent_reading_them(self):
        result = {"url": "https://example.org/chemistry", "title": "chemistry", "content": "12"}
        def call(n, _):
            if n <= 3:
                return {"name": "web_search", "args": {"query": f"chemistry exercise {n}"}, "id": str(n)}
            if n == 4:
                return {"name": "visit_webpage", "args": {"url": result["url"]}, "id": str(n)}
            return None
        model = FakeModel(call)
        with patch.object(tools, "_ddgs_search", return_value=[result]), patch.object(tools, "_get_resource", return_value=response(text="<html>12</html>")):
            self.assertEqual(self.run_agent(model), "12")
        self.assertEqual(model.final_calls, 0)

    def test_three_errors_in_one_parallel_batch_count_as_one_round(self):
        messages = [AIMessage(content="", tool_calls=[{"name": "read_pdf", "args": {}, "id": str(i)} for i in range(3)])]
        messages += [ToolMessage(content="error", tool_call_id=str(i), name="read_pdf", status="error") for i in range(3)]
        self.assertIsNone(agent._stop_reason(messages))

    def test_equivalent_default_arguments_share_a_cache_entry(self):
        first = {"name": "visit_webpage", "args": {"url": "https://example.org/Chemistry_(Exercises)"}, "id": "one"}
        second = {"name": "visit_webpage", "args": {**first["args"], "max_chars": 8000}, "id": "two"}
        self.assertEqual(agent._signature(first), agent._signature(second))

    def test_search_read_run_python_then_answer(self):
        def call(n, messages):
            calls = [("web_search", {"query": "example calculation"}),
                     ("visit_webpage", {"url": "https://example.org/data"}),
                     ("run_python", {"code": SCRIPT_OK})]
            if n > len(calls):
                self.assertEqual(json.loads(messages[-1].content)["output"].strip(), "12")
                return None
            name, args = calls[n-1]
            return {"name": name, "args": args, "id": str(n)}
        model = FakeModel(call)
        with patch.object(tools, "_ddgs_search", return_value=[{"url": "https://example.org/data", "title": "Example calculation", "content": "Calculate 3*4"}]), \
             patch.object(tools, "_get_resource", return_value=response(text='<html><article>Calculate 3*4</article></html>')):
            self.assertEqual(self.run_agent(model), "12")
        self.assertEqual(model.final_calls, 0)
        self.assertIn("read_html", model.registered)
        self.assertIn("run_python", model.registered)
        self.assertNotIn("python_calculator", model.registered)

    def test_question_state_does_not_leak(self):
        model = FakeModel(lambda n, _: None if n % 2 == 0 else
                          {"name": "run_python", "args": {"code": SCRIPT_OK}, "id": str(n)})
        with patch.object(agent, "ChatOpenAI", return_value=model), patch("builtins.print"):
            instance = agent.BasicAgent()
            self.assertEqual(instance("one"), "12")
            self.assertEqual(instance("two"), "12")
        self.assertEqual(model.history[-1].name, "run_python")

    def test_repeated_success_returns_evidence_with_new_call_id(self):
        model = FakeModel(lambda n, _: {"name": "visit_webpage", "args": {"url": "https://example.org/data"}, "id": str(n)} if n < 3 else None)
        with patch.object(tools.httpx, "get", return_value=response(text="<html><main>12</main></html>")) as get:
            self.assertEqual(self.run_agent(model), "12")
        results = [m for m in model.history if isinstance(m, ToolMessage)]
        self.assertEqual([m.tool_call_id for m in results], ["1", "2"])
        self.assertEqual([m.status for m in results], ["success", "success"])
        self.assertEqual(results[0].content, results[1].content)
        self.assertEqual(get.call_count, 1)

    def test_final_text_is_returned_verbatim(self):
        # 通用 agent 不再从模型输出里提取 <answer> 或末行，成段回答必须原样返回。
        class Chatty:
            def bind_tools(self, _):
                return SimpleNamespace(invoke=lambda _: AIMessage(content="结论：12。\n\n依据：https://example.org/data"))

        with patch.object(agent, "ChatOpenAI", return_value=Chatty()), patch("builtins.print"):
            self.assertEqual(agent.BasicAgent()("算 3*4"), "结论：12。\n\n依据：https://example.org/data")

    def test_parallel_duplicate_calls_execute_once(self):
        calls = [{"name": "visit_webpage", "args": {"url": "https://example.org"}, "id": str(n)} for n in (1, 2)]
        node = agent.ToolNode(tools.TOOLS)
        builder = agent.StateGraph(agent.AgentState)
        builder.add_node("tools", lambda state: agent._execute_tools(node, state))
        builder.add_edge(agent.START, "tools")
        builder.add_edge(agent.START if False else agent.START, "tools") if False else None
        builder.add_edge("tools", agent.END)
        with patch.object(tools.httpx, "get", return_value=response(text="<html>evidence</html>")) as get:
            result = builder.compile().invoke({"messages": [AIMessage(content="", tool_calls=calls)]})
        self.assertEqual(get.call_count, 1)
        self.assertEqual([m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)], ["1", "2"])

    def test_two_web_sources_meet_minimum_and_leave_other_tools_available(self):
        def call(n, messages):
            if n == 1:
                return {"name": "web_search", "args": {"query": "Jan Zizka"}, "id": "search"}
            if n == 2:
                return {"name": "run_python", "args": {"code": SCRIPT_OK}, "id": "calculate"}
            self.assertEqual(json.loads(messages[-1].content)["output"].strip(), "12")
            return None

        def search(_query, _max_results, domains):
            if domains == ["wikipedia.org"]:
                return [{"url": "https://en.wikipedia.org/wiki/Jan_%C5%BDi%C5%BEka",
                         "title": "Jan Zizka - Wikipedia", "content": "Wiki"}]
            return [{"url": "https://example.org/profile", "title": "Profile", "content": "Other"}]

        def fetch(url):
            return response(url=url, text=f"<html><title>Source</title><body>{url}</body></html>")

        model = FakeModel(call)
        with patch.object(tools, "_ddgs_search", side_effect=search), \
             patch.object(tools, "_get_resource", side_effect=fetch) as get:
            self.assertEqual(self.run_agent(model, "Who is Jan Zizka?"), "12")
        self.assertEqual(model.rounds, 3)
        self.assertEqual(model.final_calls, 0)
        self.assertEqual({_canonical.args[0] for _canonical in get.call_args_list},
                         {"https://en.wikipedia.org/wiki/Jan_%C5%BDi%C5%BEka", "https://example.org/profile"})

    def test_visit_after_two_web_sources_can_fetch_more_evidence(self):
        search_payload = {"results": [
            {"url": "https://en.wikipedia.org/wiki/Jan_Zizka"},
            {"url": "https://example.org/profile"},
        ]}
        messages = [
            ToolMessage(content=json.dumps(search_payload), name="web_search", tool_call_id="search"),
            ToolMessage(content=json.dumps({"url": "https://en.wikipedia.org/wiki/Jan_Zizka"}),
                        name="visit_webpage", tool_call_id="wiki"),
            ToolMessage(content=json.dumps({"url": "https://example.org/profile"}),
                        name="visit_webpage", tool_call_id="other"),
            AIMessage(content="", tool_calls=[
                {"name": "visit_webpage", "args": {"url": "https://example.org/extra"}, "id": "extra"},
                {"name": "run_python", "args": {"code": SCRIPT_OK}, "id": "calculate"},
            ]),
        ]
        builder = agent.StateGraph(agent.AgentState)
        node = agent.ToolNode(tools.TOOLS)
        builder.add_node("tools", lambda state: agent._execute_tools(node, state))
        builder.add_edge(agent.START, "tools")
        builder.add_edge("tools", agent.END)
        with patch.object(tools, "_get_resource", return_value=response(
                url="https://example.org/extra", text="<html><body>Extra evidence</body></html>")) as fetch:
            output = builder.compile().invoke({"messages": messages})
        results = {message.tool_call_id: message for message in output["messages"]
                   if isinstance(message, ToolMessage)}
        self.assertEqual(results["extra"].status, "success")
        self.assertIn("Extra evidence", results["extra"].content)
        self.assertEqual(json.loads(results["calculate"].content)["output"].strip(), "12")
        fetch.assert_called_once_with("https://example.org/extra")

    def test_failed_ordinary_source_moves_to_next_unique_url(self):
        search_payload = {"results": [
            {"url": "https://en.wikipedia.org/wiki/Jan_%C5%BDi%C5%BEka"},
            {"url": "https://bad.example/profile"},
            {"url": "https://good.example/profile"},
        ]}
        messages = [
            ToolMessage(content=json.dumps(search_payload), name="web_search", tool_call_id="search"),
            AIMessage(content="", tool_calls=[
                {"name": "visit_webpage", "args": {"url": "https://en.wikipedia.org/wiki/Jan_Žižka"}, "id": "wiki"},
                {"name": "visit_webpage", "args": {"url": "https://bad.example/profile"}, "id": "bad"},
            ]),
            ToolMessage(content=json.dumps({"url": "https://en.wikipedia.org/wiki/Jan_%C5%BDi%C5%BEka"}),
                        name="visit_webpage", tool_call_id="wiki"),
            ToolMessage(content="timeout", name="visit_webpage", tool_call_id="bad", status="error"),
        ]
        calls = agent._required_visit_calls(messages, 3)
        self.assertEqual([call["args"]["url"] for call in calls], ["https://good.example/profile"])

    def test_same_encoded_visit_url_has_one_cache_signature(self):
        encoded = {"name": "visit_webpage",
                   "args": {"url": "https://en.wikipedia.org/wiki/Jan_%C5%BDi%C5%BEka"}, "id": "one"}
        unicode_url = {"name": "visit_webpage",
                       "args": {"url": "https://en.wikipedia.org/wiki/Jan_Žižka"}, "id": "two"}
        self.assertEqual(agent._signature(encoded), agent._signature(unicode_url))

    def test_search_and_read_sources_complete_outer_research(self):
        payload = {
            "search": {"results": [
                {"url": "https://en.wikipedia.org/wiki/Attention_Is_All_You_Need"},
                {"url": "https://arxiv.org/abs/1706.03762"},
            ]},
            "attempted_urls": ["https://en.wikipedia.org/wiki/Attention_Is_All_You_Need",
                               "https://arxiv.org/abs/1706.03762"],
            "sources": [
                {"url": "https://en.wikipedia.org/wiki/Attention_Is_All_You_Need",
                 "source_type": "wikipedia"},
                {"url": "https://arxiv.org/abs/1706.03762", "source_type": "ordinary"},
            ],
            "complete": True,
        }
        messages = [ToolMessage(content=json.dumps(payload), name="search_and_read",
                                tool_call_id="combined")]
        self.assertTrue(agent._research_complete(messages))
        self.assertEqual(agent._required_visit_calls(messages, 2), [])


class ExtractionTests(ArtifactIsolation, unittest.TestCase):
    def test_search_and_read_runs_through_tool_node(self):
        data = {"results": [{"url": "https://example.org/entry", "title": "Entry", "content": "Snippet"},
                            {"url": "https://example.org/other", "title": "Other"}]}
        call = {"name": "search_and_read", "id": "combined", "args": {
            "query": "entry", "include_domains": ["example.org"], "max_chars": 5, "selector": "article"}}
        builder = agent.StateGraph(agent.AgentState)
        builder.add_node("tools", agent.ToolNode(tools.TOOLS))
        builder.add_edge(agent.START, "tools")
        builder.add_edge("tools", agent.END)
        with patch.object(tools, "_ddgs_search", return_value=data["results"]) as search, \
             patch.object(tools.httpx, "get", return_value=response(
                 text="<html><body>Menu<article>Evidence text</article></body></html>")) as get:
            output = builder.compile().invoke({"messages": [AIMessage(content="", tool_calls=[call])]})
        message = output["messages"][-1]
        page = json.loads(message.content)
        self.assertEqual(message.tool_call_id, "combined")
        self.assertEqual(message.status, "success")
        self.assertEqual(page["sources"][0]["content"], "Evide")
        self.assertEqual(page["sources"][0]["next_start_char"], 5)
        self.assertFalse(page["complete"])
        search.assert_called_once_with("entry", 5, ["example.org"])
        self.assertEqual(get.call_count, 1)
        self.assertEqual(get.call_args.args[0], "https://example.org/entry")

    def test_search_and_read_empty_results_do_not_fetch(self):
        with patch.object(tools, "_ddgs_search", return_value=[]), \
             patch.object(tools.httpx, "get") as get:
            result = invoke(tools.search_and_read, {"query": "nothing"})
        self.assertEqual(result.status, "error")
        self.assertIn("没有搜索结果", result.content)
        get.assert_not_called()

    def test_search_and_read_preserves_search_failure(self):
        with patch.object(tools, "_ddgs_search", side_effect=ValueError("DDGS_SEARCH_FAILED")), \
             patch.object(tools.httpx, "get") as get:
            result = invoke(tools.search_and_read, {"query": "entry"})
        self.assertEqual(result.status, "error")
        self.assertIn("DDGS_SEARCH_FAILED", result.content)
        get.assert_not_called()

    def test_search_and_read_preserves_reader_failure_with_url(self):
        data = {"results": [{"url": "https://example.org/report.pdf", "title": "Report"}]}
        with patch.object(tools, "_ddgs_search", return_value=data["results"]), \
             patch.object(tools.httpx, "get", return_value=response(content=b"%PDF-1.4")):
            result = invoke(tools.search_and_read, {"query": "report"})
        self.assertEqual(result.status, "error")
        self.assertIn("https://example.org/report.pdf", result.content)
        self.assertIn("read_pdf", result.content)

    def test_search_and_read_accepts_json_domain_string_and_reads_two_sources(self):
        ordinary = [{"url": "https://arxiv.org/abs/1706.03762", "title": "Attention Is All You Need"}]
        wikipedia = [{"url": "https://en.wikipedia.org/wiki/Attention_Is_All_You_Need",
                      "title": "Attention Is All You Need - Wikipedia"}]

        def fetch(url):
            return response(url=url, text=f"<html><title>Source</title><body>{url}</body></html>")

        with patch.object(tools, "_ddgs_search", side_effect=[ordinary, wikipedia]) as search, \
             patch.object(tools, "_get_resource", side_effect=fetch) as get, tools.tool_session():
            result = tools.search_and_read.invoke({
                "query": "Attention Is All You Need", "include_domains": '["arxiv.org"]'})
        self.assertTrue(result["complete"])
        self.assertEqual([item["source_type"] for item in result["sources"]], ["wikipedia", "ordinary"])
        self.assertEqual(search.call_args_list[0].args,
                         ("Attention Is All You Need", 5, ["arxiv.org"]))
        self.assertEqual(search.call_args_list[1].args,
                         ("Attention Is All You Need", 1, ["wikipedia.org"]))
        self.assertEqual(get.call_count, 2)

    def test_ddgs_contract_and_domain_filter(self):
        data = [{"href": "https://example.org/chemistry", "title": "Chemistry", "body": "Source text"}]
        searcher = MagicMock()
        searcher.text.return_value = data
        with patch.object(tools, "DDGS", return_value=searcher) as ddgs:
            result = tools.web_search.invoke({"query": "chemistry", "include_domains": ["example.org"]})
        self.assertEqual(result["provider"], "ddgs")
        self.assertEqual(result["results"][0]["url"], "https://example.org/chemistry")
        self.assertEqual(result["results"][0]["summary"], "Source text")
        ddgs.assert_called_once_with(proxy=tools._ddgs_proxy(), timeout=15)
        self.assertIn("site:example.org", searcher.text.call_args.args[0])
        self.assertEqual(searcher.text.call_args.kwargs["backend"], "auto")

    def test_ddgs_zero_results_and_failure(self):
        with patch.object(tools, "_ddgs_search", return_value=[]):
            result = tools.web_search.invoke({"query": "nothing"})
        self.assertEqual(result["results"], [])
        with patch.object(tools, "_ddgs_search", side_effect=RuntimeError("No results found")):
            self.assertEqual(invoke(tools.web_search, {"query": "nothing"}).status, "error")

    def test_only_first_parallel_web_search_adds_exact_wikipedia(self):
        calls = [{"name": "web_search", "args": {"query": query, "max_results": 2}, "id": query}
                 for query in ("first query", "second query")]

        def search(query, max_results, domains):
            if domains == ["wikipedia.org"]:
                return [{"url": "https://en.wikipedia.org/wiki/Jan_Zizka",
                         "title": "Jan Zizka - Wikipedia", "content": "Wiki"}]
            return [{"url": f"https://example.org/{query.replace(' ', '-')}",
                     "title": query, "content": "Ordinary"}]

        node = agent.ToolNode(tools.TOOLS)
        builder = agent.StateGraph(agent.AgentState)
        builder.add_node("tools", node)
        builder.add_edge(agent.START, "tools")
        builder.add_edge("tools", agent.END)
        with patch.object(tools, "_ddgs_search", side_effect=search) as mocked, tools.tool_session():
            output = builder.compile().invoke({"messages": [AIMessage(content="", tool_calls=calls)]})
        payloads = [json.loads(message.content) for message in output["messages"]
                    if isinstance(message, ToolMessage)]
        self.assertEqual(sum("first_wikipedia_search" in payload for payload in payloads), 1)
        self.assertEqual(sum(call.args[2] == ["wikipedia.org"] for call in mocked.call_args_list), 1)
        wiki_urls = [item["url"] for payload in payloads for item in payload["results"]
                     if "wikipedia" in item["url"]]
        self.assertEqual(wiki_urls, ["https://en.wikipedia.org/wiki/Jan_Zizka"])

    def test_first_wikipedia_failure_keeps_normal_search_results(self):
        ordinary = [{"url": "https://example.org/result", "title": "Result", "content": "Evidence"}]
        with patch.object(tools, "_ddgs_search",
                          side_effect=[ordinary, RuntimeError("No results found")]), tools.tool_session():
            result = tools.web_search.invoke({"query": "test"})
        self.assertEqual(result["results"][0]["url"], "https://example.org/result")
        self.assertEqual(result["first_wikipedia_search"]["status"], "error")

    def test_first_normal_search_failure_can_still_return_wikipedia(self):
        wiki = [{"url": "https://en.wikipedia.org/wiki/Jan_Zizka",
                 "title": "Jan Zizka - Wikipedia", "content": "Wiki"}]
        with patch.object(tools, "_ddgs_search",
                          side_effect=[RuntimeError("primary failed"), wiki]), tools.tool_session():
            result = tools.web_search.invoke({"query": "Jan Zizka"})
        self.assertEqual(result["results"][0]["url"], "https://en.wikipedia.org/wiki/Jan_Zizka")
        self.assertEqual(result["primary_search"]["status"], "error")
        self.assertEqual(result["first_wikipedia_search"]["status"], "success")

    def test_visit_network_error_tells_model_to_change_url(self):
        request = httpx.Request("GET", "https://blocked.example/page")
        with patch.object(tools, "_get_resource", side_effect=httpx.ConnectTimeout("timeout", request=request)):
            result = invoke(tools.visit_webpage, {"url": "https://blocked.example/page"})
        self.assertEqual(result.status, "error")
        self.assertIn("更换目标 URL", result.content)
        self.assertIn("不要再次访问这个地址", result.content)

    def test_find_page_offsets_match_reader(self):
        html = '<html><main><p>Before</p><h2>Studio albums</h2><p>Album A</p><h2>Live albums</h2></main></html>'
        with patch.object(tools.httpx, "get", return_value=response(text=html)) as get, tools.tool_session():
            found = tools.find_in_page.invoke({"url": "https://example.org", "query": "studio ALBUMS"})
            text = tools.visit_webpage.invoke({"url": "https://example.org", "start_char": found["matches"][0]["match_char"]})
        self.assertTrue(text["content"].startswith("Studio albums"))
        self.assertEqual(get.call_count, 1)

    def test_find_pdf_identifies_page_for_full_read(self):
        with patch.object(storage.httpx, "get", return_value=response(content=pdf_bytes())):
            found = storage.find_in_pdf("https://example.org", "PDF WORLD", max_matches=1)
            page = storage.read_pdf("https://example.org",
                                    start_page=found["matches"][0]["page"], max_pages=1)
        self.assertEqual(page["pages"][0]["text"], "Hello PDF world")
        self.assertEqual(found["next_start_page"], 2)

    def test_http_200_challenge_is_not_evidence(self):
        html = '<html><title>Making sure you&#39;re not a bot!</title><body>Please verify</body></html>'
        with patch.object(tools, "_get_resource",
                          return_value=response(text=html, headers={"content-type": "text/html"})):
            for selected, args in [(tools.visit_webpage, {}), (tools.read_html, {})]:
                with self.subTest(tool=selected.name):
                    result = invoke(selected, {"url": "https://example.org/file.pdf", **args})
                    self.assertEqual(result.status, "error")
                    self.assertIn("未取得正文", result.content)
        # 文件工具看的是字节：验证页既不是 PDF 也不是工作簿，同样不算证据、不进资料库
        blocked = response(text=html, url="https://example.org/file.pdf",
                           headers={"content-type": "text/html"})
        with patch.object(storage.httpx, "get", return_value=blocked):
            with self.assertRaises(ValueError) as pdf:
                storage.read_pdf("https://example.org/file.pdf")
            with self.assertRaises(ValueError) as finder:
                storage.find_in_pdf("https://example.org/file.pdf", "deposited")
            with self.assertRaises(ValueError) as workbook:
                storage.read_excel("https://example.org/file.pdf", "人机验证页不是工作簿")
        for caught in (pdf, finder):
            # 验证页要给出"换来源"的可执行提示，而不是一句"不是 PDF"
            self.assertIn("人机验证或拒绝访问", str(caught.exception))
        self.assertIn("Excel 工作簿", str(workbook.exception))
        self.assertEqual(self.counts(), (0, 0, 0))
        self.assertEqual(self.files(), [])

    def test_webpage_navigation_links_and_css_structure(self):
        html = '<html><head><base href="https://example.org/books/"></head><main><nav><a href="next">Next</a></nav><article><p class="price">42</p><a href="/file.pdf">PDF</a></article></main></html>'
        with patch.object(tools, "_get_resource", return_value=response(text=html)):
            first = tools.visit_webpage.invoke({"url": "https://example.org", "max_links": 1})
            rest = tools.visit_webpage.invoke({"url": "https://example.org", "start_link": first["next_start_link"]})
            selected = tools.visit_webpage.invoke({"url": "https://example.org", "selector": "article"})
            structure = tools.read_html.invoke({"url": "https://example.org", "selector": "article"})
            raw = tools.read_html.invoke({"url": "https://example.org", "selector": ".price", "mode": "html"})
            missing = invoke(tools.read_html, {"url": "https://example.org", "selector": "#absent"})
        self.assertEqual(first["content"], "Next\n42\nPDF")
        self.assertEqual(selected["content"], "42\nPDF")
        self.assertEqual(first["links"][0]["url"], "https://example.org/books/next")
        self.assertEqual(rest["links"][0]["url"], "https://example.org/file.pdf")
        self.assertIn('"class": ["price"]', structure["content"])
        self.assertEqual(raw["content"], '<p class="price">42</p>')
        self.assertEqual(missing.status, "error")

    def test_pdf_continues_inside_page_without_losing_text(self):
        with patch.object(storage.httpx, "get", return_value=response(content=pdf_bytes())):
            first = storage.read_pdf("https://example.org", max_chars=5)
            rest = storage.read_pdf("https://example.org", start_page=first["next_start_page"],
                                   start_char=first["next_start_char"])
        self.assertEqual(first["pages"][0]["text"] + rest["pages"][0]["text"], "Hello PDF world")
        self.assertEqual(rest["pages"][1]["page"], 2)
        self.assertTrue(rest["needs_ocr"])
        self.assertFalse(rest["truncated"])

    def test_pdf_download_endpoint_returns_404_with_pdf_body(self):
        """Logitech 的 dam 路径会给 PDF 响应回 404；判断依据是响应体，不是状态码。"""
        payload = pdf_bytes()
        response_404 = httpx.Response(404, request=httpx.Request("GET", "https://example.org/brio.pdf"),
                                       content=payload,
                                       headers={"content-disposition": "attachment; filename=brio.pdf"})
        with patch.object(storage.httpx, "get", return_value=response_404):
            result = storage.read_pdf("https://example.org/brio.pdf")
        self.assertEqual(result["pages"][0]["text"], "Hello PDF world")
        self.assertTrue(result["artifact_saved"])
        self.assertTrue(result["ingestion"]["job_id"])      # 404 但字节可用，照样提交入库任务
        self.assertEqual(self.counts(), (1, 1, 1))
        # 不是 PDF 的 404 仍然是错误
        response_404_html = httpx.Response(404, request=httpx.Request("GET", "https://example.org/missing.pdf"),
                                           text="<html>not found</html>", headers={"content-type": "text/html"})
        with patch.object(storage.httpx, "get", return_value=response_404_html):
            with self.assertRaises(httpx.HTTPStatusError):
                storage.read_pdf("https://example.org/missing.pdf")
        self.assertEqual(self.counts(), (1, 1, 1))

    def test_excel_dates_text_and_both_pagination_axes(self):
        data = io.BytesIO()
        with pd.ExcelWriter(data, engine="openpyxl") as writer:
            pd.DataFrame([["NA", "编号", "日期"], ["001", 2, datetime(2025, 1, 1)], ["尾行", 3, 4]]).to_excel(writer, sheet_name="数据", header=False, index=False)
            pd.DataFrame([[9]]).to_excel(writer, sheet_name="第二张", header=False, index=False)
        summary = "示例工作簿：编号与日期，另有第二张"
        with patch.object(storage.httpx, "get", return_value=response(content=data.getvalue())):
            preview = storage.read_excel("https://example.org", summary)
            first = storage.read_excel("https://example.org", summary, max_rows=2, max_columns=2,
                                       preview=False)
            right = storage.read_excel("https://example.org", summary, start_column=2, preview=False)
            second = storage.read_excel("https://example.org", summary, sheet_name="第二张", preview=False)
        self.assertEqual(preview["columns"], ["NA", "编号", "日期"])
        self.assertEqual(preview["total_rows"], 3)
        self.assertEqual(preview["sample_rows"], [["NA", "编号", "日期"], ["001", 2, "2025-01-01T00:00:00"],
                                                  ["尾行", 3, 4]])
        self.assertEqual(first["rows"], [["NA", "编号"], ["001", 2]])
        self.assertEqual((first["next_start_row"], first["next_start_column"]), (2, 2))
        self.assertIn("2025-01-01", right["rows"][1][0])
        self.assertEqual(second["rows"], [[9]])
        # 工作表名写错照样报错，且不因为换工作表就重复登记版本
        with patch.object(storage.httpx, "get", return_value=response(content=data.getvalue())):
            with self.assertRaises(ValueError):
                storage.read_excel("https://example.org", summary, sheet_name="没有这张表")
        self.assertEqual(self.counts(), (1, 1, 1))
        self.assertEqual({preview["ingestion"]["job_id"], second["ingestion"]["job_id"]},
                         {preview["ingestion"]["job_id"]})

    def test_run_python_rejects_blank_code(self):
        result = tools.run_python.invoke({"code": "   "})
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "bad_request")

    def test_run_python_returns_real_output(self):
        result = tools.run_python.invoke({"code": SCRIPT_OK})
        self.assertTrue(result["success"])
        self.assertEqual(result["output"].strip(), "12")
        self.assertEqual(invoke(tools.read_pdf, {}).status, "error")
    def test_excel_keeps_numeric_looking_text(self):
        data = io.BytesIO()
        pd.DataFrame([["001"], ["002"]]).to_excel(data, header=False, index=False)
        with patch.object(storage.httpx, "get", return_value=response(content=data.getvalue())):
            result = storage.read_excel("https://example.org", "编号列表", preview=False)
        self.assertEqual(result["rows"], [["001"], ["002"]])

    def test_text_utf16_and_binary_rejection(self):
        with patch.object(tools, "_get_resource", return_value=response(content='你好\n42'.encode('utf-16'))):
            self.assertEqual(tools.read_text_file.invoke({"url": "https://example.org"})["content"], '你好\n42')
        with patch.object(tools, "_get_resource", return_value=response(content=b'%PDF-1.4')):
            self.assertEqual(invoke(tools.read_text_file, {"url": "https://example.org"}).status, "error")

    def test_real_http_file_parsers(self):
        excel = io.BytesIO()
        pd.DataFrame([["A", "B"], [2, 3]]).to_excel(excel, header=False, index=False)
        assets = {"/page": ("text/html", b'<html><title>Local</title><article>3 * 4</article></html>'),
                  "/file.pdf": ("application/pdf", pdf_bytes()), "/file.xlsx": ("application/octet-stream", excel.getvalue())}
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                kind, data = assets[self.path]
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            with patch.dict("os.environ", {"NO_PROXY": "127.0.0.1"}):
                self.assertEqual(tools.visit_webpage.invoke({"url": url + "/page"})["title"], "Local")
                self.assertEqual(storage.read_pdf(url + "/file.pdf")["pages"][0]["text"], "Hello PDF world")
                self.assertEqual(storage.read_excel(url + "/file.xlsx", "A、B 两列示例表", preview=False)["rows"],
                                 [["A", "B"], [2, 3]])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class ArtifactTests(ArtifactIsolation, unittest.TestCase):
    """只覆盖下载落盘：成功才保存、按 URL 去重、失败不进资料库。"""

    @staticmethod
    def excel_bytes(*sheets):
        data = io.BytesIO()
        with pd.ExcelWriter(data, engine="openpyxl") as writer:
            for name, rows in sheets:
                pd.DataFrame(rows).to_excel(writer, sheet_name=name, header=False, index=False)
        return data.getvalue()

    def test_pdf_is_saved_byte_identical_and_indexed(self):
        payload = pdf_bytes()
        url = "https://example.org/files/report.pdf"
        with patch.object(storage.httpx, "get",
                          return_value=response(content=payload, url=url)):
            # 带 #锚点 的地址与规范地址是同一份资料：库里不该分出两个文档
            result = storage.read_pdf(url + "#page=2")
        digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(result["local_path"], self.path_of("pdf", "report.pdf"))
        self.assertTrue(result["artifact_saved"])
        self.assertNotIn("artifact_error", result)
        self.assertEqual(result["page_count"], 2)
        self.assertEqual(result["pages"][0]["text"], "Hello PDF world")
        self.assertEqual(self.stored(result["local_path"]).read_bytes(), payload)   # 字节原样落盘
        self.assertEqual(result["version"], digest)                                 # 版本就是字节的摘要
        self.assertEqual(result["file_name"], "report.pdf")
        self.assertTrue(result["read_ok"])
        self.assertTrue(result["ingestion"]["job_id"])      # 读到内容就提交了入库任务
        self.assertEqual(result["ingestion"]["status"], "queued")
        self.assertEqual((result["ingestion"]["attempts"], result["ingestion"]["round"]), (0, 1))
        self.assertIsNone(result["ingestion"]["last_error"])
        doc = self.doc(url, "pdf")
        self.assertEqual(doc["source_url"], url)
        self.assertEqual(doc["kind"], "pdf")
        self.assertEqual(doc["file_name"], "report.pdf")
        version = self.version(doc["doc_id"], digest)
        self.assertEqual(version["kind"], "pdf")
        self.assertEqual(version["size"], len(payload))
        self.assertEqual(version["local_path"], result["local_path"])
        self.assertEqual(version["final_url"], url)
        self.assertEqual(version["page_count"], 2)
        self.assertEqual(version["status"], "stored")      # 服务进程没跑，先落在待处理
        self.assertTrue(version["created_at"])
        self.assertEqual(self.job(doc["doc_id"], version["id"])["job_id"], result["ingestion"]["job_id"])

    def test_pdf_pagination_and_search_share_one_copy(self):
        payload = pdf_bytes()
        url = "https://example.org/files/report.pdf"
        fetched = response(content=payload, url=url)
        with patch.object(storage.httpx, "get", return_value=fetched) as get:
            first = storage.read_pdf(url, max_chars=5)
            rest = storage.read_pdf(url, start_page=first["next_start_page"],
                                    start_char=first["next_start_char"])
            found = storage.find_in_pdf(url, "PDF WORLD")
        self.assertEqual(first["pages"][0]["text"] + rest["pages"][0]["text"], "Hello PDF world")
        self.assertEqual(first["local_path"], found["local_path"])
        self.assertEqual(found["empty_pages"], [2])           # 第二页没有文字，标记要 OCR
        self.assertTrue(found["needs_ocr"])
        self.assertEqual(self.files("pdf"), ["report.pdf"])
        self.assertEqual(self.counts(), (1, 1, 1))            # 一份资料、一个版本、一个任务
        self.assertEqual({first["ingestion"]["job_id"], rest["ingestion"]["job_id"],
                          found["ingestion"]["job_id"]}, {first["ingestion"]["job_id"]})
        get.assert_called_once()                              # 续页与检索读的都是本地副本

    def test_second_task_reuses_the_indexed_file(self):
        payload = pdf_bytes()
        url = "https://example.org/files/report.pdf"
        jobs = []
        for _ in range(2):  # 两次独立任务，各拿一份新的响应对象
            with patch.object(storage.httpx, "get", return_value=response(content=payload, url=url)):
                result = storage.read_pdf(url)
            jobs.append(result["ingestion"]["job_id"])
        self.assertEqual(result["local_path"], self.path_of("pdf", "report.pdf"))
        self.assertEqual(self.files("pdf"), ["report.pdf"])
        self.assertEqual(self.counts(), (1, 1, 1))
        self.assertEqual(set(jobs), {jobs[0]})                # 同一版本复用同一个任务，不重复排队
        self.assertTrue(result["from_cache"])

    def test_updated_content_is_not_refetched_while_the_local_copy_is_intact(self):
        """本地副本和登记摘要一致就复用它：不在每次读取时都回源重下（服务器换了内容也要先有下载动作）。"""
        url = "https://example.org/files/report.pdf"
        stale, fresh = pdf_bytes(), pdf_bytes("Newer PDF value")
        with patch.object(storage.httpx, "get", return_value=response(content=stale, url=url)):
            first = storage.read_pdf(url)
        with patch.object(storage.httpx, "get", return_value=response(content=fresh, url=url)) as fetch:
            second = storage.read_pdf(url)
        fetch.assert_not_called()
        self.assertEqual(second["local_path"], first["local_path"])
        self.assertEqual(second["pages"][0]["text"], "Hello PDF world")   # 本次读到的就是本地那份
        self.assertEqual(self.files("pdf"), ["report.pdf"])
        doc = self.doc(url, "pdf")
        stale_row = self.version(doc["doc_id"], hashlib.sha256(stale).hexdigest())
        self.assertEqual(stale_row["size"], len(stale))
        self.assertEqual(self.counts(), (1, 1, 1))
        # 旧版本已经入库时，新字节的登记不能把它洗回待处理
        self.execute("UPDATE versions SET status='indexed' WHERE id=?", stale_row["id"])
        # 本地副本被破坏后才会重新下载，并把资料库刷成新版本
        self.stored(first["local_path"]).write_bytes(b"%PDF-1.4 damaged")
        with patch.object(storage, "_atomic_write", wraps=storage._atomic_write) as writer, \
             patch.object(storage.httpx, "get", return_value=response(content=fresh, url=url)):
            third = storage.read_pdf(url)
        self.assertEqual(third["pages"][0]["text"], "Newer PDF value")
        self.assertEqual(writer.call_count, 1)
        self.assertEqual(third["version"], hashlib.sha256(fresh).hexdigest())   # 版本就是本次读到的字节
        fresh_row = self.version(doc["doc_id"], hashlib.sha256(fresh).hexdigest())
        self.assertEqual(self.stored(fresh_row["local_path"]).read_bytes(), fresh)
        self.assertEqual(fresh_row["status"], "stored")
        self.assertGreater(fresh_row["id"], stale_row["id"])                   # 新字节是新的版本行
        self.assertEqual(self.version(doc["doc_id"], stale_row["sha256"])["status"], "indexed")
        state = self.ingestion_state(doc["doc_id"])
        self.assertEqual(state["latest_id"], fresh_row["id"])                   # 检索侧看得到的是最新版
        self.assertEqual(self.files("pdf"), ["report-1.pdf", "report.pdf"])    # 旧版本文件不被顶掉
        self.assertEqual(self.counts()[1:], (2, 2))

    def test_identical_content_is_reused_without_rewriting(self):
        url = "https://example.org/files/report.pdf"
        payload = pdf_bytes()
        with patch.object(storage, "_atomic_write", wraps=storage._atomic_write) as writer, \
             patch.object(storage.httpx, "get", return_value=response(content=payload, url=url)) as get:
            first = storage.read_pdf(url)
            second = storage.read_pdf(url)
        self.assertEqual(writer.call_count, 1)
        self.assertEqual(get.call_count, 1)                              # 第二次连下载都不需要
        self.assertEqual(second["local_path"], first["local_path"])
        self.assertEqual(self.stored(first["local_path"]).read_bytes(), payload)
        self.assertEqual(second["version"], first["version"])
        self.assertEqual(second["ingestion"]["job_id"], first["ingestion"]["job_id"])
        self.assertEqual(self.counts(), (1, 1, 1))

    def test_existing_version_is_backfilled_without_rewriting(self):
        """文件与登记都在、只缺统计元数据：读一次确认字节一致后补登记，文件和时间都不重写。

        旧索引允许条目没有 sha256；现在的 schema 里摘要非空，缺的换成页数这类信息。
        """
        url = "https://example.org/files/report.pdf"
        payload = pdf_bytes()
        folder = rag_state.files_dir() / "pdf"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "report.pdf"
        target.write_bytes(payload)
        conn = rag_state.connect()
        try:
            doc = rag_state.get_or_create_document(conn, url, "pdf", "report.pdf")
            version, created = rag_state.add_version(
                conn, doc_id=doc["doc_id"], sha256=rag_state.content_digest(payload), kind="pdf",
                local_path=str(target), size=len(payload), file_name="report.pdf", final_url=url,
                summary=None, page_count=None, empty_pages=None, sheet_names=None)
        finally:
            conn.close()
        self.assertTrue(created)
        stamp = "2026-01-01T00:00:00+08:00"
        self.execute("UPDATE versions SET created_at=?, page_count=NULL WHERE id=?", stamp, version["id"])
        with patch.object(storage, "_atomic_write", wraps=storage._atomic_write) as writer, \
             patch.object(storage.httpx, "get", return_value=response(content=payload, url=url)):
            result = storage.read_pdf(url)
        self.assertEqual(writer.call_count, 0)
        self.assertTrue(result["artifact_saved"])
        self.assertEqual(result["local_path"], str(target))
        row = self.version(doc["doc_id"], result["version"])
        self.assertEqual(row["id"], version["id"])               # 复用同一行登记，没有新增版本
        self.assertEqual(row["created_at"], stamp)             # 登记时间仍是第一次下载的时间
        self.assertEqual(row["page_count"], 2)                 # 缺的元数据这次补上了
        self.assertEqual(row["status"], "stored")
        self.assertEqual(self.counts(), (1, 1, 1))

    def test_updated_excel_is_picked_up_when_the_local_copy_is_gone(self):
        """本地副本还在就复用；本地没了才重新下载，并把资料库刷成新版本。"""
        url = "https://example.org/files/books.xlsx"
        stale = self.excel_bytes(("数据", [["A", 1]]))
        fresh = self.excel_bytes(("数据", [["A", 1]]), ("第二张", [[2]]))
        summary = "图书清单：数据表按名称与数量记录"
        with patch.object(storage.httpx, "get", return_value=response(content=stale, url=url)):
            first = storage.read_excel(url, summary)
        self.assertEqual(first["sheet_names"], ["数据"])
        self.stored(first["local_path"]).unlink()          # 本地副本消失，必须回源
        with patch.object(storage.httpx, "get", return_value=response(content=fresh, url=url)) as fetch:
            second = storage.read_excel(url, summary, sheet_name="第二张", preview=False)
        fetch.assert_called_once()
        self.assertEqual(second["sheet_names"], ["数据", "第二张"])
        self.assertEqual(second["rows"], [[2]])
        self.assertEqual(second["local_path"], first["local_path"])
        self.assertEqual(second["version"], hashlib.sha256(fresh).hexdigest())
        self.assertEqual(self.stored(second["local_path"]).read_bytes(), fresh)
        self.assertEqual(self.files("excel"), ["books.xlsx"])
        doc = self.doc(url, "excel")
        row = self.version(doc["doc_id"], hashlib.sha256(fresh).hexdigest())
        self.assertEqual(row["size"], len(fresh))
        self.assertEqual(row["local_path"], second["local_path"])
        self.assertEqual(row["summary"], summary)
        self.assertEqual(json.loads(row["sheet_names"]), ["数据", "第二张"])
        self.assertEqual(self.counts()[1:], (2, 2))        # 旧版本行保留，新版本另起一行与一个任务

    def test_challenge_page_and_failed_download_are_not_saved(self):
        html = '<html><title>Making sure you&#39;re not a bot!</title><body>Please verify</body></html>'
        with patch.object(storage.httpx, "get", return_value=response(
                text=html, url="https://example.org/file.pdf", headers={"content-type": "text/html"})):
            with self.assertRaises(ValueError):
                storage.read_pdf("https://example.org/file.pdf")
        missing = response(404, url="https://example.org/missing.pdf")
        with patch.object(storage.httpx, "get", return_value=missing):
            with self.assertRaises(httpx.HTTPStatusError):
                storage.read_pdf("https://example.org/missing.pdf")
        self.assertEqual(self.counts(), (0, 0, 0))       # 失败响应连资料壳都不留
        self.assertEqual(self.files(), [])
        self.assertFalse(self.downloads.exists())        # 文件服务不写旧的 downloads/

    def test_broken_files_are_not_saved(self):
        broken_pdf = response(content=b"%PDF-1.4 only a header", url="https://example.org/bad.pdf")
        with patch.object(storage.httpx, "get", return_value=broken_pdf):
            with self.assertRaises(PdfReadError):         # 声明是 PDF 但流被截断
                storage.read_pdf("https://example.org/bad.pdf")
        broken_xlsx = response(content=b"not a workbook at all", url="https://example.org/bad.xlsx")
        with patch.object(storage.httpx, "get", return_value=broken_xlsx):
            with self.assertRaises(ValueError) as excel:
                storage.read_excel("https://example.org/bad.xlsx", "无法解析的工作簿")
        self.assertIn("Excel 工作簿", str(excel.exception))
        self.assertEqual(self.counts(), (0, 0, 0))
        self.assertEqual(self.files(), [])
        self.assertFalse(self.downloads.exists())

    def test_excel_is_saved_once_across_sheets(self):
        payload = self.excel_bytes(("数据", [["name", "qty"], ["brio", 12]]), ("第二张", [[9]]))
        url = "https://example.org/files/books.xlsx"
        fetched = response(content=payload, url=url)
        with patch.object(storage.httpx, "get", return_value=fetched):
            first = storage.read_excel(url, "图书清单：brio 的销量记录")
            second = storage.read_excel(url, "换了措辞的同一份工作簿", sheet_name="第二张", preview=False)
        self.assertEqual(first["local_path"], self.path_of("excel", "books.xlsx"))
        self.assertEqual(second["local_path"], first["local_path"])
        self.assertEqual(first["columns"], ["name", "qty"])
        self.assertEqual(first["sample_rows"], [["name", "qty"], ["brio", 12]])
        self.assertEqual(second["rows"], [[9]])
        self.assertEqual(self.stored(first["local_path"]).read_bytes(), payload)
        self.assertEqual(self.files("excel"), ["books.xlsx"])
        doc = self.doc(url, "excel")
        self.assertEqual(doc["kind"], "excel")
        # 检索描述只用第一次接受的 summary：换措辞不该重建索引，也不该新增版本
        self.assertEqual(first["summary"], "图书清单：brio 的销量记录")
        self.assertFalse(first["summary_kept"])
        self.assertTrue(second["summary_kept"])
        self.assertEqual(second["summary"], first["summary"])
        self.version(doc["doc_id"], first["version"])
        self.assertEqual(self.version(doc["doc_id"], first["version"])["summary"],
                         "图书清单：brio 的销量记录")
        self.assertEqual(self.counts(), (1, 1, 1))
        self.assertEqual(first["ingestion"]["job_id"], second["ingestion"]["job_id"])

    def test_excel_requires_a_non_blank_summary(self):
        """summary 是这个版本在长期记忆里的检索描述：空白没法描述主题，直接拒绝且不下载。"""
        payload = self.excel_bytes(("Sheet1", [[1]]))
        url = "https://example.org/b.xlsx"
        for blank in ("", "   ", "\n\t", None):
            with self.subTest(summary=repr(blank)):
                with patch.object(storage.httpx, "get",
                                  return_value=response(content=payload, url=url)) as get:
                    with self.assertRaises(ValueError) as caught:
                        storage.read_excel(url, blank)
                self.assertIn("summary 不能为空", str(caught.exception))
                get.assert_not_called()
        self.assertEqual(self.counts(), (0, 0, 0))

    def test_excel_keeps_xls_and_xlsx_names_from_header(self):
        payload = self.excel_bytes(("Sheet1", [["A", 1]]))
        # 非 ASCII 文件名按 RFC 5987 用 filename* 传输，HTTP 头本身只允许 ASCII 字节。
        cases = [('attachment; filename="stock-2026.xls"', "https://example.org/one", "stock-2026.xls"),
                 ("attachment; filename*=UTF-8''%E5%BA%93%E5%AD%98.xls", "https://example.org/two", "库存.xls"),
                 ("attachment; filename=Q3-2026.xlsx", "https://example.org/three", "Q3-2026.xlsx")]
        for header, url, expected in cases:
            with self.subTest(header=header):
                with patch.object(storage.httpx, "get", return_value=response(
                        content=payload, url=url, headers={"content-disposition": header})):
                    result = storage.read_excel(url, "表头给出文件名的工作簿")
                self.assertEqual(result["local_path"], self.path_of("excel", expected))
                self.assertEqual(result["file_name"], expected)
                self.assertTrue(self.stored(result["local_path"]).is_file())
                self.assertEqual(self.doc(url, "excel")["file_name"], expected)

    def test_download_name_priority_and_sanitizing(self):
        """落盘文件名取自 Content-Disposition → 最终 URL → 默认名，并做 Windows 非法字符清洗。"""
        def name_of(resp, kind):
            return storage.download_name(resp.headers, str(resp.url), kind)

        encoded = response(content=b"", url="https://example.org/ignored.pdf", headers={
            "content-disposition": "attachment; filename*=UTF-8''%E4%B8%AD%E6%96%87%20%E6%8A%A5%E5%91%8A.pdf"})
        self.assertEqual(name_of(encoded, "pdf"), "中文 报告.pdf")
        messy = response(content=b"", url="https://example.org/x.pdf",
                         headers={"content-disposition": 'attachment; filename="a<b>:c?.pdf"'})
        self.assertEqual(name_of(messy, "pdf"), "a_b__c_.pdf")
        from_url = response(content=b"", url="https://example.org/files/books.xlsx?token=1")
        self.assertEqual(name_of(from_url, "excel"), "books.xlsx")
        unnamed = response(content=b"", url="https://example.org/")
        self.assertEqual(name_of(unnamed, "pdf"), "document.pdf")
        self.assertEqual(name_of(unnamed, "excel"), "workbook.xlsx")
        wrong_suffix = response(content=b"", url="https://example.org/download.bin")
        self.assertEqual(name_of(wrong_suffix, "excel"), "download.xlsx")
        reserved = response(content=b"", url="https://example.org/CON.pdf")
        self.assertEqual(name_of(reserved, "pdf"), "_CON.pdf")

    def test_indexed_file_is_reused_without_touching_the_network(self):
        """资料库里已经有同一个 URL 的文件时，直接读本地：下载地址抖动不该断掉交接链。"""
        url = "https://example.org/files/books.xlsx"
        payload = self.excel_bytes(("数据", [["name", "qty"], ["brio", 12]]))
        summary = "图书清单：brio 的销量"
        with patch.object(storage.httpx, "get", return_value=response(content=payload, url=url)):
            first = storage.read_excel(url, summary)
        with patch.object(storage.httpx, "get",
                          side_effect=httpx.ConnectError("network down")) as fetch:
            second = storage.read_excel(url, summary)
        fetch.assert_not_called()
        self.assertEqual(second["local_path"], first["local_path"])
        self.assertEqual(second["sample_rows"], [["name", "qty"], ["brio", 12]])
        # 命中本地副本要如实说明"本次没有回源核验"，不能假装刚确认过远端
        self.assertTrue(second["from_cache"])
        self.assertFalse(second["remote_checked"])
        self.assertEqual(second["version_note"], "本地副本，本次未重新核验远端")
        doc = self.doc(url, "excel")
        self.assertEqual(second["verified_at"], self.version(doc["doc_id"], first["version"])["created_at"])
        self.assertFalse(first["from_cache"])
        self.assertTrue(first["remote_checked"])
        self.assertIsNone(first["verified_at"])
        self.assertEqual(first["version_note"], "本次已重新核验远端内容")

    def test_indexed_pdf_is_reused_without_touching_the_network(self):
        url = "https://example.org/files/report.pdf"
        payload = pdf_bytes()
        with patch.object(storage.httpx, "get", return_value=response(content=payload, url=url)):
            first = storage.read_pdf(url)
        with patch.object(storage.httpx, "get",
                          side_effect=httpx.ConnectError("network down")) as fetch:
            second = storage.read_pdf(url)
        fetch.assert_not_called()
        self.assertEqual(second["local_path"], first["local_path"])
        self.assertEqual(second["pages"][0]["text"], "Hello PDF world")
        self.assertTrue(second["from_cache"])
        self.assertEqual(self.counts(), (1, 1, 1))
        # refresh=True 才是显式回源核验；字节没变时既不新增版本也不重复排队
        with patch.object(storage.httpx, "get",
                          return_value=response(content=payload, url=url)) as get:
            refreshed = storage.read_pdf(url, refresh=True)
        get.assert_called_once()
        self.assertFalse(refreshed["from_cache"])
        self.assertTrue(refreshed["remote_checked"])
        self.assertEqual(refreshed["local_path"], first["local_path"])
        self.assertEqual(refreshed["ingestion"]["job_id"], first["ingestion"]["job_id"])
        self.assertEqual(self.counts(), (1, 1, 1))

    def test_corrupted_local_copy_falls_back_to_the_network(self):
        url = "https://example.org/files/books.xlsx"
        payload = self.excel_bytes(("数据", [["A", 1]]))
        summary = "损坏恢复用的工作簿"
        with patch.object(storage.httpx, "get", return_value=response(content=payload, url=url)):
            first = storage.read_excel(url, summary)
        self.stored(first["local_path"]).write_bytes(b"corrupted on disk")
        with patch.object(storage.httpx, "get",
                          return_value=response(content=payload, url=url)) as fetch:
            second = storage.read_excel(url, summary)
        fetch.assert_called_once()          # 本地不可信，回到网络重新取
        self.assertEqual(second["sample_rows"], [["A", 1]])
        self.assertFalse(second["from_cache"])
        self.assertEqual(self.stored(first["local_path"]).read_bytes(), payload)   # 原路径重写回正确字节
        self.assertEqual(second["ingestion"]["job_id"], first["ingestion"]["job_id"])
        self.assertEqual(self.counts(), (1, 1, 1))

    def test_other_tools_never_create_artifacts(self):
        html = "<html><body><main>text</main><table><tr><td>1</td></tr></table></body></html>"
        with patch.object(tools, "_get_resource", return_value=response(text=html)), \
             patch.object(tools, "_ddgs_search",
                          return_value=[{"url": "https://example.org/page", "title": "t", "content": "c"}]):
            tools.web_search.invoke({"query": "q"})
            tools.visit_webpage.invoke({"url": "https://example.org/page"})
            tools.read_html.invoke({"url": "https://example.org/page"})
            tools.read_text_file.invoke({"url": "https://example.org/page"})
        self.assertFalse(self.downloads.exists())
        self.assertEqual(self.counts(), (0, 0, 0))    # 资料库里没有文档/版本/任务
        self.assertEqual(self.files(), [])

    def test_webpage_tables_are_saved_outside_the_artifact_index(self):
        """网页表格存成 CSV 供 run_python 使用，但不算 PDF/Excel 资料，不进资料库登记。"""
        html = ("<html><body><table><tr><th>name</th><th>qty</th></tr>"
                "<tr><td>brio</td><td>12</td></tr><tr><td>veho</td><td>30</td></tr></table></body></html>")
        with patch.object(tools, "_get_resource", return_value=response(text=html)):
            first = tools.read_webpage_tables.invoke({"url": "https://example.org/page"})
            second = tools.read_webpage_tables.invoke({"url": "https://example.org/page"})
        table = first["tables"][0]
        self.assertEqual(table["columns"], ["name", "qty"])
        self.assertEqual(table["total_rows"], 2)
        self.assertEqual(table["sample_rows"], [["brio", 12], ["veho", 30]])
        self.assertTrue(table["artifact_saved"])
        self.assertEqual(Path(table["local_path"]).parent.as_posix(), "downloads/tables")
        self.assertRegex(Path(table["local_path"]).name, r"^example\.org-[0-9a-f]{8}-table0\.csv$")
        self.assertEqual(second["tables"][0]["local_path"], table["local_path"])
        self.assertEqual([p.name for p in (self.downloads / "tables").iterdir()],
                         [Path(table["local_path"]).name])
        self.assertEqual(self.counts(), (0, 0, 0))        # 没有文档/版本/任务登记
        self.assertEqual(self.files(), [])

    def test_same_host_different_pages_do_not_overwrite_each_other(self):
        """同站不同页面的 table0 各自落盘：后读的页面不能覆盖先读页面的 CSV。"""
        first_html = "<html><body><table><tr><th>a</th></tr><tr><td>1</td></tr></table></body></html>"
        second_html = "<html><body><table><tr><th>b</th></tr><tr><td>2</td></tr></table></body></html>"
        # 传入的响应要带真实的请求 URL：落盘文件名取的就是这个地址。
        with patch.object(tools, "_get_resource",
                          return_value=response(text=first_html, url="https://example.org/one")):
            first = tools.read_webpage_tables.invoke({"url": "https://example.org/one"})
        with patch.object(tools, "_get_resource",
                          return_value=response(text=second_html, url="https://example.org/two")):
            second = tools.read_webpage_tables.invoke({"url": "https://example.org/two"})
        first_path, second_path = first["tables"][0]["local_path"], second["tables"][0]["local_path"]
        self.assertNotEqual(first_path, second_path)
        self.assertEqual([p.name for p in (self.downloads / "tables").iterdir()],
                         sorted([Path(first_path).name, Path(second_path).name]))
        self.assertEqual(self.stored(first_path).read_text(encoding="utf-8-sig"), "a\n1\n")
        self.assertEqual(self.stored(second_path).read_text(encoding="utf-8-sig"), "b\n2\n")

    def test_state_is_the_rag_entry_point(self):
        """资料库状态（文档/版本/任务）就是检索侧入口：换掉旧的 list_artifacts 目录列表。"""
        pdf_url, excel_url = "https://example.org/a.pdf", "https://example.org/b.xlsx"
        with patch.object(storage.httpx, "get",
                          return_value=response(content=pdf_bytes(), url=pdf_url)):
            pdf = storage.read_pdf(pdf_url)
        with patch.object(storage.httpx, "get",
                          return_value=response(content=self.excel_bytes(("Sheet1", [[1]])), url=excel_url)):
            excel = storage.read_excel(excel_url, "只有一列的示例工作簿")
        self.assertEqual(self.counts(), (2, 2, 2))
        for kind, result in (("excel", excel), ("pdf", pdf)):
            doc = self.doc(result["source_url"], kind)
            version = self.version(doc["doc_id"], result["version"])
            self.assertEqual(version["kind"], kind)
            self.assertTrue(Path(version["local_path"]).is_file())
            self.assertEqual(version["local_path"],
                             str(rag_state.files_dir() / kind / version["file_name"]))
            self.assertEqual(version["sha256"],
                             hashlib.sha256(Path(version["local_path"]).read_bytes()).hexdigest())
            job = self.job(doc["doc_id"], version["id"])
            self.assertEqual(job["job_id"], result["ingestion"]["job_id"])
            self.assertEqual(job["status"], "queued")            # 任务已排队，等服务进程执行
            self.assertEqual((job["attempts"], job["round"]), (0, 1))
            self.assertIsNone(job["last_error"])
            self.assertEqual(self.ingestion_state(doc["doc_id"])["latest_id"], version["id"])
        # 后台任务还没跑：没有任何"有效版本"可被检索引用
        conn = rag_state.connect()
        try:
            self.assertEqual(rag_state.all_active_versions(conn), [])
        finally:
            conn.close()

    def test_corrupted_local_copy_is_rewritten(self):
        """本地副本被改动后，不能凭登记里的旧摘要就当作可以复用。"""
        url = "https://example.org/files/report.pdf"
        payload = pdf_bytes()
        with patch.object(storage.httpx, "get", return_value=response(content=payload, url=url)):
            first = storage.read_pdf(url)
        target = self.stored(first["local_path"])
        target.write_bytes(b"%PDF-1.4 damaged on disk")
        with patch.object(storage, "_atomic_write", wraps=storage._atomic_write) as writer, \
             patch.object(storage.httpx, "get", return_value=response(content=payload, url=url)):
            second = storage.read_pdf(url)
        self.assertEqual(writer.call_count, 1)  # URL 内容没变，但本地被破坏，重新写回正确字节
        self.assertEqual(target.read_bytes(), payload)
        self.assertEqual(second["local_path"], first["local_path"])
        digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(second["version"], digest)
        doc = self.doc(url, "pdf")
        version = self.version(doc["doc_id"], digest)
        self.assertEqual(version["page_count"], 2)
        self.assertEqual(version["size"], len(payload))
        self.assertEqual(version["local_path"], second["local_path"])
        self.assertEqual(self.counts(), (1, 1, 1))               # 重写不新增版本，也不重复排队
        self.assertEqual(second["ingestion"]["job_id"], first["ingestion"]["job_id"])

    def test_index_failure_restores_the_previous_file(self):
        """登记写失败时回滚旧文件：不能留下"文件新版、登记旧版"的组合。"""
        url = "https://example.org/files/report.pdf"
        payload = pdf_bytes()
        with patch.object(storage.httpx, "get", return_value=response(content=payload, url=url)):
            first = storage.read_pdf(url)
        self.doc(url, "pdf")                             # 先确认库里只有一份文档
        registered = self.query("SELECT * FROM versions")
        jobs = self.query("SELECT * FROM jobs")
        # 覆盖前先把当时的本地文件暂存起来，登记失败就原样放回（损坏前的字节也不会被新内容顶掉）
        self.stored(first["local_path"]).write_bytes(b"%PDF-1.4 damaged on disk")
        with patch.object(storage.httpx, "get", return_value=response(content=payload, url=url)), \
             patch.object(rag_state, "add_version", side_effect=OSError("index locked")):
            second = storage.read_pdf(url)
        self.assertEqual(second["pages"][0]["text"], "Hello PDF world")   # 本次读取仍给出内容
        self.assertFalse(second["artifact_saved"])
        self.assertIn("index locked", second["artifact_error"])
        self.assertEqual(second["ingestion"]["status"], "submit_failed")
        self.assertIsNone(second["ingestion"]["job_id"])
        self.assertEqual(self.stored(first["local_path"]).read_bytes(), b"%PDF-1.4 damaged on disk")
        self.assertEqual(self.query("SELECT * FROM versions"), registered)   # 登记仍是旧版
        self.assertEqual(self.query("SELECT * FROM jobs"), jobs)             # 没有多出一个任务
        self.assertEqual(self.files("pdf"), ["report.pdf"])                   # 不留下 .bak/.part 暂存文件
        # 字节真的变了时，登记失败也不能留下半份新版本
        fresh = pdf_bytes("Newer PDF value")
        with patch.object(storage.httpx, "get", return_value=response(content=fresh, url=url)), \
             patch.object(rag_state, "add_version", side_effect=OSError("index locked")):
            third = storage.read_pdf(url)
        self.assertEqual(third["pages"][0]["text"], "Newer PDF value")
        self.assertFalse(third["artifact_saved"])
        self.assertEqual(self.counts(), (1, 1, 1))
        self.assertEqual(self.files("pdf"), ["report.pdf"])
        self.assertEqual(self.stored(first["local_path"]).read_bytes(), b"%PDF-1.4 damaged on disk")

    def test_save_failure_is_reported_but_reading_succeeds(self):
        with patch.object(storage.httpx, "get",
                          return_value=response(content=pdf_bytes(), url="https://example.org/a.pdf")), \
             patch.object(rag_state, "add_version", side_effect=OSError("disk full")):
            result = storage.read_pdf("https://example.org/a.pdf")
        self.assertFalse(result["artifact_saved"])
        self.assertIsNone(result["local_path"])
        self.assertIn("disk full", result["artifact_error"])
        self.assertEqual(result["ingestion"], {"job_id": None, "status": "submit_failed",
                                              "error": result["artifact_error"]})
        self.assertEqual(result["pages"][0]["text"], "Hello PDF world")
        self.assertEqual(self.files(), [])                        # 半份文件被清掉
        self.assertEqual(self.query("SELECT * FROM versions"), [])
        self.assertEqual(self.query("SELECT * FROM jobs"), [])
        # 登记失败只留下一个没有版本的文档壳：没有可入库的东西，检索也就拿不到它
        self.assertEqual(self.counts(), (1, 0, 0))

    def test_artifact_write_failure_is_reported_for_excel(self):
        payload = self.excel_bytes(("Sheet1", [[1]]))
        url = "https://example.org/b.xlsx"
        with patch.object(storage.httpx, "get", return_value=response(content=payload, url=url)), \
             patch.object(storage, "_atomic_write", side_effect=OSError("permission denied")):
            result = storage.read_excel(url, "单格工作簿", preview=False)
        self.assertFalse(result["artifact_saved"])
        self.assertIn("permission denied", result["artifact_error"])
        self.assertEqual(result["ingestion"]["status"], "submit_failed")
        self.assertIsNone(result["ingestion"]["job_id"])
        self.assertEqual(result["rows"], [[1]])
        self.assertEqual(self.counts(), (1, 0, 0))                # 没有版本，也没有入库任务
        self.assertEqual(self.files(), [])


class RunLogTests(ArtifactIsolation, unittest.TestCase):
    """每次调用的 print 落到 logs/<时间>_<query>.txt，控制台输出不受影响。"""

    def run_logged(self, question, model, file_url=None):
        console = io.StringIO()
        with patch.object(agent, "ChatOpenAI", return_value=model), redirect_stdout(console):
            answer = agent.BasicAgent()(question, file_url)
        return answer, console.getvalue(), sorted(self.logs.glob("*.txt"))

    def test_log_is_named_after_query_and_time(self):
        model = FakeModel(lambda n, _: {"name": "run_python", "args": {"code": SCRIPT_OK}, "id": str(n)} if n == 1 else None)
        answer, console, files = self.run_logged("Brio 505 支持哪些视场角?", model)
        self.assertEqual(answer, "12")
        self.assertEqual(len(files), 1)
        self.assertRegex(files[0].name, r"^\d{8}-\d{6}_Brio 505 支持哪些视场角\.txt$")
        content = files[0].read_text(encoding="utf-8")
        self.assertIn("query: Brio 505 支持哪些视场角?", content)
        self.assertIn("[tool call 1/12] run_python", content)
        self.assertIn("[tool result] run_python: success", content)
        self.assertIn("[final] 12", content)
        self.assertIn("[final] 12", console)  # 控制台照常打印，只是多存一份
        self.assertIn("本次运行日志", console)

    def test_log_records_the_attached_file(self):
        _, _, files = self.run_logged("读一下这份资料", FakeModel(lambda n, _: None), "https://example.org/a.pdf")
        content = files[0].read_text(encoding="utf-8")
        self.assertIn("query: 读一下这份资料", content)
        self.assertIn("[file] https://example.org/a.pdf", content)

    def test_log_name_sanitizes_illegal_characters(self):
        _, _, files = self.run_logged("为什么 a/b:c*d?e 会失败", FakeModel(lambda n, _: None))
        self.assertEqual(len(files), 1)
        self.assertRegex(files[0].name, r"^\d{8}-\d{6}_为什么 a_b_c_d_e 会失败\.txt$")

    def test_long_query_is_truncated_in_the_name(self):
        _, _, files = self.run_logged("x" * 120, FakeModel(lambda n, _: None))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].stem.split("_", 1)[1], "x" * 40)

    def test_blank_query_falls_back_to_query(self):
        _, _, files = self.run_logged("   ", FakeModel(lambda n, _: None))
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].name.endswith("_query.txt"))

    def test_same_second_same_query_does_not_overwrite(self):
        first = agent._log_path("同一个问题")
        first.write_text("第一次", encoding="utf-8")
        second = agent._log_path("同一个问题")
        self.assertTrue(first.name.endswith("_同一个问题.txt"))
        self.assertTrue(second.name.endswith("_同一个问题-2.txt"))
        self.assertEqual(first.read_text(encoding="utf-8"), "第一次")

    def test_log_failure_does_not_break_the_run(self):
        console = io.StringIO()
        with patch.object(agent, "_log_path", side_effect=OSError("denied")), \
             patch.object(agent, "ChatOpenAI", return_value=FakeModel(lambda n, _: None)), redirect_stdout(console):
            answer = agent.BasicAgent()("读不到日志也要能回答")
        self.assertEqual(answer, "12")
        self.assertIn("运行日志创建失败", console.getvalue())


class RunPythonTests(ArtifactIsolation, unittest.TestCase):
    """主模型提交的代码在本机执行，不额外调用模型。"""

    def setUp(self):
        super().setUp()
        working = patch.object(tools, "PYTHON_CWD", self.root)
        working.start()
        self.addCleanup(working.stop)
        self.data = self.root / "downloads" / "data.csv"
        self.data.parent.mkdir(parents=True)
        self.data.write_text("name,qty\nbrio,12\nveho,30\n", encoding="utf-8")

    def test_reads_complete_local_file(self):
        code = ("import pandas as pd\n"
                "df = pd.read_csv('downloads/data.csv')\n"
                "print(int(df.qty.sum()))")
        result = tools.run_python.invoke({"code": code})
        self.assertTrue(result["success"], result)
        self.assertEqual(result["output"].strip(), "42")

    def test_small_verified_values_need_no_refetch(self):
        result = tools.run_python.invoke({"code": "print(round(7.741 - 6.725, 3))"})
        self.assertTrue(result["success"])
        self.assertEqual(result["output"].strip(), "1.016")

    def test_error_returns_traceback_for_main_model_to_fix(self):
        result = tools.run_python.invoke({"code": "raise RuntimeError('boom')"})
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "error")
        self.assertIn("RuntimeError: boom", result["error"])
        fixed = tools.run_python.invoke({"code": SCRIPT_OK})
        self.assertTrue(fixed["success"])
        self.assertEqual(fixed["output"].strip(), "12")

    def test_timeout_kills_child(self):
        started = time.monotonic()
        result = tools.run_python.invoke({"code": "import time; time.sleep(20)", "timeout": 1})
        self.assertLess(time.monotonic() - started, 20)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "timeout")

    def test_output_limit(self):
        result = tools.run_python.invoke({"code": "print('x' * 10000)"})
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "output_limit")
        self.assertLessEqual(len(result["error"]), tools.PYTHON_MAX_OUTPUT + 100)

class ExcelPreviewTests(ArtifactIsolation, unittest.TestCase):
    """Excel 首次读取只给工作表/列名/行数/少量样例，模型上下文里不能出现全表。"""

    @staticmethod
    def excel_bytes(rows):
        data = io.BytesIO()
        pd.DataFrame(rows).to_excel(data, header=False, index=False)
        return data.getvalue()

    def large_sheet(self):
        rows = [["Product", "Country", "Profit"]]
        rows += [[f"P{i}", f"C{i}", i] for i in range(700)]
        rows[400] = ["Paseo-sentinel", "Mexico", 1]  # 只出现在表格中段，用于识别"整表进了上下文"
        return self.excel_bytes(rows)

    def test_main_agent_context_never_contains_the_whole_sheet(self):
        """701 行表格：模型上下文只拿到预览，中段的行不会出现。"""
        payload = self.large_sheet()

        def call(n, _messages):
            if n == 1:
                return {"name": "read_excel", "id": "read", "args": {
                    "url": "https://example.org/big.xlsx", "summary": "销售明细：产品、国家、利润三列"}}
            return None

        def read_excel(url, summary, **params):
            """工具包装器走 HTTP；单测里换成进程内直读，这段验证的是上下文大小而不是网络。"""
            return storage.read_excel(url, summary, **params)

        model = FakeModel(call)
        with patch.object(storage.httpx, "get",
                          return_value=response(content=payload, url="https://example.org/big.xlsx")), \
             patch.object(tools.rag_client, "read_excel", new=read_excel), \
             patch.object(agent, "ChatOpenAI", return_value=model), patch("builtins.print"):
            answer = agent.BasicAgent()("这份表里总利润最高的产品是哪个？")
        self.assertTrue(answer)
        context = "\n".join(str(message.content) for message in model.history)
        self.assertIn("total_rows", context)          # 结构预览确实给了模型
        self.assertNotIn("Paseo-sentinel", context)   # 中段数据行没有进上下文
        self.assertLess(len(str(model.history[-1].content)), 1500)
        self.assertIn("run_python", model.registered)
        self.assertNotIn("python_calculator", model.registered)
        self.assertEqual(self.counts()[1:], (1, 1))   # 这一次工具调用登记了一个版本和一个任务

    def test_large_sheet_returns_structure_and_a_small_sample(self):
        rows = [["Product", "Country", "Profit"]] + [[f"P{i}", f"C{i}", i * 1.5] for i in range(700)]
        payload = self.excel_bytes(rows)
        with patch.object(storage.httpx, "get",
                          return_value=response(content=payload, url="https://example.org/big.xlsx")):
            result = storage.read_excel("https://example.org/big.xlsx", "销售明细：产品、国家、利润")
        self.assertEqual(result["mode"], "preview")
        self.assertEqual(result["total_rows"], 701)
        self.assertEqual(result["columns"], ["Product", "Country", "Profit"])
        self.assertEqual(result["sample_rows"][0], ["Product", "Country", "Profit"])
        self.assertEqual(result["sample_row_count"], 5)
        self.assertNotIn("P400", json.dumps(result, ensure_ascii=False))
        self.assertLess(len(json.dumps(result, ensure_ascii=False)), 1200)
        self.assertEqual(result["local_path"], self.path_of("excel", "big.xlsx"))
        self.assertTrue(self.stored(result["local_path"]).is_file())  # 完整数据在本地文件里
        self.assertTrue(result["artifact_saved"])
        self.assertTrue(result["ingestion"]["job_id"])               # 描述已就绪，等后台切块入库
        self.assertEqual(result["summary"], "销售明细：产品、国家、利润")

    def test_run_python_uses_read_excel_local_path(self):
        payload = self.excel_bytes([["amount"], [2], [3]])
        with patch.object(storage.httpx, "get", return_value=response(
                content=payload, url="https://example.org/sum.xlsx")):
            preview = storage.read_excel("https://example.org/sum.xlsx", "金额单列表")
        code = ("import pandas as pd\n"
                f"print(int(pd.read_excel({preview['local_path']!r})['amount'].sum()))")
        with patch.object(tools, "PYTHON_CWD", self.root):
            result = tools.run_python.invoke({"code": code})
        self.assertTrue(result["success"], result)
        self.assertEqual(result["output"].strip(), "5")

    def test_explicit_block_read_still_works(self):
        rows = [["名称", "数量"], ["甲", 1], ["乙", 2], ["丙", 3]]
        payload = self.excel_bytes(rows)
        with patch.object(storage.httpx, "get",
                          return_value=response(content=payload, url="https://example.org/small.xlsx")):
            result = storage.read_excel("https://example.org/small.xlsx", "名称与数量两列", preview=False,
                                        max_rows=2)
        self.assertEqual(result["mode"], "block")
        self.assertEqual(result["rows"], rows[:2])
        self.assertEqual((result["next_start_row"], result["next_start_column"]), (2, None))

    def test_preview_rows_bounds_are_validated(self):
        url = "https://example.org/a.xlsx"
        with patch.object(storage.httpx, "get",
                          return_value=response(content=self.excel_bytes([["A"]]), url=url)) as get:
            with self.assertRaises(ValueError):        # 越界的预览行数会在解析前被拒绝
                storage.read_excel(url, "单格表", preview_rows=99)
            with self.assertRaises(TypeError):         # 非整数不静默兜底成默认值
                storage.read_excel(url, "单格表", preview_rows="many")
            get.assert_not_called()                    # 参数不合法时连文件都不该下载
        self.assertEqual(self.counts(), (0, 0, 0))


class RunPythonRecoveryTests(ArtifactIsolation, unittest.TestCase):
    def test_main_model_repairs_failed_code_without_second_model(self):
        def call(n, messages):
            if n == 1:
                return {"name": "run_python", "args": {"code": "raise RuntimeError('boom')"}, "id": "bad"}
            if n == 2:
                self.assertIn("RuntimeError: boom", messages[-1].content)
                return {"name": "run_python", "args": {"code": SCRIPT_OK}, "id": "fixed"}
            self.assertEqual(json.loads(messages[-1].content)["output"].strip(), "12")
            return None

        model = FakeModel(call)
        with patch.object(agent, "ChatOpenAI", return_value=model) as model_factory, patch("builtins.print"):
            answer = agent.BasicAgent()("算 3*4")
        model_factory.assert_called_once()
        self.assertEqual(answer, "12")
        self.assertEqual(model.rounds, 3)
        self.assertEqual(model.final_calls, 0)

    def test_failure_does_not_block_search(self):
        calls = [("run_python", {"code": "raise RuntimeError('boom')"}),
                 ("web_search", {"query": "more evidence"})]
        model = FakeModel(lambda n, _: {"name": calls[n-1][0], "args": calls[n-1][1], "id": str(n)}
                          if n <= len(calls) else None)
        with patch.object(agent, "ChatOpenAI", return_value=model), patch("builtins.print"), \
             patch.object(tools, "_ddgs_search", return_value=[]):
            agent.BasicAgent()("查找并计算")
        self.assertTrue(any(isinstance(message, ToolMessage) and message.name == "web_search"
                            and message.status == "success" for message in model.history))

class ConcurrencyTests(ArtifactIsolation, unittest.TestCase):
    def test_parallel_run_python_calls_do_not_share_script(self):
        barrier = threading.Barrier(3, timeout=20)
        results = {}

        def worker(name, value):
            barrier.wait()
            results[name] = tools.run_python.invoke({
                "code": f"import time\ntime.sleep(0.5)\nprint('{name}', {value})"})

        threads = [threading.Thread(target=worker, args=("A", 1)),
                   threading.Thread(target=worker, args=("B", 2))]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=20)
        self.assertEqual(results["A"]["output"].strip(), "A 1")
        self.assertEqual(results["B"]["output"].strip(), "B 2")

class AcceptanceScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "scripts" / "acceptance.py"
        spec = importlib.util.spec_from_file_location("agent_acceptance", path)
        cls.acceptance = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.acceptance)

    def test_verdicts(self):
        ok = [{"success": True, "status": "ok", "output": "Paseo, Mexico"}]
        failure = [{"success": False, "status": "error", "error": "RuntimeError: boom"}]
        self.assertEqual(self.acceptance.check("excel", "最高 Paseo，最低 Mexico", ok), [])
        self.assertTrue(self.acceptance.check("excel", "最高 Paseo", ok))
        self.assertTrue(self.acceptance.check("excel", "最高 Paseo，最低 Mexico", []))
        self.assertTrue(self.acceptance.check("excel", "最高 Paseo，最低 Mexico", failure))
        self.assertEqual(self.acceptance.check("excel", "最高 Paseo，最低 Mexico", [*failure, *ok]), [])
        self.assertEqual(self.acceptance.check("error", "计算失败，文件不存在", failure), [])
        self.assertTrue(self.acceptance.check("error", "总和是 42", failure))

    def test_transcript_extraction_ignores_prompt_text(self):
        transcript = "\n".join([
            "ai []: 请调用 run_python 计算",
            "ai []: {\"name\": \"run_python\", \"args\": {\"code\": \"print(12)\"}}",
            "tool run_python [success]: {\"success\": false, \"error\": \"failed\"}",
            "tool run_python [success]: {\"success\": true, \"status\": \"ok\", \"output\": \"12\"}",
        ])
        calls = self.acceptance.run_python_calls(transcript)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1]["output"], "12")

if __name__ == "__main__":
    unittest.main()
