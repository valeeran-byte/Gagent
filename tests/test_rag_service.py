"""文件服务与 RAG 的验收测试：任务状态机、版本切换、生命周期和检索输入隔离。

默认用可复现的假 embedding（同时记录真实调用参数），只有 TestRetrievalQuality 用
项目里已下载的 e5 模型跑真实向量；除生命周期用例拉起一个隔离目录里的真实服务进程外，
其余都不联网、不读写项目自己的 rag_data/。
"""
from __future__ import annotations

from datetime import datetime
import http.server
import io
import json
import os
import pathlib
import re
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from rag import client
from rag import service as file_service
from rag import storage as file_storage
from rag import state as rag_state
from rag import vectors as rag_store


MODEL_TOKENS = re.compile(r"\w+|[^\w\s]")

# 所有临时资料目录集中在一个父目录下：Chroma 的全局实例缓存会让句柄迟迟不释放，
# 逐个删不干净；改成每次运行开头整体清空这一个父目录，垃圾不会越攒越多。
SCRATCH = pathlib.Path(tempfile.gettempdir()) / "gagent_rag_tests"
shutil.rmtree(SCRATCH, ignore_errors=True)


def pdf_bytes(page_texts=("",), repeats: int = 1) -> bytes:
    writer = PdfWriter()
    for text in page_texts:
        for _ in range(max(1, repeats)):
            page = writer.add_blank_page(width=400, height=400)
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                     NameObject("/Subtype"): NameObject("/Type1"),
                                     NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
            stream = DecodedStreamObject()
            stream.set_data(b"BT /F1 12 Tf 20 380 Td (" + text.encode("latin-1", "replace") + b") Tj ET")
            page[NameObject("/Contents")] = stream
    data = io.BytesIO()
    writer.write(data)
    return data.getvalue()


def xlsx_bytes(rows=(("名称", "数值"), ("A", 1))) -> bytes:
    import pandas as pd
    buffer = io.BytesIO()
    pd.DataFrame(list(rows[1:]), columns=list(rows[0])).to_excel(buffer, index=False)
    return buffer.getvalue()


def httpx_response(url: str, content: bytes, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(200, request=httpx.Request("GET", str(url)), content=content,
                          headers=headers or {})


def fake_fetch(content: bytes, url: str, name: str):
    return file_storage.Fetched(content=content, url=url, source_url=rag_state.canonical_url(url),
                                file_name=name, from_cache=False, remote_checked=True, verified_at=None)


class FakeModel:
    """替代 SentenceTransformer：向量由词重叠决定，并记录每次 encode 的真实入参。"""

    def __init__(self):
        self.dimension = 64
        self.passage_calls: list[list[str]] = []
        self.query_calls: list[list[str]] = []
        self.max_seq_length = 512

    def tokenizer(self, texts, add_special_tokens=True):
        return {"input_ids": [[0] + [1] * len(MODEL_TOKENS.findall(text)) for text in texts]}

    def encode(self, texts, **kwargs):
        if texts and str(texts[0]).startswith("query: "):
            self.query_calls.append(list(texts))
        else:
            self.passage_calls.append(list(texts))
        return np.array([self._vector(text) for text in texts], dtype="float32")

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype="float32")
        body = text.split(": ", 1)[-1]
        for token in set(word.lower() for word in MODEL_TOKENS.findall(body)):
            vector[abs(hash(token)) % self.dimension] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


class TempRoot:
    """把资料目录挪到项目外的临时目录；模型仍在项目里，不重复下载。"""

    def setUp(self):
        self.root = SCRATCH / f"t_{uuid_hex()}"
        self.patch = patch.object(rag_state, "DATA_ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.wipe)
        rag_state.ensure_dirs()

    def wipe(self) -> None:
        """能删就删；删不动的留给下次运行开头统一清掉，不为句柄释放去睡眠等待。"""
        rag_store._collection = rag_store._client = rag_store._collection_root = None
        shutil.rmtree(self.root, ignore_errors=True)


def uuid_hex() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


class RagEnabled:
    """只在需要真实客户端链路的用例里临时打开文件服务，不改动其他测试模块的环境。"""

    def setUp(self):
        super().setUp()
        patcher = patch.dict(os.environ, {"GAGENT_RAG_DISABLE": ""})
        patcher.start()
        self.addCleanup(patcher.stop)


class ServiceCase(TempRoot, unittest.TestCase):
    """用 TestClient 跑真实 ASGI 应用 + 真实后台执行器，embedding 用假模型。"""

    def setUp(self):
        super().setUp()
        self.model = FakeModel()
        model_patch = patch.object(rag_store, "get_model", return_value=self.model)
        model_patch.start()
        self.addCleanup(model_patch.stop)
        ready_patch = patch.object(rag_store, "model_ready", return_value=True)
        ready_patch.start()
        self.addCleanup(ready_patch.stop)
        self._tester = None
        self.addCleanup(self.close_client)

    def client(self) -> TestClient:
        """一个用例只保留一个应用实例：每次请求都重启会把后台执行器从任务中途掐断。"""
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

    def restart(self) -> TestClient:
        """关掉再开：模拟服务进程重启，恢复逻辑走的就是新实例的启动流程。"""
        self.close_client()
        time.sleep(0.2)
        return self.client()

    def fast_retries(self) -> None:
        """把重试间隔改短，只测状态机不真等 5s/30s（默认值另有断言覆盖）。"""
        patcher = patch.object(rag_state, "RETRY_DELAYS", (0.05, 0.1))
        patcher.start()
        self.addCleanup(patcher.stop)

    def post(self, path: str, payload: dict) -> tuple[int, dict]:
        reply = self.client().post(path, json=payload)
        return reply.status_code, reply.json()

    def get(self, path: str, params: dict | None = None) -> tuple[int, dict]:
        reply = self.client().get(path, params=params or {})
        return reply.status_code, reply.json()

    def wait_for(self, predicate, timeout: float = 20.0, message: str = "条件未满足") -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        self.fail(message)

    def job(self, job_id: str) -> dict:
        conn = rag_state.connect()
        try:
            return rag_state.get_job(conn, job_id)
        finally:
            conn.close()

    def version(self, version_id: int) -> dict:
        conn = rag_state.connect()
        try:
            return rag_state.get_version(conn, version_id)
        finally:
            conn.close()


class TestReadEndpoints(ServiceCase):
    def test_health_identifies_service_and_protocol(self):
        status, body = self.get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["protocol"], rag_state.PROTOCOL)
        self.assertEqual(body["service_id"], rag_state.service_id())
        self.assertTrue(body["ok"])

    def test_read_pdf_returns_content_and_queued_job(self):
        payload = pdf_bytes(("Finland ranks first in the happiness report",))
        with patch.object(file_storage, "fetch", return_value=fake_fetch(payload, "https://e.org/r.pdf", "r.pdf")):
            status, body = self.post("/v1/read/pdf", {"url": "https://e.org/r.pdf", "max_pages": 1})
        self.assertEqual(status, 200)
        self.assertTrue(body["read_ok"])
        self.assertEqual(body["page_count"], 1)
        self.assertEqual(body["pages"][0]["page"], 1)
        self.assertIn("Finland", body["pages"][0]["text"])
        self.assertEqual(body["version"], rag_state.content_digest(payload))
        self.assertTrue(pathlib.Path(body["local_path"]).is_file())
        self.assertEqual(body["ingestion"]["status"], "queued")
        self.assertTrue(body["artifact_saved"])

    def test_excel_requires_non_blank_summary_on_both_sides(self):
        for summary in ("", "   "):
            status, body = self.post("/v1/read/excel", {"url": "https://e.org/x.xlsx", "summary": summary})
            self.assertEqual(status, 400, summary)
            self.assertIn("summary", body["message"])
        with patch.object(file_storage, "fetch", side_effect=lambda url, kind, refresh=False:
                          fake_fetch(xlsx_bytes(), "https://e.org/x.xlsx", "x.xlsx")):
            self.assertRaises(ValueError, file_storage.read_excel, "https://e.org/x.xlsx", "  ")

    def test_read_failure_is_reported_as_error_not_accepted(self):
        with patch.object(file_storage, "fetch", side_effect=ValueError("响应不是 PDF 文件，可能是下载错误页")):
            status, body = self.post("/v1/read/pdf", {"url": "https://e.org/bad.pdf"})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "bad_request")
        self.assertIn("PDF", body["message"])

    def test_submit_failure_keeps_read_content_but_says_so(self):
        payload = pdf_bytes(("readable content",))
        with patch.object(file_storage, "fetch", return_value=fake_fetch(payload, "https://e.org/r.pdf", "r.pdf")), \
                patch.object(rag_state, "ensure_job", side_effect=OSError("disk full")):
            status, body = self.post("/v1/read/pdf", {"url": "https://e.org/r.pdf"})
        self.assertEqual(status, 200)
        self.assertTrue(body["read_ok"], "文件内容仍然读到了")
        self.assertFalse(body["artifact_saved"])
        self.assertEqual(body["ingestion"]["status"], "submit_failed")
        self.assertIn("disk full", body["artifact_error"])

    def test_cache_hit_does_not_claim_remote_recheck(self):
        """缓存命中要如实说明"没重新核验远端"；refresh 才真的回源。"""
        payload = pdf_bytes(("cached copy",))
        calls = []

        def get(url, **kwargs):
            calls.append(str(url))
            return httpx_response(url, payload, {"content-disposition": "attachment; filename=r.pdf"})

        with patch.object(file_storage.httpx, "get", side_effect=get):
            _, saved = self.post("/v1/read/pdf", {"url": "https://e.org/r.pdf"})
            _, first = self.post("/v1/read/pdf", {"url": "https://e.org/r.pdf"})
            _, refreshed = self.post("/v1/read/pdf", {"url": "https://e.org/r.pdf", "refresh": True})
        self.assertEqual(len(calls), 2, "第二次读取不该联网")
        self.assertTrue(saved["remote_checked"])
        self.assertFalse(first["remote_checked"])
        self.assertIn("未重新核验", first["version_note"])
        self.assertTrue(refreshed["remote_checked"])
        self.assertIn("已重新核验", refreshed["version_note"])

    def test_empty_library_search_returns_no_results(self):
        status, body = self.post("/v1/search", {"query": "幸福报告里芬兰排第几"})
        self.assertEqual(status, 200)
        self.assertEqual(body["results"], [])


class TestIngestionJobs(ServiceCase):
    def submit_pdf(self, tester: TestClient, text: str = "Finland is the happiest country", name: str = "r.pdf"):
        payload = pdf_bytes((text,))
        with patch.object(file_storage, "fetch", return_value=fake_fetch(payload, f"https://e.org/{name}", name)):
            reply = tester.post("/v1/read/pdf", json={"url": f"https://e.org/{name}"})
        return reply.json(), rag_state.content_digest(payload)

    def test_pdf_indexes_whole_file_not_only_read_pages(self):
        pages = [f"page {index} about happiness scores and social support" for index in range(1, 8)]
        payload = pdf_bytes(pages)
        with patch.object(file_storage, "fetch", return_value=fake_fetch(payload, "https://e.org/many.pdf", "many.pdf")):
            body = self.client().post("/v1/read/pdf", json={"url": "https://e.org/many.pdf", "max_pages": 1}).json()
        self.assertEqual(len(body["pages"]), 1, "本次只返回要读的那一页")
        job_id = body["ingestion"]["job_id"]
        self.wait_for(lambda: self.job(job_id)["status"] == "indexed",
                      message=f"任务未完成：{self.job(job_id)}")
        stored = self.version(self.job(job_id)["version_id"])
        self.assertEqual(stored["page_count"], 7)
        found = rag_store.collection().get(where={"$and": [{"doc_id": stored["doc_id"]},
                                                           {"version": stored["sha256"]}]},
                                          include=["metadatas"])
        pages_indexed = {meta["page"] for meta in found["metadatas"]}
        self.assertEqual(pages_indexed, set(range(1, 8)), "入库必须覆盖整份 PDF 的可提取文字")
        self.assertTrue(all(str(text).startswith("passage: ") for batch in self.model.passage_calls
                            for text in batch), "资料必须用 passage 前缀")

    def test_scanned_pdf_fails_without_retrying(self):
        payload = pdf_bytes(("", ""))
        with patch.object(file_storage, "fetch", return_value=fake_fetch(payload, "https://e.org/scan.pdf", "scan.pdf")):
            body = self.client().post("/v1/read/pdf", json={"url": "https://e.org/scan.pdf"}).json()
        job_id = body["ingestion"]["job_id"]
        self.wait_for(lambda: self.job(job_id)["status"] == "failed")
        job = self.job(job_id)
        self.assertEqual(job["attempts"], 1, "无可提取文字是确定性问题，不该重跑")
        self.assertIn("OCR", job["last_error"])
        events = [(row["kind"], row["message"]) for row in self.get("/v1/events")[1]["events"]]
        self.assertEqual(events[0][0], "failed")
        self.assertIn("已停止自动重试", events[0][1])

    def test_retry_is_capped_at_three_attempts_surviving_restart(self):
        self.fast_retries()
        payload = pdf_bytes(("broken after read",))
        attempts = []

        def explode(version):
            attempts.append(version["sha256"])
            raise OSError("向量库暂时写入失败")

        tester = self.client()
        with patch.object(file_storage, "fetch", return_value=fake_fetch(payload, "https://e.org/r.pdf", "r.pdf")), \
                patch.object(rag_store, "index_version", side_effect=explode):
            body = tester.post("/v1/read/pdf", json={"url": "https://e.org/r.pdf"}).json()
            job_id = body["ingestion"]["job_id"]
            self.wait_for(lambda: self.job(job_id)["status"] == "failed", timeout=30,
                          message=f"没有走到终态：{self.job(job_id)}")
            self.assertEqual(len(attempts), 3, "首次执行 + 最多两次自动重试")
            self.assertEqual(self.job(job_id)["attempts"], 3)
            self.assertEqual(self.job(job_id)["round"], 1)

            # 服务重启（新的事件循环和执行器）不刷新次数，也不回退到 queued
            restarted = self.restart()
            time.sleep(1.0)
            job = self.job(job_id)
            self.assertEqual(job["attempts"], 3, "重启不能重置尝试次数")
            self.assertEqual(job["status"], "failed")
            again = restarted.post("/v1/read/pdf", json={"url": "https://e.org/r.pdf", "start_page": 1}).json()
        self.assertEqual(again["ingestion"]["job_id"], job_id)
        self.assertEqual(again["ingestion"]["status"], "failed")
        self.assertEqual(self.job(job_id)["attempts"], 3, "同版本再次被读取不能偷偷重置次数")

    def test_user_retry_creates_new_round(self):
        self.fast_retries()
        payload = pdf_bytes(("needs manual retry",))
        tester = self.client()

        def explode(version):
            raise OSError("临时故障")

        with patch.object(file_storage, "fetch", return_value=fake_fetch(payload, "https://e.org/r.pdf", "r.pdf")), \
                patch.object(rag_store, "index_version", side_effect=explode):
            body = tester.post("/v1/read/pdf", json={"url": "https://e.org/r.pdf"}).json()
            job_id = body["ingestion"]["job_id"]
            self.wait_for(lambda: self.job(job_id)["status"] == "failed", timeout=30,
                          message=f"没有失败：{self.job(job_id)}")
            status, reply = self.post(f"/v1/jobs/{job_id}/retry", {})
            self.assertEqual(status, 200)
            self.assertEqual(reply["round"], 2)
            self.assertNotEqual(reply["job_id"], job_id)
            self.assertEqual(self.job(job_id)["status"], "failed", "旧一轮保持终态")
            status, message = self.post(f"/v1/jobs/{job_id}/retry", {})
            self.assertEqual(status, 409, "已经结束的旧一轮不能重复重试")

    def test_interrupted_attempt_counts_and_resumes(self):
        """进程在处理中被杀掉：那一次尝试仍算已用，重启后接着下一轮而不是从零开始。"""
        self.fast_retries()
        path = rag_state.files_dir() / "pdf" / "r.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf_bytes(("crash mid job",)))
        conn = rag_state.connect()
        try:
            doc = rag_state.get_or_create_document(conn, "https://e.org/r.pdf", "pdf", "r.pdf")
            version, _ = rag_state.add_version(conn, doc_id=doc["doc_id"], sha256="c" * 64, kind="pdf",
                                               local_path=str(path), size=1, file_name="r.pdf",
                                               final_url="u", summary=None, page_count=1, empty_pages=[],
                                               sheet_names=None)
            job, _ = rag_state.ensure_job(conn, doc["doc_id"], version["id"], "pdf")
            # 模拟"已 claim 并记了 1 次尝试，然后进程被杀"
            conn.execute("UPDATE jobs SET status='processing', attempts=1 WHERE job_id=?", (job["job_id"],))
        finally:
            conn.close()
        job_id = job["job_id"]
        with patch.object(rag_store, "index_version", side_effect=OSError("仍在故障")) as blocker:
            self.client()
            # 先等"这次尝试真的开始执行"，再判断次数：claim 会先写 attempts，两步之间有时差
            self.wait_for(lambda: self.job(job_id)["attempts"] >= 2, timeout=30,
                          message=f"重启后没有接管中断任务：{self.job(job_id)}")
            self.wait_for(lambda: blocker.call_count >= 1, timeout=30,
                          message="接管后没有真正执行入库阶段")
        self.assertEqual(self.job(job_id)["attempts"], 2, "被中断的那一次不能重跑，也不能被清零")

    def test_retry_delays_and_attempt_cap_match_the_spec(self):
        self.assertEqual((rag_state.JOB_MAX_ATTEMPTS, rag_state.RETRY_DELAYS), (3, (5.0, 30.0)))

    def test_pagination_and_repeat_reads_create_one_job(self):
        payload = pdf_bytes(("first page about happiness", "second page about rankings",
                             "third page about methodology"))
        tester = self.client()
        job_ids = set()
        with patch.object(file_storage, "fetch", return_value=fake_fetch(payload, "https://e.org/r.pdf", "r.pdf")):
            for page in (1, 1, 2, 3):
                body = tester.post("/v1/read/pdf", json={"url": "https://e.org/r.pdf", "start_page": page}).json()
                job_ids.add(body["ingestion"]["job_id"])
            find = tester.post("/v1/find/pdf", json={"url": "https://e.org/r.pdf", "query": "two"}).json()
            job_ids.add(find["ingestion"]["job_id"])
        self.assertEqual(len(job_ids), 1, "翻页/重复读取/find 不能各自造一个入库任务")
        conn = rag_state.connect()
        try:
            rows = conn.execute("SELECT COUNT(*) AS n FROM versions").fetchone()["n"]
            jobs = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
        finally:
            conn.close()
        self.assertEqual((rows, jobs), (1, 1))

    def test_rerunning_a_job_does_not_duplicate_vectors(self):
        payload = pdf_bytes(("deterministic chunk ids",))
        tester = self.client()
        with patch.object(file_storage, "fetch", return_value=fake_fetch(payload, "https://e.org/r.pdf", "r.pdf")):
            body = tester.post("/v1/read/pdf", json={"url": "https://e.org/r.pdf"}).json()
        job_id = body["ingestion"]["job_id"]
        self.wait_for(lambda: self.job(job_id)["status"] == "indexed", timeout=30)
        first = rag_store.vector_count()
        version = self.version(self.job(job_id)["version_id"])
        rag_store.index_version(version)
        rag_store.index_version(version)
        self.assertEqual(rag_store.vector_count(), first, "重复执行只覆盖相同 id")


class TestVersionSwitch(ServiceCase):
    def indexed_pages(self, doc_id: str, sha: str) -> set[int]:
        found = rag_store.collection().get(where={"$and": [{"doc_id": doc_id}, {"version": sha}]},
                                           include=["metadatas"])
        return {meta["page"] for meta in found["metadatas"]}

    def active_flag(self, doc_id: str, sha: str) -> set[int]:
        found = rag_store.collection().get(where={"$and": [{"doc_id": doc_id}, {"version": sha}]},
                                           include=["metadatas"])
        return {int(meta["active"]) for meta in found["metadatas"]}

    def test_new_version_replaces_old_only_after_completion(self):
        tester = self.client()
        first = pdf_bytes(("happiness report 2023 edition", "annex about sampling"))
        with patch.object(file_storage, "fetch", return_value=fake_fetch(first, "https://e.org/w.pdf", "w.pdf")):
            old = tester.post("/v1/read/pdf", json={"url": "https://e.org/w.pdf"}).json()
        old_job = old["ingestion"]["job_id"]
        self.wait_for(lambda: self.job(old_job)["status"] == "indexed", timeout=30)
        old_version = self.version(self.job(old_job)["version_id"])

        block = threading.Event()
        self.addCleanup(block.set)  # 断言失败也要放行执行器，否则它会带着旧补丁继续跑
        original = rag_store.index_version

        def slow_index(version):
            # 卡到测试放行位置：给一个固定秒数的等待会在负载高时提前放行，判成"已经切换"
            if version["id"] != old_version["id"]:
                self.assertTrue(block.wait(120), "新版本入库阶段没被放行")
            return original(version)

        second = pdf_bytes(("happiness report 2024 edition", "annex about sampling"))
        with patch.object(file_storage, "fetch", return_value=fake_fetch(second, "https://e.org/w.pdf", "w.pdf")), \
                patch.object(rag_store, "index_version", side_effect=slow_index):
            new = tester.post("/v1/read/pdf", json={"url": "https://e.org/w.pdf"}).json()
            new_job = new["ingestion"]["job_id"]
            self.assertNotEqual(new["version"], old["version"])
            # 卡住新版本的写入阶段，检查"新向量还没写完就不切换"这段窗口
            self.wait_for(lambda: self.job(new_job)["status"] == "processing", timeout=20,
                          message="新版本没进入处理中")
            hits = rag_store.search("happiness report 2024 edition", pdf_min_score=0.0)
            self.assertTrue(hits, "旧版本仍可作为当前有效结果")
            self.assertTrue(all(hit["version"] == old_version["sha256"] for hit in hits),
                            [(hit["version"][:8], hit["page"]) for hit in hits])
            self.assertTrue(all(hit["newer_version_pending"] for hit in hits), "必须带旧版本标记")
            block.set()
            self.wait_for(lambda: self.job(new_job)["status"] == "indexed", timeout=30)
        after = rag_store.search("happiness report 2024 edition", pdf_min_score=0.0)
        self.assertTrue(all(hit["version"] == new["version"] for hit in after))
        self.assertEqual(self.indexed_pages(old_version["doc_id"], old_version["sha256"]), set())

    def test_late_finishing_old_job_does_not_rollback(self):
        conn = rag_state.connect()
        try:
            doc = rag_state.get_or_create_document(conn, "https://e.org/w.pdf", "pdf", "w.pdf")
            older, _ = rag_state.add_version(conn, doc_id=doc["doc_id"], sha256="a" * 64, kind="pdf",
                                             local_path="x", size=1, file_name="w.pdf", final_url="u",
                                             summary=None, page_count=1, empty_pages=[], sheet_names=None)
            newer, _ = rag_state.add_version(conn, doc_id=doc["doc_id"], sha256="b" * 64, kind="pdf",
                                             local_path="y", size=1, file_name="w.pdf", final_url="u",
                                             summary=None, page_count=1, empty_pages=[], sheet_names=None)
            self.assertEqual(rag_state.activate_version(conn, newer["id"]), (True, None))
            switched, superseded = rag_state.activate_version(conn, older["id"])
            self.assertEqual((switched, superseded), (False, None), "旧版本晚完成不能把有效版本退回去")
            self.assertEqual(rag_state.active_version_id(conn, doc["doc_id"]), newer["id"])
            self.assertEqual(rag_state.get_version(conn, older["id"])["status"], "indexed")
        finally:
            conn.close()
        state = rag_state.ingestion_state(rag_state.connect(), [doc["doc_id"]])[doc["doc_id"]]
        self.assertEqual(state["active_version"], newer["id"])

    def test_startup_reconciles_active_flags(self):
        conn = rag_state.connect()
        try:
            doc = rag_state.get_or_create_document(conn, "https://e.org/w.pdf", "pdf", "w.pdf")
            version, _ = rag_state.add_version(conn, doc_id=doc["doc_id"], sha256="c" * 64, kind="pdf",
                                               local_path="p", size=1, file_name="w.pdf", final_url="u",
                                               summary=None, page_count=2, empty_pages=[], sheet_names=None)
        finally:
            conn.close()
        path = rag_state.files_dir() / "pdf" / "w.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf_bytes(("happiness scores and rankings",)))
        record = rag_state.get_version(rag_state.connect(), version["id"])
        record = {**record, "local_path": str(path)}
        count = rag_store.index_version(record)
        self.assertEqual(rag_store.version_chunk_count(record["doc_id"], record["sha256"]), count)
        self.assertEqual(self.active_flag(record["doc_id"], record["sha256"]), {0},
                         "写完先不算有效版本")
        conn = rag_state.connect()
        try:
            rag_state.activate_version(conn, record["id"])  # 库里已切换，向量标记还停在写入时的状态
        finally:
            conn.close()
        self.assertEqual(rag_store.reconcile_active(), count)
        self.assertEqual(self.active_flag(record["doc_id"], record["sha256"]), {1})
        rag_store.set_version_active(record["doc_id"], record["sha256"], False)
        rag_store.reconcile_active()
        self.assertEqual(self.active_flag(record["doc_id"], record["sha256"]), {1})


class TestModelBootstrap(unittest.TestCase):
    def test_use_or_download_reuses_complete_local_model(self):
        with patch.object(rag_store, "model_ready", return_value=True), \
                patch.object(rag_store, "prune_model_dir") as prune, \
                patch.object(rag_store, "prepare_model") as download:
            self.assertEqual(rag_store.use_or_download(), str(rag_store.model_dir()))
        prune.assert_called_once_with()
        download.assert_not_called()

    def test_use_or_download_downloads_when_model_is_missing(self):
        with patch.object(rag_store, "model_ready", return_value=False), \
                patch.object(rag_store, "prepare_model", return_value="downloaded-model") as download:
            self.assertEqual(rag_store.use_or_download(), "downloaded-model")
        download.assert_called_once_with()


class TestEmbeddingContract(ServiceCase):
    def chunks(self, text: str) -> list[dict]:
        return rag_store.chunk_page(text, 1, "doc-aaaaaaaa")

    def test_prefixes_applied_to_both_sides(self):
        records = rag_store.build_excel_records({"id": 1, "doc_id": "d", "sha256": "e" * 64, "kind": "excel",
                                                 "file_name": "x.xlsx", "final_url": "u", "local_path": "p",
                                                 "summary": "世界幸福报告数据表，含各国得分与排名"})
        rag_store._upsert([r["id"] for r in records], [r["text"] for r in records],
                          [rag_store._metadata_of({"id": 1, "doc_id": "d", "sha256": "e" * 64, "kind": "excel",
                                                   "file_name": "x.xlsx", "final_url": "u", "local_path": "p"},
                                                  rag_store.EXCEL_KIND) for _ in records])
        self.assertTrue(all(text.startswith("passage: ") for batch in self.model.passage_calls
                            for text in batch))
        rag_store.encode_query("芬兰排第几")
        self.assertTrue(all(text.startswith("query: ") for batch in self.model.query_calls for text in batch))

    def test_vectors_are_normalized_and_nothing_is_truncated(self):
        long_text = "happiness report " * 900  # 远超模型长度上限，必须继续拆而不是截断
        records = self.chunks(long_text)
        self.assertGreater(len(records), 3)
        for record in records:
            self.assertLessEqual(rag_store.count_tokens(record["text"]), rag_store.CHUNK_TARGET_TOKENS,
                                 "每块不得超过目标长度，也就不会被模型截断")
        vectors = rag_store.encode_passages([r["text"] for r in records[:3]])
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)

    def test_chunk_sizes_around_three_hundred_tokens_with_overlap(self):
        page = "\n\n".join(f"paragraph {index} explains happiness scores social support healthy life"
                           f" expectancy freedom generosity and corruption perception in detail"
                           for index in range(60))
        records = self.chunks(page)
        sizes = [rag_store.count_tokens(r["text"]) for r in records]
        self.assertGreater(len(records), 2, sizes)
        self.assertTrue(all(size <= rag_store.CHUNK_TARGET_TOKENS for size in sizes), sizes)
        self.assertGreater(max(sizes), 250, "块应该接近 350 token 而不是碎成小片")
        starts = [r["start_char"] for r in records]
        ends = [r["end_char"] for r in records]
        self.assertEqual(starts, sorted(starts))
        self.assertLess(starts[1], ends[0], "相邻块要按设定重叠，不能硬切在段落边界")
        joined = "\n".join(r["text"] for r in records)
        self.assertIn("paragraph 0", joined)
        self.assertIn("paragraph 59", joined, "整页内容都必须出现在切块里")

    def test_scan_pages_without_text_recorded(self):
        payload = pdf_bytes(("readable page", "", "another readable page"))
        with patch.object(file_storage, "fetch", return_value=fake_fetch(payload, "https://e.org/m.pdf", "m.pdf")):
            body = self.client().post("/v1/read/pdf", json={"url": "https://e.org/m.pdf", "max_pages": 1}).json()
        job_id = body["ingestion"]["job_id"]
        self.wait_for(lambda: self.job(job_id)["status"] == "indexed", timeout=30)
        stored = self.version(self.job(job_id)["version_id"])
        self.assertEqual(json.loads(stored["empty_pages"]), [2], "无文字页要记录，不能当作已索引全文")
        self.assertEqual(stored["page_count"], 3)


class TestSearchIsolation(ServiceCase):
    def seed(self):
        """写两条 PDF 片段并激活其版本，返回 document 行。"""
        conn = rag_state.connect()
        try:
            doc = rag_state.get_or_create_document(conn, "https://e.org/w.pdf", "pdf", "w.pdf")
            version, _ = rag_state.add_version(conn, doc_id=doc["doc_id"], sha256="f" * 64, kind="pdf",
                                               local_path="p", size=1, file_name="w.pdf",
                                               final_url="https://e.org/w.pdf", summary=None,
                                               page_count=2, empty_pages=[], sheet_names=None)
            rag_state.activate_version(conn, version["id"])
        finally:
            conn.close()
        records = [{"id": f"{version['doc_id']}-p1-0", "text": "Finland ranks first with a happiness score of 7.741",
                    "page": 1, "start_char": 0, "end_char": 48},
                   {"id": f"{version['doc_id']}-p2-0", "text": "The United States ranks 23rd with a score of 6.920",
                    "page": 2, "start_char": 0, "end_char": 46}]
        rag_store._upsert([r["id"] for r in records], [r["text"] for r in records],
                          [rag_store._metadata_of(version, rag_store.PDF_KIND, page=r["page"],
                                                  start_char=r["start_char"], end_char=r["end_char"])
                           for r in records])
        rag_store.set_version_active(version["doc_id"], version["sha256"], True)
        return version

    def test_query_reaches_embedding_verbatim(self):
        self.seed()
        self.model.query_calls.clear()
        question = "美国与芬兰的幸福得分差距有多大？"
        status, body = self.post("/v1/search", {"query": question})
        self.assertEqual(status, 200)
        sent = [text for batch in self.model.query_calls for text in batch]
        self.assertEqual(sent, ["query: " + question], "embedding 入参只能是本轮问题原文")
        for text in sent:
            self.assertNotIn(str(datetime.now().year - 1900), text)
            self.assertNotIn("当前日期时间", text)
            self.assertNotIn("SYSTEM_PROMPT", text)
            self.assertNotIn(file_service.__name__, text)

    def test_long_query_split_without_history(self):
        self.seed()
        self.model.query_calls.clear()
        question = " ".join(f"sentence {index} compares national happiness scores"
                            for index in range(80))
        _, body = self.post("/v1/search", {"query": question})
        sent = [text for batch in self.model.query_calls for text in batch]
        self.assertGreater(len(sent), 1, "超长输入应按长度拆分")
        for text in sent:
            self.assertTrue(text.startswith("query: "))
            self.assertTrue(text[len("query: "):] in question, "拆分内容必须来自本轮输入本身")
        self.assertIsInstance(body["results"], list)

    def test_stale_vectors_cannot_starve_real_results(self):
        """库里残留的 active 标记向量不能占掉候选名额，让真资料查不到。"""
        version = self.seed()  # 两条正常命中：Finland / United States
        ghost = {"id": 99, "doc_id": "doc_ghost", "sha256": "9" * 64, "kind": "pdf",
                 "file_name": "ghost.pdf", "final_url": "https://e.org/ghost.pdf", "local_path": "g"}
        meta = rag_store._metadata_of(ghost, rag_store.PDF_KIND, page=1)
        meta["active"] = 1  # SQLite 里根本没有这条版本记录，向量却带着 active 标记
        rag_store._upsert(["ghost-p1-0"], ["Finland happiness score 7.741 ranking first"], [meta])
        query = rag_store.encode_query("Finland happiness score")
        only_ghost = rag_store._hits(rag_store.PDF_KIND, query, 1, [])
        self.assertEqual(only_ghost, [], "没有有效版本时不该把残留向量当资料返回")
        picked = rag_store._hits(rag_store.PDF_KIND, query, 1, [version["sha256"]])
        self.assertEqual([row["version"] for row in picked], [version["sha256"]],
                         "top-1 名额必须落在有效版本上")
        hits = rag_store.search("Finland happiness score", pdf_min_score=0.0)
        self.assertTrue(hits and all(hit["version"] == version["sha256"] for hit in hits))
        self.assertEqual(rag_store.drop_orphans(), 1)
        self.assertNotIn("ghost-p1-0", rag_store.collection().get(include=[])["ids"])

    def test_result_shape_carries_provenance(self):
        self.seed()
        _, body = self.post("/v1/search", {"query": "Finland happiness score 7.741",
                                           "pdf_min_score": 0.0})
        results = body["results"]
        self.assertTrue(results)
        first = results[0]
        self.assertEqual(first["source_id"], "R1")
        self.assertEqual(first["kind"], "pdf_chunk")
        self.assertEqual(first["file_name"], "w.pdf")
        self.assertEqual(first["version"], "f" * 64)
        self.assertIn("page", first)
        self.assertIn("local_path", first)
        self.assertNotIn("metadata", first)

    def test_scores_and_limits_applied(self):
        doc = self.seed()
        _, all_hits = self.post("/v1/search", {"query": "Finland happiness score", "pdf_min_score": 0.0,
                                               "max_results": 1, "max_chars": 6000})
        self.assertEqual(len(all_hits["results"]), 1, "max_results 必须生效")
        _, filtered = self.post("/v1/search", {"query": "Finland happiness score", "pdf_min_score": 0.999})
        self.assertEqual(filtered["results"], [])
        conn = rag_state.connect()
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"], 0,
                             "检索不该产生事件或写入")
        finally:
            conn.close()


class TestServiceDiscovery(RagEnabled, TempRoot, unittest.TestCase):
    """服务发现：只认本项目、本协议的实例，端口被别人占用时不误用。"""

    def write_info(self, **overrides) -> dict:
        info = {"port": 4599, "pid": 1234, "protocol": rag_state.PROTOCOL,
                "service_id": rag_state.service_id(), "started_at": "now"}
        info.update(overrides)
        rag_state.service_info_path().write_text(json.dumps(info), encoding="utf-8")
        return info

    def test_missing_or_foreign_info_is_ignored(self):
        self.assertIsNone(client.running())
        self.write_info(service_id="another-project")
        self.assertIsNone(client.running(), "别的项目留下的 service.json 不能被当成自己的服务")
        rag_state.service_info_path().write_text("{ not json", encoding="utf-8")
        self.assertIsNone(client.running())

    def test_protocol_and_health_mismatch(self):
        info = self.write_info()

        class Reply:
            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

        with patch.object(client, "service_info", return_value=info):
            with patch.object(client.httpx, "get", side_effect=OSError("connection refused")):
                self.assertIsNone(client.running(), "探活失败（端口被别的程序占用）不复用")
            for payload in ({"ok": False}, {"ok": True, "protocol": "gagent-rag/0",
                                             "service_id": info["service_id"]},
                            {"ok": True, "protocol": rag_state.PROTOCOL, "service_id": "other"}):
                with patch.object(client.httpx, "get", return_value=Reply(payload)):
                    self.assertIsNone(client.running(), f"健康应答不匹配必须拒绝：{payload}")
            with patch.object(client.httpx, "get", return_value=Reply(
                    {"ok": True, "protocol": rag_state.PROTOCOL, "service_id": info["service_id"]})):
                self.assertEqual(client.running()["port"], 4599)

    def test_endpoint_does_not_start_service_for_polling(self):
        with patch.object(client, "running", return_value=None):
            self.assertRaises(client.ServiceUnavailable, client.endpoint, False)
            with patch.object(client, "_ensure_started", return_value={"port": 4599}) as started:
                self.assertEqual(client.endpoint(), "http://127.0.0.1:4599")
                started.assert_called_once()
            self.assertEqual(client.endpoint(), "http://127.0.0.1:4599", "地址有缓存，不会反复探活")

    def test_disabled_switch_avoids_spawning(self):
        with patch.dict(os.environ, {"GAGENT_RAG_DISABLE": "1"}):
            self.assertFalse(client.enabled())
            with patch.object(client, "_spawn") as spawner:
                self.assertRaises(client.ServiceUnavailable, client.endpoint)
                spawner.assert_not_called()
            out = client.search("问题")
            self.assertEqual(out["results"], [])
            self.assertIn("GAGENT_RAG_DISABLE", out["reason"])

    def test_search_never_raises_out(self):
        with patch.object(client, "_request", side_effect=client.ServiceUnavailable("服务没起来", "unavailable")):
            out = client.search("问题")
        self.assertEqual(out["results"], [])
        self.assertIn("服务没起来", out["reason"])
        with patch.object(client, "_request", side_effect=client.ServiceError("参数错误", "bad_request")):
            self.assertEqual(client.search("问题")["results"], [])


class TestDetachedLifecycle(RagEnabled, TempRoot, unittest.TestCase):
    """CLI 退出后服务继续处理任务：真实子进程 + 真实隔离数据目录。"""

    def setUp(self):
        super().setUp()
        self.addCleanup(self.stop_service)  # 用例自己拉起的服务必须收掉，别留后台进程
        self.served = {"/slow.pdf": pdf_bytes(("happiness report annex with many pages " * 30,), repeats=6)}
        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)

    def stop_service(self) -> None:
        """用例结束就收掉自己拉起的后台服务，别在用户机器上留僵尸进程。"""
        try:
            info = json.loads(rag_state.service_info_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        pid = info.get("pid")
        if pid:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)

    def _handler(self):
        served = self.served

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = served.get(self.path.split("?")[0])
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        return Handler

    def run_client(self, code: str) -> str:
        script = f"""
import json, os, sys
sys.path.insert(0, {json.dumps(str(pathlib.Path(__file__).resolve().parents[1]))})
os.environ["GAGENT_RAG_DATA"] = {json.dumps(str(self.root))}
from rag import client
{code}
"""
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", script], capture_output=True,
                                timeout=180, cwd=str(pathlib.Path(__file__).resolve().parents[1]))
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        return result.stdout.decode("utf-8", "replace")

    def test_service_keeps_working_after_client_exits(self):
        output = self.run_client(f"""
body = client.read_pdf("{self.base}/slow.pdf", max_pages=1)
print(json.dumps({{"job": body["ingestion"]["job_id"], "service_pid": client.health()["pid"]}}))
""")
        submitted = json.loads(output.strip().splitlines()[-1])
        job_id, service_pid = submitted["job"], submitted["service_pid"]
        # 提交任务的客户端进程已经退出，服务进程必须继续把任务做完
        deadline, statuses = time.monotonic() + 150, []
        while time.monotonic() < deadline:
            status = json.loads(self.run_client(f"""
print(json.dumps(client.job("{job_id}")["status"]))
""").strip().splitlines()[-1])
            statuses.append(status)
            if status in ("indexed", "failed"):
                break
            time.sleep(3)
        self.assertEqual(statuses[-1], "indexed", f"任务没在后台完成：{statuses}")
        events = json.loads(self.run_client("""
print(json.dumps([[e["kind"], e["message"]] for e in client.events_after(0)["events"]]))
""").strip().splitlines()[-1])
        self.assertEqual(events[-1][0], "indexed")
        self.assertIn("已加入长期记忆", events[-1][1])
        hits = json.loads(self.run_client("""
print(json.dumps([[h["kind"], h.get("page"), round(h["score"], 3)]
                  for h in client.search("happiness report annex")["results"]]))
""").strip().splitlines()[-1])
        self.assertTrue(hits, "新客户端进程应能检索到已入库资料")
        self.assertEqual(hits[0][0], "pdf_chunk")
        again = json.loads(self.run_client(f"""
print(json.dumps(client.health()["pid"]))
""").strip().splitlines()[-1])
        self.assertEqual(again, service_pid, "第二个客户端复用了同一个服务实例，没有再起一个")

if __name__ == "__main__":
    unittest.main()
