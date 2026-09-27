"""跨会话回忆的独立验收脚本：真实 e5 模型 + 真实 ASGI 服务 + 真实会话原文。

设计口径（独立于实现者的自述）：
- 会话原文全部复制/写入临时 sessions 目录；向量与 SQLite 挪到临时 DATA_ROOT，
  项目自己的 sessions/ 只读、rag_data/ 一个字节都不写。
- 索引与检索走真实链路：TestClient 触发 lifespan → sync_all → 后台执行器用
  真实 multilingual-e5-small 建向量 → HTTP /v1/conversations/search|recall。
- 查询集由本脚本独立标注：14 个正样本（期望命中的会话）、18 个无关问题（期望
  not_found）、1 个近似重复会话对（期望 ambiguous）。每个样本都写下标注依据。
- 结论阈值写死在下面两条，先给原始数字再给结论，不用"接口成功/库里有数据"代替：
    POSITIVE_RATE_FLOOR = 0.85   （正样本找到正确会话的比例下限）
    UNRELATED_FP_CEIL   = 0.20   （无关问题误命中比例上限）
  另外附带更严的观察线（>=0.90 / 0），只报告不参与退出码。

退出码：0 = 没有 FAIL；1 = 有 FAIL；2 = 前置条件不满足（模型缺失等）。

用法：
    & "C:\\Python\\envs\\agent\\python.exe" -X utf8 scripts/acceptance_conversations.py
    可选：--skip-regression（跳过 11 号标准的三个回归套件）
          --stub-summary（摘要不用真实 API，改用桩；默认用真实 API）
          --keep（保留临时目录便于排查）
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

PROJECT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import conversation_summary  # noqa: E402
import session_store  # noqa: E402
from rag import conversations as conv  # noqa: E402
from rag import state as rag_state  # noqa: E402
from rag import vectors as rag_store  # noqa: E402

REAL_SESSIONS = PROJECT / "sessions"
POSITIVE_RATE_FLOOR = 0.85
UNRELATED_FP_CEIL = 0.20
CANARY_KEY = "sk-acceptance-canary-must-not-leak-9f21"

# ---------------------------------------------------------------- 标注查询集
# 每条正样本：(query, 期望会话, 标注依据)。query 全部是重新措辞，不复制原文。
POSITIVES: list[tuple[str, str, str]] = [
    ("2025 年 LPL 派去全球总决赛的四支战队是哪几个", "99c1e6",
     "真实会话 99c1e6 标题/首轮主题的换说法"),
    ("S15 那届 LPL 四支队伍最后分别打到了哪一轮", "99c1e6",
     "真实会话 99c1e6 第 2 轮的最终战绩（TES 四强等）"),
    ("2026 年 LPL 出征 S16 的队伍名单都有谁", "9735a1",
     "真实会话 9735a1 标题/首轮主题的换说法"),
    ("HLE 拿了 MSI 冠军、T1 又三连冠，LPL 今年还有机会吗", "9735a1",
     "真实会话 9735a1 第 2 轮独有的 LCK 压制内容"),
    ("多头注意力相比只用单个注意力头，解决了什么问题", "be0fa5",
     "真实会话 be0fa5 的论文核心论点换说法"),
    ("Transformer 论文为什么把 QKV 投影到多个子空间", "be0fa5",
     "真实会话 be0fa5 第 3.2.2 节内容换说法"),
    ("Excel 文件入库最后决定用哪种方式", "ingest01",
     "合成会话 ingest01：Excel 用模型给的 summary 入库"),
    ("PDF 入库需要先让模型写 summary 吗", "ingest01",
     "合成会话 ingest01：PDF 直接提取正文，与 Excel 相反"),
    ("缓存层最后确定用哪个方案", "cache01",
     "合成会话 cache01：先 LRU 后被改成 SQLite"),
    ("为什么不继续用进程内 LRU 做缓存了", "cache01",
     "合成会话 cache01 第 2 轮的推翻理由"),
    ("灰度发布第一批放多少流量进去", "deploy01",
     "合成长会话 deploy01 第 8 轮的中段细节（5%）"),
    ("自动回滚的触发条件是怎么设的", "deploy01",
     "合成长会话 deploy01 第 12 轮的中段细节（错误率/延迟阈值）"),
    ("文档切块的长度和重叠参数是怎么定的", "chunk01",
     "合成会话 chunk01：350 目标 / 50 重叠 / 496 上限"),
    ("备份策略最后按哪个方案执行", "refer01",
     "合成会话 refer01：上一轮给两个方案，本轮只说『用第二种』/『按这个做』"),
]

# 无关问题：与全部 11 个已索引会话的主题都不搭界；期望 not_found，不能强塞历史。
# 独立标注，覆盖 18 个不同领域与不同长度（短 6 字 ~ 长 40 字），避免 0/6 的置信度太低。
UNRELATED: list[tuple[str, str]] = [
    ("红烧肉要炖多久才能软烂", "烹饪/短"),
    ("怎么用 pandas 读取 CSV 并做数据透视表", "数据分析（与 Excel 入库主题表面接近，故意保留）"),
    ("吉他换弦的步骤是什么", "乐器维护/短"),
    ("量子纠缠的贝尔不等式怎么推导", "物理"),
    ("明天北京天气怎么样", "天气/短"),
    ("猫咪多久洗一次澡比较合适", "宠物/短"),
    ("为什么海水是咸的而湖水是淡的", "地理/短"),
    ("孩子发烧到 38.5 度应该先物理降温还是直接吃退烧药", "儿科医学/长"),
    ("二手房交易里满五唯一和满二唯一的税费差多少", "房产税务/长"),
    ("明天去西安旅游三天怎么安排路线比较合理", "旅游/长"),
    ("怎么用一个平底锅做出不塌的舒芙蕾", "烘焙/中"),
    ("李白和杜甫的诗歌风格有什么区别", "文学/中"),
    ("足球越位规则这两年改了什么", "体育（非电竞）/中"),
    ("如何申请劳动仲裁以及需要准备哪些材料", "法律/长"),
    ("阳台上适合种哪些不容易死的香草", "园艺/中"),
    ("新手买相机应该看哪些参数", "摄影/短"),
    ("怎么给旧手机换电池而不弄坏屏幕", "数码维修/中"),
    ("黑洞照片是怎么拍出来的", "天文/短"),
]

# 近似重复会话对：两个会话正文完全一致、只有 id 和标题不同；期望 ambiguous 而不是二选一。
AMBIGUOUS: list[tuple[str, tuple[str, str]]] = [
    ("季度预算方案最后定的是哪一版", ("budgeta", "budgetb")),
]

# 标准 3 指代：这三个字面量出现在 refer01 的后续轮里，只有带上一轮上下文才能理解。
REFERENT_MARKERS = ("用第二种", "按这个做")

CHUNK_LABELS = ("会话标题：", "上一轮用户问题：", "上一轮回答中的邻近内容：",
                "本轮用户问题：", "本轮最终回答：")
CONTEXT_LABELS = CHUNK_LABELS[:3]


def compose_tokens(text: str) -> dict:
    """拆一条片段：上下文（标题/上一轮）× 本轮主体各占多少 token、有几行上下文。"""
    context = body = context_lines = 0
    for line in str(text or "").splitlines():
        for name in CHUNK_LABELS:
            if line.startswith(name):
                tokens = conv._tokens(line.split("：", 1)[1])
                if name in CONTEXT_LABELS:
                    context += tokens
                    if name != "会话标题：":
                        context_lines += 1
                else:
                    body += tokens
                break
    total = conv._tokens(text)
    return {"total": total, "context": context, "body": body,
            "context_lines": context_lines,
            "context_share": round(context / total, 4) if total else 0.0}


# ---------------------------------------------------------------- 语料

def filler(tag: str, count: int) -> str:
    """确定性可复现的长文本：让长会话真的有很多片段，把中段推到很深的位置。"""
    pool = [
        f"{tag}：发布前要跑完整回归，包括接口契约、离线任务和前端冒烟。",
        f"{tag}：监控面板要预先加好错误率、延迟、吞吐三块指标。",
        f"{tag}：值班表提前一周排好，联系人写进发布单。",
        f"{tag}：变更窗口避开业务高峰，默认放在周二凌晨。",
        f"{tag}：配置项全部走配置中心，不允许手改线上文件。",
    ]
    return "".join(pool[index % len(pool)] for index in range(count))


def turn(user: str, answer: str) -> dict:
    return {"user": user, "final_answer": answer}


def synthetic_sessions() -> list[dict]:
    stamp = "2026-03-01T09:00:00+08:00"
    out: list[dict] = []

    def make(session_id: str, title: str, turns: list[dict]) -> dict:
        return {"id": session_id, "title": title, "created_at": stamp, "updated_at": stamp,
                "turns": turns}

    out.append(make("ingest01", "文件入库方案", [
        turn("我们要给文件做 RAG 入库，PDF 和 Excel 分别怎么处理？",
             "PDF 直接提取正文后切块入库；Excel 先让模型读网页介绍生成 summary，"
             "再按这段 summary 建向量，不逐单元格向量化。"),
        turn("Excel 那块确定用 summary 了吗？",
             "确定：Excel 用模型提供的 summary 入库，PDF 则直接提取正文，两者互不混用。"),
    ]))
    out.append(make("cache01", "缓存方案定稿", [
        turn("缓存层就用进程内 LRU 吧，简单", "好，缓存层用进程内 LRU。"),
        turn("算了，不用 LRU 了，改成本地 SQLite 缓存",
             "好，缓存改用本地 SQLite，不再用进程内 LRU。"),
    ]))
    out.append(make("refer01", "备份策略确认", [
        turn("备份策略有两种方案：第一种是每天全量备份，第二种是每周全量加每天增量。你建议哪种？",
             "建议第二种：每周全量 + 每天增量，恢复更快，日常开销也更小。"),
        turn("用第二种", "好，按第二种执行：每周全量备份，工作日每天做增量备份。"),
        turn("按这个做", "已按第二种方案落地：每周全量 + 每天增量。"),
    ]))
    deploy_turns: list[dict] = []
    for index in range(16):
        if index == 7:
            deploy_turns.append(turn(
                "灰度上线的第一批要放多少流量？",
                "第一批只放 5% 流量，观察 30 分钟没有异常再逐步放大到 25%、50%、最后 100%。"
                + filler("灰度", 6)))
        elif index == 11:
            deploy_turns.append(turn(
                "如果出问题怎么自动回滚？",
                "自动回滚触发条件：5 分钟内接口错误率超过 1%，或者 P99 延迟超过 800ms 持续 3 分钟；"
                "触发后立即切回上一个稳定版本。" + filler("回滚", 6)))
        else:
            deploy_turns.append(turn(
                f"上线第 {index} 个准备项要注意什么？", f"第 {index} 项：" + filler(f"准备{index}", 7)))
    out.append(make("deploy01", "上线部署与灰度", deploy_turns))
    out.append(make("budgeta", "季度预算方案 A", [
        turn("季度预算方案最后定的是哪一版？", "季度预算方案最后定 A 版：控制成本优先。")]))
    out.append(make("budgetb", "季度预算方案 B", [
        turn("季度预算方案最后定的是哪一版？", "季度预算方案最后定 A 版：控制成本优先。")]))
    out.append(make("chunk01", "文档切块参数", [
        turn("文档切块的长度和重叠参数是怎么定的？",
             "目标 350 token 一块，上一块结尾重叠 50 token，硬上限 496 token；"
             "超过上限继续拆，不静默截断。")]))
    out.append(make("docker01", "部署环境迁移", [
        turn("部署环境最后定用 Docker Swarm 还是别的编排？",
             "部署环境最后定为 Docker Swarm，集群只有三台机器，先不上 Kubernetes。")]))
    return out


def load_real_sessions() -> list[dict]:
    store = session_store.SessionStore(REAL_SESSIONS)
    sessions = []
    for path in sorted(REAL_SESSIONS.glob("*.json")):
        if path.name == session_store.STATE_FILENAME:
            continue
        sessions.append(store.read(path.stem))
    return sessions


# ---------------------------------------------------------------- 结果收集

class Report:
    def __init__(self) -> None:
        self.standards: list[dict] = []
        self.metrics: dict = {}
        self.defects: list[dict] = []

    def add(self, number: int, title: str, ok, detail: str, evidence: str = "") -> None:
        # ok 可能是 None（未验证）或任意真值表达式；这里统一收敛成三态，避免把 list 当键。
        verdict = "SKIP" if ok is None else ("PASS" if bool(ok) else "FAIL")
        self.standards.append({"no": number, "title": title, "verdict": verdict,
                               "detail": detail, "evidence": evidence})

    def add_defect(self, defect_id: str, severity: str, title: str, detail: str,
                   evidence: str = "") -> None:
        self.defects.append({"id": defect_id, "severity": severity, "title": title,
                             "detail": detail, "evidence": evidence})

    def print(self) -> None:
        print("\n" + "=" * 78)
        print("逐条结论")
        print("=" * 78)
        for item in self.standards:
            print(f"[{item['verdict']}] {item['no']}. {item['title']}")
            print(f"        {item['detail']}")
            if item["evidence"]:
                print(f"        证据：{item['evidence']}")
        if self.defects:
            print("\n" + "=" * 78)
            print("发现的缺陷（带实测数字）")
            print("=" * 78)
            for item in self.defects:
                print(f"[{item['severity']}] {item['id']} {item['title']}")
                print(f"        {item['detail']}")
                if item["evidence"]:
                    print(f"        复现：{item['evidence']}")


# ---------------------------------------------------------------- 验收主体

class Acceptance:
    def __init__(self, *, use_real_summary: bool, keep: bool) -> None:
        self.report = Report()
        self.use_real_summary = use_real_summary
        self.keep = keep
        self.root = pathlib.Path(tempfile.gettempdir()) / f"gagent_accept_conversations_{os.getpid()}"
        self.sessions = self.root / "sessions"
        self.data = self.root / "data"
        self.patchers: list = []
        self.client = None
        self.llm_calls: list[str] = []
        self.expected_indexed = 0

    # ---------- 环境 ----------

    def setup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self.sessions.mkdir(parents=True, exist_ok=True)
        self.data.mkdir(parents=True, exist_ok=True)
        # 双保险：进程内 patch + 环境变量（rag.client._spawn 会把环境变量传给服务子进程）。
        self._env_backup = os.environ.get("GAGENT_SESSIONS_DIR")
        os.environ["GAGENT_SESSIONS_DIR"] = str(self.sessions)
        self._patch(rag_state, "DATA_ROOT", self.data)
        self._patch(session_store, "SESSIONS_DIR", self.sessions)
        # 注意：rag/conversations.py 在导入期 `from session_store import SESSIONS_DIR` 绑定的是
        # 导入时的值，所以 conv.* 一律显式传 sessions_dir=self.sessions。
        self._reset_vector_singletons()
        rag_state.ensure_dirs()
        assert pathlib.Path(rag_state.DATA_ROOT) == self.data, "DATA_ROOT 未隔离"
        assert conv._store_for(self.sessions).root == self.sessions, "会话目录未隔离"

    def _patch(self, target, name, value) -> None:
        patcher = patch.object(target, name, value)
        patcher.start()
        self.patchers.append(patcher)

    def _reset_vector_singletons(self) -> None:
        rag_store._collection = getattr(rag_store, "_client", None)
        rag_store._collection = None
        rag_store._client = None
        rag_store._collection_root = None
        conv._collection = None
        conv._collection_root = None

    def teardown(self) -> None:
        if self.client is not None:
            try:
                self.client.__exit__(None, None, None)
            except Exception:
                pass
            self.client = None
        for patcher in reversed(self.patchers):
            try:
                patcher.stop()
            except Exception:
                pass
        if getattr(self, "_env_backup", None) is None:
            os.environ.pop("GAGENT_SESSIONS_DIR", None)
        else:
            os.environ["GAGENT_SESSIONS_DIR"] = self._env_backup
        self._reset_vector_singletons()
        if not self.keep:
            shutil.rmtree(self.root, ignore_errors=True)

    # ---------- 写语料 ----------

    def write_corpus(self) -> None:
        store = session_store.SessionStore(self.sessions)
        self.real = load_real_sessions()
        for session in self.real:
            store.write(session)
        self.synthetic = synthetic_sessions()
        for session in self.synthetic:
            store.write(session)
        indexable = sum(1 for item in self.real + self.synthetic if item.get("turns"))
        self.expected_indexed = indexable
        print(f"语料：真实会话 {len(self.real)} 个（有轮次 "
              f"{sum(1 for s in self.real if s['turns'])}）+ 合成会话 {len(self.synthetic)} 个；"
              f"预期可索引 {indexable} 个")

    # ---------- 起服务 ----------

    def start_service(self) -> None:
        from fastapi.testclient import TestClient
        from rag import service as file_service

        if self.use_real_summary:
            real_factory = conversation_summary._default_llm_factory
            calls = self.llm_calls

            def counting_factory(settings, timeout):
                calls.append(settings.get("MODEL", ""))
                return real_factory(settings, timeout)

            self._patch(conversation_summary, "_default_llm_factory", counting_factory)
        started = time.monotonic()
        tester = TestClient(file_service.app)
        tester.__enter__()  # 触发 lifespan：sync_all + 后台执行器
        self.client = tester
        print(f"服务已启动（含 lifespan 同步），等待真实 e5 建索引……")
        self.wait_indexed()
        print(f"索引完成：{conv.vector_count()} 条向量，用时 {time.monotonic() - started:.1f}s")

    def wait_indexed(self, timeout: float = 300.0) -> None:
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            conn = rag_state.connect()
            try:
                pending = conv.pending_count(conn)
                indexed = conv.indexed_count(conn)
                failed = conv.failed_count(conn)
            finally:
                conn.close()
            last = f"indexed={indexed} pending={pending} failed={failed}"
            if pending == 0 and failed == 0 and indexed >= self.expected_indexed:
                print(f"  {last}")
                return
            if failed:
                break
            time.sleep(0.5)
        raise RuntimeError(f"索引未在 {timeout}s 内完成：{last}")

    # ---------- HTTP 封装 ----------

    def post(self, path: str, payload: dict) -> tuple[int, dict]:
        reply = self.client.post(path, json=payload)
        return reply.status_code, reply.json()

    def get(self, path: str, params: dict | None = None) -> tuple[int, dict]:
        reply = self.client.get(path, params=params or {})
        return reply.status_code, reply.json()

    def search(self, query: str, **kwargs) -> dict:
        """模块层检索：与阈值校准同一口径（默认 0.86 / 0.006 / 0.93）。"""
        kwargs.setdefault("sessions_dir", self.sessions)
        return conv.search_conversations(query, **kwargs)

    def cos(self, query: str, text: str) -> float:
        """直接算余弦：用于把"全片段分数"和"只有本轮正文的分数"拆开对比。"""
        import numpy as np
        left = np.array(rag_store.encode_query(query), dtype="float32")
        right = rag_store.encode_passages([text])[0]
        return float(np.dot(left, right))

    def recall(self, query: str, *, expect_status: int = 200, **payload) -> dict:
        body = {"query": query, **payload}
        status, data = self.post("/v1/conversations/recall", body)
        assert status == expect_status, f"recall HTTP {status}: {data}"
        return data

    # ================================================================ 指标

    def measure_quality(self) -> None:
        print("\n" + "=" * 78)
        print("关键指标：正样本命中 / 无关误命中（真实 e5，默认阈值 0.86）")
        print("=" * 78)
        rows = []
        hits = 0
        relaxed_hits = 0
        best_scores = []
        for query, expected, why in POSITIVES:
            result = self.search(query)
            top = (result["candidates"] or [{}])[0]
            top_id = top.get("session_id")
            top_score = top.get("score")
            present = expected in {item["session_id"] for item in result["candidates"] or []}
            hit = result["status"] == "found" and top_id == expected
            relaxed = present and result["status"] in ("found", "ambiguous")
            hits += 1 if hit else 0
            relaxed_hits += 1 if relaxed else 0
            if top_score is not None:
                best_scores.append((top_score, query, expected, top_id))
            rows.append({"query": query, "expected": expected, "why": why,
                         "status": result["status"], "top1": top_id, "top1_score": top_score,
                         "hit": hit, "relaxed_hit": relaxed})
            flag = "命中" if hit else ("含正确会话" if relaxed else "未命中")
            print(f"  [{flag}] {query[:34]:34s} 期望={expected} 实际top1={top_id} "
                  f"score={top_score} status={result['status']}")
        positive_rate = hits / len(POSITIVES)
        relaxed_rate = relaxed_hits / len(POSITIVES)

        false_hits = 0
        unrelated_scores = []
        false_rows = []
        for query, why in UNRELATED:
            result = self.search(query)
            top = (result["candidates"] or [{}])[0]
            top_score = top.get("score")
            wrong = result["status"] in ("found", "ambiguous")
            false_hits += 1 if wrong else 0
            if top_score is not None:
                unrelated_scores.append((top_score, query, top.get("session_id")))
            if wrong:
                hits_of_top = top.get("hits") or []
                best_hit = hits_of_top[0] if hits_of_top else {}
                false_rows.append({"query": query, "status": result["status"],
                                   "session": top.get("session_id"), "score": top_score,
                                   "turn_id": best_hit.get("turn_id"),
                                   "turn_index": best_hit.get("turn_index"),
                                   "excerpt": str(best_hit.get("text") or "")[:160]})
            print(f"  [{'误命中' if wrong else '正确拒绝'}] {query[:30]:30s} "
                  f"status={result['status']} top1={top.get('session_id')} score={top_score}"
                  + (f" 命中轮次={(top.get('hits') or [{}])[0].get('turn_id')}" if wrong else ""))
        unrelated_fp = false_hits / len(UNRELATED)

        best_scores.sort(key=lambda item: -item[0])
        unrelated_scores.sort(key=lambda item: -item[0])
        self.report.metrics["positive_total"] = len(POSITIVES)
        self.report.metrics["positive_hits"] = hits
        self.report.metrics["positive_rate"] = round(positive_rate, 4)
        self.report.metrics["positive_relaxed_hits"] = relaxed_hits
        self.report.metrics["positive_relaxed_rate"] = round(relaxed_rate, 4)
        self.report.metrics["unrelated_total"] = len(UNRELATED)
        self.report.metrics["unrelated_false_hits"] = false_hits
        self.report.metrics["unrelated_fp_rate"] = round(unrelated_fp, 4)
        self.report.metrics["positive_rows"] = rows
        self.report.metrics["best_positive_min"] = best_scores[-1][0] if best_scores else None
        self.report.metrics["unrelated_max"] = unrelated_scores[0][0] if unrelated_scores else None
        self.report.metrics["worst_positive"] = best_scores[-1] if best_scores else None
        self.report.metrics["worst_unrelated"] = unrelated_scores[0] if unrelated_scores else None
        self.report.metrics["false_positive_rows"] = false_rows
        print(f"\n  >>> 找到正确会话（严格：status=found 且 top1 正确）："
              f"{hits}/{len(POSITIVES)} = {positive_rate:.1%}")
        print(f"  >>> 找到正确会话（宽松：正确会话在候选里、found 或 ambiguous）："
              f"{relaxed_hits}/{len(POSITIVES)} = {relaxed_rate:.1%}")
        print(f"  >>> 无关问题误命中：{false_hits}/{len(UNRELATED)} = {unrelated_fp:.1%}")
        if best_scores and unrelated_scores:
            print(f"  >>> 正样本最高分区间 {best_scores[-1][0]:.4f}-{best_scores[0][0]:.4f}；"
                  f"无关问题最高分 {unrelated_scores[0][0]:.4f}（{unrelated_scores[0][1]}）")
            print(f"  >>> 正样本最低分 {best_scores[-1][0]:.4f}（{best_scores[-1][1]}）")
        print(f"  >>> 更严观察线（正样本 >= 90% 且无关 0%）："
              f"{'达到' if positive_rate >= 0.90 and false_hits == 0 else '未达到'}")

    def measure_threshold_sweep(self) -> None:
        """逐阈值重跑检索，给出"正样本命中率 × 误命中率"，并把 found 与 ambiguous 分开统计。"""
        thresholds = (0.855, 0.860, 0.865, 0.870, 0.875, 0.880)
        rows = []
        print("\n阈值扫描（每个阈值重新检索；误命中只统计 status=found/ambiguous，"
              "not_found 但报候选不计入）")
        print(f"  {'阈值':>6} {'正样本严格':>11} {'正样本宽松':>11} {'误命中合计':>11} "
              f"{'其中found':>10} {'其中ambig':>10} {'仅候选(允许)':>12}")
        for threshold in thresholds:
            strict = relaxed = 0
            fp_total = fp_found = fp_ambiguous = near_miss = 0
            for query, expected, _why in POSITIVES:
                result = self.search(query, min_score=threshold)
                ids = {item["session_id"] for item in result["candidates"] or []}
                top = (result["candidates"] or [{}])[0].get("session_id")
                if result["status"] == "found" and top == expected:
                    strict += 1
                if expected in ids and result["status"] in ("found", "ambiguous"):
                    relaxed += 1
            for query, _why in UNRELATED:
                result = self.search(query, min_score=threshold)
                status = result["status"]
                if status == "found":
                    fp_found += 1
                elif status == "ambiguous":
                    fp_ambiguous += 1
                elif result["candidates"]:
                    near_miss += 1
                fp_total += 1 if status in ("found", "ambiguous") else 0
            rows.append({"threshold": threshold,
                         "positive_strict": strict,
                         "positive_strict_rate": round(strict / len(POSITIVES), 4),
                         "positive_relaxed": relaxed,
                         "positive_relaxed_rate": round(relaxed / len(POSITIVES), 4),
                         "false_hits": fp_total, "fp_rate": round(fp_total / len(UNRELATED), 4),
                         "fp_found": fp_found, "fp_ambiguous": fp_ambiguous,
                         "near_miss_only": near_miss})
            print(f"  {threshold:>6.3f} {strict:>6}/{len(POSITIVES):<4} {relaxed:>6}/{len(POSITIVES):<4} "
                  f"{fp_total:>6}/{len(UNRELATED):<4} {fp_found:>10} {fp_ambiguous:>10} {near_miss:>12}")
        self.report.metrics["threshold_sweep"] = rows
        # 分数间隔：一次低门槛检索拿到每个查询的 top1 分数
        probes = []
        for query, expected, _why in POSITIVES:
            result = self.search(query, min_score=0.50)
            top = (result["candidates"] or [{}])[0]
            probes.append(("pos", query, expected, top.get("score"), top.get("session_id")))
        for query, _why in UNRELATED:
            result = self.search(query, min_score=0.50)
            top = (result["candidates"] or [{}])[0]
            probes.append(("neg", query, None, top.get("score"), top.get("session_id")))
        positives = [item for item in probes if item[0] == "pos" and item[4] == item[2] and item[3]]
        negatives = [item for item in probes if item[0] == "neg" and item[3]]
        worst_positive = min(positives, key=lambda item: item[3]) if positives else None
        worst_negative = max(negatives, key=lambda item: item[3]) if negatives else None
        if worst_positive and worst_negative:
            print(f"  正确 top1 的最低正样本分={worst_positive[3]:.4f}（{worst_positive[1]}）；"
                  f"最高无关分={worst_negative[3]:.4f}（{worst_negative[1]}，"
                  f"top1={worst_negative[4]}）；间隔={worst_positive[3] - worst_negative[3]:.4f}")
        self.report.metrics["separation"] = {
            "lowest_correct_positive": [worst_positive[3], worst_positive[1]] if worst_positive else None,
            "highest_unrelated": [worst_negative[3], worst_negative[1],
                                  worst_negative[4]] if worst_negative else None,
            "gap": round(worst_positive[3] - worst_negative[3], 4)
            if worst_positive and worst_negative else None}

    def check_ambiguous(self) -> dict:
        query, pair = AMBIGUOUS[0]
        result = self.search(query)
        ids = {item["session_id"] for item in result["candidates"] or []}
        scores = [item["score"] for item in result["candidates"] or []]
        print(f"\n  近似重复会话对：{query} → status={result['status']} ids={sorted(ids)} "
              f"scores={scores}")
        return {"query": query, "status": result["status"], "ids": sorted(ids),
                "expected": sorted(pair), "scores": scores}

    # ================================================================ 11 条标准

    def standard_1(self) -> None:
        """旧 session 自动建索引，新 session 能找回以前讨论（真实 sessions/）。"""
        conn = rag_state.connect()
        try:
            indexed = conv.indexed_count(conn)
            pending = conv.pending_count(conn)
            versions = conn.execute(
                "SELECT session_id, revision, status, chunk_count, max_tokens FROM conversation_versions"
                " ORDER BY session_id").fetchall()
        finally:
            conn.close()
        real_ids = {item["id"] for item in self.real if item["turns"]}
        indexed_real = {row["session_id"] for row in versions} & real_ids
        query = "2025 年 LPL 派去全球总决赛的四支战队是哪几个"
        result = self.recall(query, include_summary=False)
        # 两个真实会话都是 LPL 话题：allowed 的正确答案是 found 或 ambiguous（候选里要有 99c1e6）。
        reached = (result["status"] == "found" and (result.get("session") or {}).get("id") == "99c1e6") \
            or (result["status"] == "ambiguous"
                and "99c1e6" in {item.get("session_id") for item in result.get("candidates") or []})
        evidence = result.get("evidence") or []
        if result["status"] == "ambiguous":
            for item in result.get("candidates") or []:
                if item.get("session_id") == "99c1e6":
                    evidence = item.get("evidence") or []
                    break
        ok = (indexed >= self.expected_indexed and pending == 0
              and len(indexed_real) == len(real_ids) and reached and evidence)
        detail = (f"lifespan 自动入队并建索引 {indexed}/{self.expected_indexed} 个会话，pending={pending}；"
                  f"真实会话全部建索引：{sorted(indexed_real)}；"
                  f"HTTP recall『{query}』→ status={result['status']} "
                  f"session={(result.get('session') or {}).get('id')}，"
                  f"99c1e6 的原文证据={len(evidence)} 轮"
                  + ("（ambiguous 分支：候选里认定正确会话）" if result["status"] == "ambiguous" else ""))
        self.report.add(1, "旧会话自动建索引 + 新会话找回以前讨论", ok, detail,
                        "python -X utf8 scripts/acceptance_conversations.py")
        self.report.metrics["standard1"] = {"indexed": indexed, "vector_count": conv.vector_count(),
                                     "real_indexed": sorted(indexed_real),
                                     "chunks": {row["session_id"]: row["chunk_count"] for row in versions}}

    def standard_2(self, ambiguous_info: dict) -> None:
        rate = self.report.metrics["positive_rate"]
        worst = self.report.metrics["worst_positive"]
        ok = rate >= POSITIVE_RATE_FLOOR
        detail = (f"{self.report.metrics['positive_total']} 个重新措辞的正样本：严格命中 "
                  f"{self.report.metrics['positive_hits']}/{self.report.metrics['positive_total']} = "
                  f"{rate:.1%}（下限 {POSITIVE_RATE_FLOOR:.0%}），宽松（含 ambiguous 里出现正确会话）"
                  f"{self.report.metrics['positive_relaxed_hits']}/"
                  f"{self.report.metrics['positive_total']} = "
                  f"{self.report.metrics['positive_relaxed_rate']:.1%}；"
                  f"最低分样本：{worst[1] if worst else None} score={worst[0] if worst else None} "
                  f"实际 top1={worst[3] if worst else None}；"
                  f"近似重复会话对判定={ambiguous_info['status']} {ambiguous_info['ids']}")
        self.report.add(2, "同一意思换一种说法仍能找到正确会话", ok, detail,
                        "python -X utf8 scripts/acceptance_conversations.py（指标段）")

    def standard_3(self) -> None:
        """『用第二种』『按这个做』必须带上一轮原文上下文才能理解。"""
        session = next(item for item in self.synthetic if item["id"] == "refer01")
        records = conv.build_records(session)
        by_turn: dict[int, list[dict]] = {}
        for record in records:
            by_turn.setdefault(record["turn_index"], []).append(record)
        carried = {}
        # 修复 D1 后的正确口径：每个片段只带"紧邻上一轮"，所以
        # 第 1 轮（用户说"用第二种"）应看到第 0 轮的"第一种"；
        # 第 2 轮（用户说"按这个做"）应看到第 1 轮的"第二种"，并且不得再带上第 0 轮的"第一种"。
        for index, marker, referent, forbidden in ((1, "用第二种", "第一种", None),
                                                  (2, "按这个做", "第二种", "第一种")):
            chunks = by_turn.get(index, [])
            text = "\n".join(item["text"] for item in chunks)
            carried[index] = {
                "marker": marker,
                "chunks": len(chunks),
                "has_marker": any(marker in item["text"] for item in chunks),
                "has_prev_user": "上一轮用户问题：" in text,
                "has_referent": referent in text,
                "accumulated_older": bool(forbidden and forbidden in text),
                "prev_user_lines": text.count("上一轮用户问题："),
            }
        result = self.search("备份策略最后按哪个方案执行")
        top = (result["candidates"] or [{}])[0]
        follow = self.search("增量备份是怎么安排的")
        ok = (all(carried[index]["has_marker"] and carried[index]["has_prev_user"]
                  and carried[index]["has_referent"] and not carried[index]["accumulated_older"]
                  and carried[index]["prev_user_lines"] == carried[index]["chunks"]
                  for index in (1, 2))
              and result["status"] == "found" and top.get("session_id") == "refer01"
              and any(item["session_id"] == "refer01" for item in follow["candidates"] or []))
        detail = (f"refer01：第 2 轮片段含『用第二种』={carried[1]['has_marker']}、"
                  f"带上轮原文（含『第一种』）={carried[1]['has_referent']}；"
                  f"第 3 轮片段含『按这个做』={carried[2]['has_marker']}、"
                  f"带上轮原文（含『第二种』）={carried[2]['has_referent']}、"
                  f"误带更早轮次的『第一种』={carried[2]['accumulated_older']}；"
                  f"每轮『上一轮用户问题』行数={[carried[1]['prev_user_lines'], carried[2]['prev_user_lines']]}"
                  f"（应与片段数 {[carried[1]['chunks'], carried[2]['chunks']]} 相等）；"
                  f"『备份策略最后按哪个方案执行』→ {result['status']} top1={top.get('session_id')}"
                  f" score={top.get('score')}；『增量备份是怎么安排的』候选含 refer01="
                  f"{any(item['session_id'] == 'refer01' for item in follow['candidates'] or [])}")
        self.report.add(3, "『可以/用第二种/按这个做』结合上一轮上下文能正确理解", ok, detail,
                        "python -X utf8 scripts/acceptance_conversations.py（标准 3 段）")
        self.report.metrics["standard3"] = carried

    def standard_4(self) -> None:
        """用户后来推翻旧决定：证据要落在最终确认的那一轮，摘要区分 confirmed/superseded。"""
        result = self.recall("缓存层最后确定用哪个方案", include_summary=self.use_real_summary)
        evidence = result.get("evidence") or []
        turns = [item.get("turn_id") for item in evidence]
        final_text = " ".join(str(item.get("final_answer") or "") for item in evidence)
        both_present = "进程内 LRU" in final_text and "SQLite" in final_text
        final_turn = next((item for item in evidence if item["turn_id"] == "cache01:1"), {})
        final_is_latest = "SQLite" in str(final_turn.get("final_answer") or "")
        summary = result.get("summary") or {}
        confirmed = " ".join(item["text"] for item in summary.get("confirmed_decisions") or [])
        superseded = " ".join(item["text"] for item in summary.get("superseded_decisions") or [])
        if self.use_real_summary:
            summary_ok = ("SQLite" in confirmed) and ("LRU" in superseded)
            summary_note = (f"真实模型摘要：confirmed含SQLite={'SQLite' in confirmed}，"
                            f"superseded含LRU={'LRU' in superseded}，status={result.get('summary_status')}")
        else:
            summary_ok = True
            summary_note = "摘要未用真模型（--stub-summary），只验证证据落在最终轮"
        ok = bool(evidence) and final_is_latest and both_present and summary_ok
        detail = (f"recall 证据轮次={turns}；最终轮 cache01:1 的答案是 SQLite 决定={final_is_latest}；"
                  f"新旧两种说法都在证据里={both_present}；{summary_note}")
        self.report.add(4, "后来推翻旧决定 → 回忆采用最终确认版本", ok, detail,
                        "python -X utf8 scripts/acceptance_conversations.py（标准 4 段）")
        self.report.metrics["standard4"] = {"evidence_turn_ids": turns, "answer_tail": final_text[-160:],
                                            "summary_status": result.get("summary_status"),
                                            "confirmed": confirmed[:300],
                                            "superseded": superseded[:300],
                                            "summary": summary}

    def standard_5(self) -> None:
        """首次生成、第二次复用、会话更新后按新版本重新生成。"""
        if not self.use_real_summary:
            self.report.add(5, "摘要生成/复用/失效重生成", None,
                            "本次以 --stub-summary 运行，未验证真实模型摘要链路")
            return
        # 标准 4 已经为 cache01 生成过一份摘要：先清掉，保证这里的"第一次"确实是首次生成。
        conn = rag_state.connect()
        try:
            conn.execute("DELETE FROM conversation_summaries WHERE session_id='cache01'")
            conn.commit()
        finally:
            conn.close()
        before_calls = len(self.llm_calls)
        first = self.recall("缓存层最后确定用哪个方案", include_summary=True,
                            api_settings=self.real_settings())
        second = self.recall("缓存层最后确定用哪个方案", include_summary=True,
                             api_settings=self.real_settings())
        rows_after_first = self._summary_rows("cache01")
        # 会话更新后重新生成
        data = session_store.SessionStore(self.sessions).read("cache01")
        data["turns"].append(turn("缓存失效策略怎么定？", "本地 SQLite 缓存加 TTL 和版本号双重失效。"))
        session_store.SessionStore(self.sessions).write(data)
        status, sync = self.post("/v1/conversations/sync", {"session_id": "cache01", "all": False})
        assert status == 200, sync
        self.wait_indexed()
        third = self.recall("缓存层最后确定用哪个方案", include_summary=True,
                            api_settings=self.real_settings())
        rows_after_third = self._summary_rows("cache01")
        calls = len(self.llm_calls) - before_calls
        ok = (first.get("summary_status") == "generated" and second.get("summary_status") == "cached"
              and second.get("summary") == first.get("summary")
              and third.get("summary_status") == "generated"
              and third.get("summary_source_revision") != first.get("summary_source_revision")
              and len(rows_after_first) == 1 and len(rows_after_third) == 2
              and calls == 2)
        detail = (f"第一次={first.get('summary_status')}（source_revision="
                  f"{(first.get('summary_source_revision') or '')[:8]}，模型请求 {calls} 次）；"
                  f"第二次={second.get('summary_status')} 与第一次 summary 相同="
                  f"{second.get('summary') == first.get('summary')}；"
                  f"追加一轮并重新同步后第三次={third.get('summary_status')}"
                  f"（新 source_revision={(third.get('summary_source_revision') or '')[:8]}，"
                  f"与第一次不同={third.get('summary_source_revision') != first.get('summary_source_revision')}）；"
                  f"缓存行数 1→{len(rows_after_first)}→{len(rows_after_third)}；模型累计请求={calls}")
        self.report.add(5, "首次生成摘要 / 第二次复用 / 会话更新后重新生成", ok, detail,
                        "python -X utf8 scripts/acceptance_conversations.py（标准 5 段，真实 API）")
        self.report.metrics["standard5"] = {"first": first.get("summary_status"),
                                     "second": second.get("summary_status"),
                                     "third": third.get("summary_status"), "model_calls": calls}

    def _summary_rows(self, session_id: str) -> list[dict]:
        conn = rag_state.connect()
        try:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM conversation_summaries WHERE session_id=?", (session_id,)).fetchall()]
        finally:
            conn.close()

    def standard_6(self) -> None:
        """长会话中段信息可检索，且没有任何片段被模型截断。"""
        version = self._detail("deploy01")["index"]
        mid = self.search("灰度发布第一批放多少流量进去")
        rollback_query = "自动回滚的触发条件是怎么设的"
        rollback = self.search(rollback_query)
        session = next(item for item in self.synthetic if item["id"] == "deploy01")
        records = conv.build_records(session)
        over = [item for item in records if conv._tokens(item["text"]) > conv.CHUNK_LIMIT_TOKENS]
        # 中段轮次位置：第 8 轮（index 7）和第 12 轮（index 11）
        middle_hit = [hit for hit in (mid["candidates"] or [{}])[0].get("hits") or []
                      if hit["turn_index"] in (7, 8)]
        ok = (version["max_tokens"] <= conv.CHUNK_LIMIT_TOKENS and not over
              and mid["status"] == "found" and (mid["candidates"] or [{}])[0]["session_id"] == "deploy01"
              and any(hit["turn_index"] in (7, 8) for hit in middle_hit)
              and rollback["status"] == "found"
              and (rollback["candidates"] or [{}])[0]["session_id"] == "deploy01")
        # 回滚查询的失败根因量化：全片段 vs 只有本轮正文的余弦差
        rollback_record = next((item for item in records if item["turn_index"] == 11), None)
        dilution = {}
        if rollback_record is not None:
            body_only = "\n".join(line for line in rollback_record["text"].splitlines()
                                  if line.startswith(("本轮用户问题：", "本轮最终回答：")))
            full_score = self.cos(rollback_query, rollback_record["text"])
            body_score = self.cos(rollback_query, body_only)
            composition = compose_tokens(rollback_record["text"])
            dilution = {"full": round(full_score, 4), "body_only": round(body_score, 4),
                        "delta": round(body_score - full_score, 4),
                        "context_share": composition["context_share"],
                        "context_tokens": composition["context"],
                        "total_tokens": composition["total"]}
        detail = (f"deploy01 共 {len(session['turns'])} 轮 / {len(records)} 个片段，"
                  f"版本记录 max_tokens={version['max_tokens']}（上限 {conv.CHUNK_LIMIT_TOKENS}），"
                  f"超限片段={len(over)}；"
                  f"中段第 8 轮查询 → {mid['status']} top1="
                  f"{(mid['candidates'] or [{}])[0].get('session_id')} "
                  f"score={(mid['candidates'] or [{}])[0].get('score')} "
                  f"命中轮次={sorted({hit['turn_index'] for hit in middle_hit})}；"
                  f"中段第 12 轮回滚查询 → {rollback['status']} top1="
                  f"{(rollback['candidates'] or [{}])[0].get('session_id')}"
                  + (f"；同一条片段：全片段余弦={dilution['full']}，只留本轮正文="
                     f"{dilution['body_only']}（差 {dilution['delta']:+.4f}），"
                     f"上下文占 {dilution['context_share']:.0%}"
                     f"（{dilution['context_tokens']}/{dilution['total_tokens']} token）"
                     if dilution else ""))
        self.report.add(6, "长会话中段信息可找到、不受模型截断影响", ok, detail,
                        "python -X utf8 scripts/acceptance_conversations.py（标准 6 段）")
        self.report.metrics["standard6"] = {"turns": len(session["turns"]), "chunks": len(records),
                                            "max_tokens": version["max_tokens"],
                                            "over_limit": len(over), "dilution": dilution}
        if dilution and not ok and rollback["status"] != "found":
            self.report.add_defect(
                "D1b", "中",
                "上下文前缀仍把合法改写查询的相似度压低到门槛之下",
                f"查询『{rollback_query}』：同一条片段全片段余弦={dilution['full']}，"
                f"只留本轮 user/answer 时={dilution['body_only']}（{dilution['delta']:+.4f}），"
                f"该片段上下文占 {dilution['context_share']:.0%}"
                f"（{dilution['context_tokens']}/{dilution['total_tokens']} token）；"
                f"正文本身能过 0.86，加前缀后低于门槛 → not_found。",
                "python -X utf8 scripts/acceptance_conversations.py（标准 6 段）")

    def _detail(self, session_id: str) -> dict:
        status, body = self.get(f"/v1/conversations/{session_id}")
        assert status == 200, body
        return body

    def standard_7(self, ambiguous_info: dict) -> None:
        result = self.recall(ambiguous_info["query"], include_summary=False)
        candidates = result.get("candidates") or []
        ids = {item["session_id"] for item in candidates}
        scores = {item["session_id"]: item["score"] for item in candidates}
        expected = set(ambiguous_info["expected"])
        # HTTP 的 ambiguous 响应里候选带的原文片段字段叫 evidence（不是 hits）：
        # 每个候选都要有自己的整轮原文，便于主模型按原文判断。
        with_evidence = all(item.get("evidence") for item in candidates)
        # 两个近似重复会话都必须进候选且分差 < margin；同批可能还有第三个过门槛的会话
        # （见报告：填充文本多的长会话会充当"分数汇"），这不算缺陷，但要如实记录。
        ok = (result["status"] == "ambiguous" and expected <= ids
              and all(item.get("title") for item in candidates)
              and with_evidence
              and result.get("summary") is None
              and abs(scores.get(sorted(expected)[0], 1.0)
                      - scores.get(sorted(expected)[1], 0.0)) < conv.CONVERSATION_AMBIGUOUS_MARGIN * 3)
        detail = (f"模块层 status={ambiguous_info['status']} ids={ambiguous_info['ids']} "
                  f"scores={ambiguous_info['scores']}；"
                  f"HTTP recall status={result['status']}，候选="
                  f"{[(item.get('session_id'), item.get('title')) for item in candidates]}，"
                  f"每个候选带原文 evidence="
                  f"{[(item.get('session_id'), len(item.get('evidence') or [])) for item in candidates]}，"
                  f"未生成摘要={result.get('summary') is None}；"
                  f"额外候选={sorted(ids - expected) or '无'}")
        self.report.add(7, "多个相似会话返回 ambiguous、不混成一份历史", ok, detail,
                        "python -X utf8 scripts/acceptance_conversations.py（标准 7 段）")

    def standard_8(self) -> None:
        wrong = []
        lines = []
        false_compositions = []
        for query, why in UNRELATED:
            result = self.recall(query, include_summary=True, api_settings=self.real_settings())
            if result["status"] != "not_found" or result.get("candidates"):
                wrong.append(query)
                module = self.search(query)
                top = (module["candidates"] or [{}])[0]
                hit = (top.get("hits") or [{}])[0]
                composition = compose_tokens(hit.get("text") or "")
                false_compositions.append({
                    "query": query, "status": result["status"],
                    "session": top.get("session_id"), "score": top.get("score"),
                    "turn_id": hit.get("turn_id"), "context_share": composition["context_share"],
                    "context_tokens": composition["context"], "body_tokens": composition["body"],
                    "total_tokens": composition["total"],
                    "excerpt": " ".join(str(hit.get("text") or "").split())[:120]})
            lines.append(f"{query}→{result['status']}")
        fp_rate = len(wrong) / len(UNRELATED)
        # 判定按事先写死的上限；detail 里同时给出严格口径（几条不是 not_found）。
        ok = fp_rate <= UNRELATED_FP_CEIL
        detail = (f"{len(UNRELATED)} 个无关问题 HTTP recall：误命中 {len(wrong)}/{len(UNRELATED)} = "
                  f"{fp_rate:.1%}（上限 {UNRELATED_FP_CEIL:.0%}）；非 not_found 的：{wrong or '无'}；"
                  f"{'；'.join(lines)}")
        self.report.add(8, "无关问题返回 not_found、不强塞历史内容", ok, detail,
                        "python -X utf8 scripts/acceptance_conversations.py（标准 8 段）")
        self.report.metrics["standard8_false_hits"] = false_compositions
        for item in false_compositions:
            print(f"    FP 片段构成：{item['query']} → {item['session']} {item['turn_id']} "
                  f"score={item['score']} 上下文={item['context_tokens']}/"
                  f"{item['total_tokens']}（{item['context_share']:.0%}）")
        if wrong:
            soup = [item for item in false_compositions if item["context_share"] >= 0.6]
            hit_text = "；".join(
                "{session} {turn} score={score} 上下文占 {share:.0%}".format(
                    session=item["session"], turn=item["turn_id"], score=item["score"],
                    share=item["context_share"])
                for item in false_compositions)
            self.report.add_defect(
                "D2", "高" if fp_rate > UNRELATED_FP_CEIL else "低",
                f"出厂阈值 0.86 下无关问题误命中 {len(wrong)}/{len(UNRELATED)} = {fp_rate:.1%}"
                f"（{'、'.join(item['status'] for item in false_compositions)}）",
                f"误命中：{wrong}；命中片段：{hit_text}；"
                f"其中 {len(soup)} 条命中的片段上下文占 60% 以上。"
                f"当前正样本严格命中 {self.report.metrics.get('positive_hits')}/"
                f"{self.report.metrics.get('positive_total')}、"
                f"宽松 {self.report.metrics.get('positive_relaxed_hits')}/"
                f"{self.report.metrics.get('positive_total')}；"
                f"最低正确正样本分 {self.report.metrics.get('best_positive_min')}、"
                f"最高无关分 {self.report.metrics.get('unrelated_max')}；"
                f"阈值扫描见 metrics.threshold_sweep。",
                "python -X utf8 scripts/acceptance_conversations.py（指标段/标准 8 段）")

    def standard_9(self) -> None:
        """幂等：重复通知、失败重试、服务中断恢复都不丢轮次、不重复入库。"""
        store = session_store.SessionStore(self.sessions)
        before_vectors = conv.vector_count()
        # 1) 重复通知
        for _ in range(3):
            conv.sync_session("ingest01", sessions_dir=self.sessions)
        self.wait_indexed()
        after_repeat = conv.vector_count()
        conn = rag_state.connect()
        try:
            versions = conn.execute("SELECT COUNT(*) AS n FROM conversation_versions"
                                    " WHERE session_id='ingest01'").fetchone()["n"]
            jobs = conn.execute("SELECT COUNT(*) AS n FROM conversation_jobs j JOIN conversation_versions v"
                                " ON j.version_id=v.id WHERE v.session_id='ingest01'").fetchone()["n"]
        finally:
            conn.close()
        # 2) 保存失败后补存：写盘失败时轮次不落盘，重试后只落一次
        sess = store.read("ingest01")
        turns_before = len(sess["turns"])
        raised = {}

        def boom(_self, _session):
            raised["hit"] = True
            raise OSError("模拟保存失败")

        with patch.object(session_store.SessionStore, "write", boom):
            try:
                store.save_turn(sess, "保存失败时的新问题", "不该落盘的答案")
            except OSError:
                pass
        disk = store.read("ingest01")
        after_fail = len(disk["turns"])
        # 内存里那轮还在（save_turn 先更新内存），补存成功后磁盘只多一轮
        store.write(sess)
        recovered = store.read("ingest01")
        conv.sync_session("ingest01", sessions_dir=self.sessions)
        self.wait_indexed()
        after_resave = conv.vector_count()
        conn = rag_state.connect()
        try:
            active = rag_state.active_conversation_version(conn, "ingest01")
        finally:
            conn.close()
        revision_ok = active["revision"] == session_store.revision_of(recovered)
        # 3) 服务中断：抢到任务但没跑完 → recover 重新入队，重跑不产生重复向量
        conn = rag_state.connect()
        try:
            conn.execute("UPDATE conversation_jobs SET status='processing' WHERE version_id=?",
                         (active["id"],))
            conn.commit()
            recovered_jobs = rag_state.recover_interrupted_conversations(conn, time.time())
        finally:
            conn.close()
        self.wait_indexed()
        after_recover = conv.vector_count()
        # 补存会新增一轮 → 新版本 → 向量数合理增加；"不重复入库"的口径是：
        # 再重复通知若干次后向量数不再变、版本/任务行仍是 1。
        for _ in range(3):
            conv.sync_session("ingest01", sessions_dir=self.sessions)
        self.wait_indexed()
        after_more_notices = conv.vector_count()
        ok = (raised.get("hit") and after_fail == turns_before and len(recovered["turns"]) == turns_before + 1
              and after_repeat == before_vectors and versions == 1 and jobs == 1
              and after_resave > after_repeat and revision_ok
              and recovered_jobs >= 1 and after_recover == after_resave
              and after_more_notices == after_recover)
        detail = (f"重复通知 3 次：版本行={versions} 任务行={jobs} 向量数 {before_vectors}→{after_repeat}"
                  f"（应相等）；模拟写盘失败：磁盘轮次 {turns_before}→{after_fail}（应不变），"
                  f"补存后 {len(recovered['turns'])} 轮（应 +1），索引 revision 与磁盘一致={revision_ok}；"
                  f"补存重新入库后向量数 {after_resave}（> 之前，因为多了一轮）；"
                  f"再重复通知 3 次后向量数={after_more_notices}（应不变）；"
                  f"伪造 processing 中断后 recover 重新入队 {recovered_jobs} 个任务，"
                  f"重跑后向量数 {after_recover}（应不变）")
        self.report.add(9, "服务中断/保存失败补存/重复通知都不丢轮次、不重复入库", ok, detail,
                        "python -X utf8 scripts/acceptance_conversations.py（标准 9 段）")
        self.report.metrics["standard9"] = {"before": before_vectors, "after_repeat": after_repeat,
                                            "after_resave": after_resave,
                                            "after_more_notices": after_more_notices,
                                            "after_recover": after_recover,
                                            "versions": versions, "jobs": jobs}

    def standard_10(self) -> None:
        # (a) 摘要 API 不可达（真实代码路径，BASE_URL 指向死端口）。
        # 用还没有摘要缓存的真实会话 be0fa5 定向回忆，否则会先命中缓存、走不到失败路径。
        dead = {"API_KEY": CANARY_KEY, "MODEL": "stub-model", "BASE_URL": "http://127.0.0.1:1/v1"}
        broken = self.recall("多头注意力机制的作用", session_id="be0fa5", include_summary=True,
                             api_settings=dead)
        # (b) 摘要层直接抛异常（patch 整个函数，绕过缓存）
        with patch.object(conversation_summary, "use_or_create_summary",
                          side_effect=RuntimeError("模拟摘要层崩溃")):
            crashed = self.recall("缓存层最后确定用哪个方案", include_summary=True,
                                  api_settings=self.real_settings())
        # (c) 显式关闭摘要
        skipped = self.recall("缓存层最后确定用哪个方案", include_summary=False)
        ok = bool(broken["status"] == "found" and broken.get("summary") is None
                  and broken.get("evidence") and broken.get("summary_status") == "failed"
                  and crashed["status"] == "found" and crashed.get("summary") is None
                  and crashed.get("evidence")
                  and skipped["status"] == "found" and skipped.get("summary_status") == "skipped"
                  and skipped.get("evidence"))
        detail = (f"摘要 API 不可达（be0fa5，无缓存）：status={broken['status']} summary_status="
                  f"{broken.get('summary_status')} summary={broken.get('summary')} "
                  f"evidence={len(broken.get('evidence') or [])} 轮；"
                  f"摘要层抛异常：status={crashed['status']} summary_status="
                  f"{crashed.get('summary_status')} evidence={len(crashed.get('evidence') or [])} 轮；"
                  f"include_summary=False：summary_status={skipped.get('summary_status')} "
                  f"evidence={len(skipped.get('evidence') or [])} 轮")
        self.report.add(10, "摘要 API 失败时仍返回可用原文（recall 不整体失败）", ok, detail,
                        "python -X utf8 scripts/acceptance_conversations.py（标准 10 段）")

    def standard_11(self, skip: bool) -> None:
        if skip:
            self.report.add(11, "同一 session 连续对话/文件 RAG/API 切换回归", None,
                            "本次以 --skip-regression 运行，未执行三个回归套件")
            return
        modules = ["tests.test_cli", "tests.test_rag_service", "tests.test_conversation_service"]
        print("\n  运行回归套件：" + " ".join(modules))
        started = time.monotonic()
        env = {key: value for key, value in os.environ.items() if key != "GAGENT_SESSIONS_DIR"}
        proc = subprocess.run([sys.executable, "-X", "utf8", "-m", "unittest", *modules],
                              cwd=str(PROJECT), capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=env)
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        summary = "\n".join(tail[-12:])
        match = re.search(r"Ran (\d+) tests?.*?\n(OK|FAILED[^\n]*)", proc.stderr or "", re.S)
        ran = match.group(1) if match else "?"
        verdict = match.group(2) if match else "?"
        ok = proc.returncode == 0
        detail = (f"{'；'.join(modules)} → exit={proc.returncode}，Ran {ran} tests，{verdict}，"
                  f"用时 {time.monotonic() - started:.1f}s")
        self.report.add(11, "同一 session 连续对话/文件 RAG/API 切换回归", ok, detail,
                        "& \"C:\\Python\\envs\\agent\\python.exe\" -X utf8 -m unittest "
                        "tests.test_cli tests.test_rag_service tests.test_conversation_service")
        self.report.metrics["standard11"] = {"returncode": proc.returncode, "ran": ran, "verdict": verdict,
                                      "tail": summary}

    # ================================================================ 易错点复核

    def check_exclusion_and_target(self) -> None:
        print("\n" + "=" * 78)
        print("易错点复核")
        print("=" * 78)
        checks = {}
        # 当前会话默认排除
        query = "缓存层最后确定用哪个方案"
        without = self.search(query, min_score=0.50)
        excluded = self.search(query, exclude_sessions=["cache01"])
        checks["exclude_current"] = {
            "without_exclude": [[item["session_id"], item["score"]]
                                for item in without["candidates"] or []],
            "with_exclude": [[item["session_id"], item["score"]]
                             for item in excluded["candidates"] or []],
            "status_with_exclude": excluded["status"]}
        print(f"  『{query}』不排除：{[ (i['session_id'], i['score']) for i in without['candidates'] or [] ]}")
        print(f"  『{query}』排除当前会话 cache01 后 → status={excluded['status']} "
              f"候选={[(i['session_id'], i['score']) for i in excluded['candidates'] or []]}"
              f"（正确来源就是 cache01；若此时给别的会话判 found，就是把无关历史当答案）")
        http_excluded = self.recall(query, exclude_session="cache01", include_summary=False)
        print(f"  HTTP recall exclude_session=cache01 → status={http_excluded['status']} "
              f"session={http_excluded.get('session')}")
        # 排除当前会话后落到别的会话、且被判 found：这是排除路径上的误命中
        wrong_found = ([item["session_id"] for item in excluded["candidates"] or []]
                       if excluded["status"] == "found" else [])
        if wrong_found:
            scores = {item["session_id"]: item["score"] for item in excluded["candidates"] or []}
            without_scores = [item["score"] for item in without["candidates"] or []]
            lead = round(without_scores[0] - without_scores[1], 6) if len(without_scores) >= 2 else None
            self.report.add_defect(
                "D4", "中",
                "排除当前会话后，同一个查询会对另一个沾边会话判定 found",
                f"查询『{query}』（正确来源就是被排除的 cache01）→ status=found，"
                f"候选={[(sid, scores[sid]) for sid in wrong_found]}；"
                f"不排除时候选与分数={[(i['session_id'], i['score']) for i in without['candidates'] or []]}，"
                f"即正确会话与两个无关会话都在 0.86 以上，第一、第二候选只差 {lead}"
                f"（margin={conv.CONVERSATION_AMBIGUOUS_MARGIN}）——排除当前会话是常态路径，"
                f"此时调用方拿到的是 found + 备份方案的历史，而不是 not_found。",
                "python -X utf8 scripts/acceptance_conversations.py（易错点复核段）")
        # 显式指定 session_id：即使它不是最佳候选，也以它为准
        explicit = self.recall("2025 年 LPL 全球总决赛的队伍", session_id="be0fa5",
                               include_summary=False)
        print(f"  HTTP recall session_id=be0fa5（与查询主题无关）→ status={explicit['status']} "
              f"session={explicit.get('session', {}).get('id')} "
              f"evidence轮次={[item['turn_id'] for item in explicit.get('evidence') or []]}")
        # 定向回忆时 exclude_session 不该把目标排掉
        targeted = self.recall("缓存层最后确定用哪个方案", session_id="cache01",
                               exclude_session="cache01", include_summary=False)
        checks["target_wins"] = {"status": targeted["status"],
                                 "session": (targeted.get("session") or {}).get("id"),
                                 "evidence": len(targeted.get("evidence") or [])}
        print(f"  HTTP recall session_id=cache01 且 exclude_session=cache01 → "
              f"status={targeted['status']} session={(targeted.get('session') or {}).get('id')} "
              f"evidence={len(targeted.get('evidence') or [])} 轮（定向应以指定会话为准）")
        checks["exclude_http"] = {"status": http_excluded["status"],
                                  "session": http_excluded.get("session")}
        checks["explicit_target"] = {"status": explicit["status"],
                                     "session": (explicit.get("session") or {}).get("id"),
                                     "evidence": len(explicit.get("evidence") or [])}
        self.report.metrics["exclusion"] = checks

    def check_summary_traceability(self) -> None:
        """摘要里的 turn_ids 必须落在返回的 evidence 轮次里。"""
        if not self.use_real_summary:
            print("  摘要 turn_ids 溯源：本次未用真模型，跳过")
            self.report.metrics["traceability"] = {"skipped": True}
            return
        result = self.recall("缓存层最后确定用哪个方案", include_summary=True,
                             api_settings=self.real_settings())
        summary = result.get("summary") or {}
        evidence_ids = {item["turn_id"] for item in result.get("evidence") or []}
        cited = []
        for key in conversation_summary.SUMMARY_KEYS:
            value = summary.get(key)
            if isinstance(value, list):
                for item in value:
                    cited.extend(item.get("turn_ids") or [])
        cited = [item for item in cited if item]
        missing = sorted({item for item in cited if item not in evidence_ids})
        ok = bool(cited) and not missing
        print(f"  摘要 turn_ids 溯源：引用={sorted(set(cited))}，evidence 轮次={sorted(evidence_ids)}，"
              f"落在 evidence 之外的={missing or '无'} → {'通过' if ok else '不通过'}")
        self.report.metrics["traceability"] = {"cited": sorted(set(cited)),
                                        "evidence": sorted(evidence_ids), "missing": missing, "ok": ok}

    def check_no_credential_leak(self) -> None:
        """凭据不能进向量/缓存/日志/事件，也不能回显在响应里。"""
        dead = {"API_KEY": CANARY_KEY, "MODEL": "stub-model", "BASE_URL": "http://127.0.0.1:1/v1"}
        response = self.recall("缓存层最后确定用哪个方案", include_summary=True, api_settings=dead)
        blob_hits = []
        for dirpath, _dirs, files in os.walk(self.data):
            for name in files:
                path = pathlib.Path(dirpath) / name
                try:
                    payload = path.read_bytes()
                except OSError:
                    continue
                if CANARY_KEY.encode("utf-8") in payload:
                    blob_hits.append(str(path.relative_to(self.data)))
        conn = rag_state.connect()
        try:
            summary_hits = conn.execute(
                "SELECT COUNT(*) AS n FROM conversation_summaries WHERE summary_json LIKE ?",
                (f"%{CANARY_KEY}%",)).fetchone()["n"]
            event_rows = [dict(row) for row in conn.execute("SELECT * FROM events").fetchall()]
            event_hits = [row for row in event_rows if CANARY_KEY in json.dumps(row, ensure_ascii=False)]
        finally:
            conn.close()
        response_text = json.dumps(response, ensure_ascii=False)
        ok = (not blob_hits and not summary_hits and not event_hits and CANARY_KEY not in response_text)
        print(f"  凭据泄漏扫描：临时数据目录文件命中={blob_hits or '无'}；"
              f"摘要表命中={summary_hits}；事件表命中={len(event_hits)}；"
              f"响应体回显={'有' if CANARY_KEY in response_text else '无'}")
        self.report.metrics["credential_leak"] = {"files": blob_hits, "summary_rows": summary_hits,
                                           "events": len(event_hits),
                                           "echo": CANARY_KEY in response_text, "ok": ok}

    def check_active_version_only(self) -> None:
        """改旧轮次后旧向量必须真的失效：旧话题在新版本里一点不剩，检索要落空。"""
        store = session_store.SessionStore(self.sessions)
        session = store.read("docker01")
        old_revision = session_store.revision_of(session)
        old_query = self.search("部署环境用 Docker Swarm 还是别的编排")
        changed = dict(session)
        # 整个旧话题（部署 / Docker Swarm）被换掉，换成别的会话都没覆盖的消息队列主题，
        # 这样"旧内容检索不到"没有语义歧义，不会被"新正文也提到 Swarm"混淆。
        changed["turns"] = [turn("消息队列最后用 RabbitMQ 还是 Kafka？",
                                 "消息队列最后定为 RabbitMQ，先不上 Kafka。")]
        store.write(changed)
        conv.sync_session("docker01", sessions_dir=self.sessions)
        self.wait_indexed()
        new_revision = session_store.revision_of(changed)
        post = self.search("部署环境用 Docker Swarm 还是别的编排")
        post_new = self.search("消息队列最后用 RabbitMQ 了吗")
        gone = conv.delete_version_vectors("docker01", old_revision)
        conn = rag_state.connect()
        try:
            active = rag_state.active_conversation_version(conn, "docker01")
        finally:
            conn.close()
        old_ids_absent = conv._ids_of("docker01", old_revision) == []
        # 旧话题查询可能仍然命中别的会话（例如 deploy01 也是部署话题）——那不是残留向量；
        # 硬口径只看 docker01 自己还能不能过门槛。
        docker_above = [item for item in (post["candidates"] or [])
                        if item["session_id"] == "docker01"
                        and item["score"] >= conv.CONVERSATION_MIN_SCORE]
        other_above = [item["session_id"] for item in (post["candidates"] or [])
                       if item["session_id"] != "docker01"
                       and item["score"] >= conv.CONVERSATION_MIN_SCORE]
        ok = (old_query["status"] == "found"
              and (old_query["candidates"] or [{}])[0]["session_id"] == "docker01"
              and old_ids_absent and gone == 0
              and active["revision"] == new_revision
              and post_new["status"] == "found"
              and (post_new["candidates"] or [{}])[0]["session_id"] == "docker01")
        # 旧向量确实没了，但会话标题没变、而标题也进向量：残留相似度来自标题，不是旧向量。
        title_line = f"会话标题：{changed.get('title') or ''}"
        new_body = changed["turns"][0]["final_answer"]
        residual = {
            "docker01_score_after_edit": round(float((docker_above or [{}])[0].get("score") or 0), 6)
            if docker_above else None,
            "title_only_cos": round(self.cos("部署环境用 Docker Swarm 还是别的编排", title_line), 4),
            "new_body_only_cos": round(self.cos("部署环境用 Docker Swarm 还是别的编排", new_body), 4),
        }
        print(f"  有效版本限定：改前查询→{old_query['status']} "
              f"top1={(old_query['candidates'] or [{}])[0].get('session_id')}；"
              f"改后旧向量残留={conv._ids_of('docker01', old_revision)}（应空）"
              f" delete_version_vectors 返回={gone}（应 0）；"
              f"改后旧话题查询→{post['status']}，docker01 过门槛="
              f"{[item['score'] for item in docker_above] or '无'}，"
              f"别的会话过门槛={other_above or '无'}；新话题查询→{post_new['status']} top1="
              f"{(post_new['candidates'] or [{}])[0].get('session_id')}；"
              f"active revision 已换={active['revision'] == new_revision}；"
              f"残留分数只来自标题：标题单独 cos={residual['title_only_cos']}，"
              f"新正文单独 cos={residual['new_body_only_cos']}")
        self.report.metrics["active_only"] = {"old_ids": conv._ids_of("docker01", old_revision),
                                              "old_query_status": old_query["status"],
                                              "new_query_status": post_new["status"],
                                              "docker_above": [item["score"] for item in docker_above],
                                              "residual": residual,
                                              "ok": ok}

    def check_sessions_dir_isolation(self) -> None:
        """patch session_store.SESSIONS_DIR 是否真能隔离 conversations 的默认会话目录。"""
        probe = (
            "import tempfile, pathlib\n"
            "from unittest.mock import patch\n"
            "import session_store\n"
            "from rag import conversations as conv, service\n"
            "tmp = pathlib.Path(tempfile.gettempdir()) / 'gagent_iso_probe'\n"
            "with patch.object(session_store, 'SESSIONS_DIR', tmp / 'sessions'):\n"
            "    print('service', service._sessions_dir())\n"
            "    print('conv', conv._sessions_dir())\n"
            "    print('store', session_store.SessionStore().root)\n"
        )
        env = {key: value for key, value in os.environ.items() if key != "GAGENT_SESSIONS_DIR"}
        proc = subprocess.run([sys.executable, "-X", "utf8", "-c", probe], cwd=str(PROJECT),
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              env=env)
        rows = {}
        for line in (proc.stdout or "").splitlines():
            parts = line.split(" ", 1)
            if len(parts) == 2:
                rows[parts[0]] = parts[1].strip()
        service_dir = rows.get("service", "")
        conv_dir = rows.get("conv", "")
        isolated = bool(service_dir and conv_dir == service_dir
                        and "gagent_iso_probe" in conv_dir)
        print(f"  会话目录隔离：patch session_store.SESSIONS_DIR → service={service_dir}；"
              f"conversations={conv_dir}（应一致）")
        self.report.metrics["sessions_dir_isolation"] = {"service": service_dir, "conv": conv_dir,
                                                        "isolated": isolated}
        if not isolated:
            self.report.add_defect(
                "D3", "低",
                "patch session_store.SESSIONS_DIR 隔离不了 conversations 的默认会话目录",
                f"`from session_store import SESSIONS_DIR` 在 conversations.py 导入时绑定；"
                f"同一个进程里 patch 属性后：rag.service._sessions_dir()={service_dir}，"
                f"而 rag.conversations._sessions_dir()={conv_dir}；"
                f"rag.conversations 里所有省略 sessions_dir 的调用（sync_all()/sync_session(id)）"
                f"都会去读项目真实 sessions/。用 GAGENT_SESSIONS_DIR 环境变量（导入前设置）"
                f"可以完全隔离；本验收脚本因此对所有 conv.* 调用都显式传 sessions_dir。",
                "python -X utf8 scripts/acceptance_conversations.py（易错点段）")

    def check_tool_layer_exclusion(self) -> None:
        """真正的入口 agent_tools.recall_conversation：当前会话默认排除、指定当前会话被拒。"""
        import agent_tools
        calls: list[dict] = []

        def recorder(query, *, session_id=None, exclude_session=None, **kwargs):
            calls.append({"query": query, "session_id": session_id,
                          "exclude_session": exclude_session})
            return {"status": "not_found", "query": query, "candidates": [], "hits": []}

        with patch.object(agent_tools.rag_client, "recall_conversation", recorder), \
                agent_tools.conversation_scope("cache01"):
            default = agent_tools.recall_conversation.invoke(
                {"query": "缓存层最后确定用哪个方案", "session_id": None})
            other = agent_tools.recall_conversation.invoke(
                {"query": "备份策略最后按哪个方案执行", "session_id": "refer01"})
            refused = agent_tools.recall_conversation.invoke(
                {"query": "缓存层最后确定用哪个方案", "session_id": "cache01"})
        by_query = {(item["query"], item["session_id"]): item for item in calls}
        default_call = by_query.get(("缓存层最后确定用哪个方案", None), {})
        other_call = by_query.get(("备份策略最后按哪个方案执行", "refer01"), {})
        ok = (default_call.get("exclude_session") == "cache01"
              and other_call.get("exclude_session") is None
              and refused.get("status") == "current_session")
        print(f"  工具入口：未指定时 exclude_session={default_call.get('exclude_session')}"
              f"（当前会话 cache01，应排除）；显式指定 refer01 时 exclude_session="
              f"{other_call.get('exclude_session')}（应为 None）；指定当前会话 → "
              f"status={refused.get('status')}")
        self.report.metrics["tool_layer"] = {"default": default_call, "other": other_call,
                                             "refused_status": refused.get("status"),
                                             "call_count": len(calls), "ok": ok}

    def check_summary_stale(self) -> None:
        """摘要生成期间会话被更新：必须标 stale，note 里要写明覆盖的是哪个版本。"""
        if not self.use_real_summary:
            print("  摘要 stale 检查：本次未用真模型，跳过")
            self.report.metrics["stale"] = {"skipped": True}
            return
        store = session_store.SessionStore(self.sessions)
        before = store.read("refer01")
        source_revision = session_store.revision_of(before)
        state = {"mutated": False}
        current_factory = conversation_summary._default_llm_factory

        def mutating_factory(settings, timeout):
            if not state["mutated"]:
                state["mutated"] = True
                data = store.read("refer01")
                data["turns"].append(turn("生成期间又追加的一轮", "这一轮在摘要生成过程中写入。"))
                store.write(data)
            return current_factory(settings, timeout)

        self._patch(conversation_summary, "_default_llm_factory", mutating_factory)
        result = self.recall("备份策略最后按哪个方案执行", include_summary=True,
                             api_settings=self.real_settings())
        current_revision = session_store.revision_of(store.read("refer01"))
        note = str(result.get("summary_note") or "")
        ok = (state["mutated"] and result.get("summary_status") == "stale"
              and result.get("summary_stale") is True
              and result.get("summary_source_revision") == source_revision
              and result.get("summary_current_revision") == current_revision
              and source_revision[:12] in note and current_revision[:12] in note)
        print(f"  摘要 stale：status={result.get('summary_status')} stale_flag="
              f"{result.get('summary_stale')}；覆盖版本={str(result.get('summary_source_revision'))[:12]}"
              f"（生成开始时 {source_revision[:12]}）；当前版本="
              f"{str(result.get('summary_current_revision'))[:12]}（磁盘 {current_revision[:12]}）；"
              f"note 含两个版本号={source_revision[:12] in note and current_revision[:12] in note}")
        self.report.metrics["stale"] = {"status": result.get("summary_status"),
                                        "stale": result.get("summary_stale"), "note": note[:300],
                                        "ok": ok}

    def check_chunking(self) -> None:
        """标题 + 上一轮上下文不能挤掉本轮主体，也不能静默截断。"""
        long_user = "背景很长：" + ("这一段描述业务约束和验收要求。" * 60)
        long_answer = "结论开头。" + ("中段分析内容，包含具体参数与取舍。" * 120) + "最后一段的收尾结论。"
        session = {"id": "chunkcheck", "title": "很长的标题" + "标" * 40,
                   "created_at": "", "updated_at": "",
                   "turns": [turn("上一轮问题" + "问" * 80, "上一轮回答" + "答" * 200),
                             turn(long_user, long_answer)]}
        records = conv.build_records(session)
        this_turn = [item for item in records if item["turn_index"] == 1]
        normalize = lambda text: "".join(str(text).split())  # noqa: E731
        joined = normalize("".join(item["text"] for item in this_turn))
        # 每个句子在片段里都会另起一行（带角色标签），所以按"整句出现次数"核对静默丢内容：
        # user 那句应出现 60 次、回答中段那句应出现 120 次，一次都不能少。
        user_times = joined.count("这一段描述业务约束和验收要求")
        mid_times = joined.count("中段分析内容，包含具体参数与取舍")
        head_kept = "背景很长：" in joined and "结论开头。" in joined
        tail_kept = "最后一段的收尾结论。" in joined
        over = [item for item in records if conv._tokens(item["text"]) > conv.CHUNK_LIMIT_TOKENS]
        bodyless = [item for item in this_turn
                    if "本轮用户问题：" not in item["text"] and "本轮最终回答：" not in item["text"]]
        context_tokens = []
        for item in this_turn:
            context = sum(conv._tokens(line.split("：", 1)[1])
                          for line in item["text"].splitlines()
                          if line.startswith(("会话标题：", "上一轮用户问题：", "上一轮回答中的邻近内容：")))
            context_tokens.append(context)
        with_prev = all("上一轮用户问题：" in item["text"] for item in this_turn)
        body_chars = sum(len(item["text"]) for item in this_turn)
        context_budget = conv.CONTEXT_TOTAL_BUDGET_TOKENS
        ok = (user_times == 60 and mid_times == 120 and head_kept and tail_kept
              and not over and not bodyless and with_prev
              and max(context_tokens, default=0) <= (conv.TITLE_BUDGET_TOKENS + context_budget))
        print(f"  切块预算：本轮 {len(this_turn)} 个片段，字符 {body_chars}，超限片段={len(over)}；"
              f"长 user 整句出现 {user_times}/60 次，长回答中段整句出现 {mid_times}/120 次，"
              f"首尾句保留={head_kept and tail_kept}；空正文片段={len(bodyless)}；"
              f"每片都带上一轮上下文={with_prev}；单片上下文 token 最大={max(context_tokens, default=0)}"
              f"（预算 {conv.TITLE_BUDGET_TOKENS + context_budget}）")
        self.report.metrics["chunking"] = {"records": len(this_turn),
                                           "user_sentence_times": user_times,
                                           "answer_sentence_times": mid_times,
                                           "head_tail_kept": head_kept and tail_kept,
                                           "over_limit": len(over), "bodyless": len(bodyless),
                                           "max_context_tokens": max(context_tokens, default=0),
                                           "ok": ok}
        # D1：上下文是否按轮累积（不是只带紧邻上一轮）
        multi = {"id": "acc01", "title": "六轮会话", "created_at": "", "updated_at": "",
                 "turns": [turn(f"第 {index} 轮的问题，主题编号 {index}。",
                                f"第 {index} 轮的结论：" + "这一轮的具体分析内容。" * 20)
                           for index in range(6)]}
        multi_records = conv.build_records(multi)
        shares = [compose_tokens(item["text"]) for item in multi_records]
        worst = max(shares, key=lambda item: item["context_share"])
        worst_record = multi_records[shares.index(worst)]
        last_turn = [item for item in multi_records if item["turn_index"] == 5]
        accumulation = {
            "records": len(multi_records),
            "last_turn_context_lines": compose_tokens(last_turn[-1]["text"])["context_lines"],
            "max_context_tokens": worst["context"],
            "max_context_share": worst["context_share"],
            "worst_turn": worst_record["turn_index"],
            "worst_total": worst["total"],
            "body_texts_used": [bool(item.get("body_text")) for item in multi_records],
        }
        self.report.metrics["context_accumulation"] = accumulation
        print(f"  上下文累积（6 轮会话）：{accumulation['records']} 个片段，最坏片段是第 "
              f"{accumulation['worst_turn']} 轮，上下文 {accumulation['max_context_tokens']}/"
              f"{accumulation['worst_total']} token = {accumulation['max_context_share']:.0%}，"
              f"最后一片的上下文行数 {accumulation['last_turn_context_lines']}"
              f"（紧邻 1 轮=2 行：prev_user + prev_answer）")
        if accumulation["max_context_share"] > 0.5 or accumulation["last_turn_context_lines"] > 2:
            self.report.add_defect(
                "D1a", "中",
                "切块把每一轮的『上一轮上下文』累积进后续所有片段，最坏片段上下文占大头",
                f"6 轮会话最坏片段：上下文 {accumulation['max_context_tokens']}/"
                f"{accumulation['worst_total']} token（{accumulation['max_context_share']:.0%}），"
                f"最后一片的上下文行数={accumulation['last_turn_context_lines']}"
                f"（按注释应只带紧邻 1 轮=2 行）。",
                "python -X utf8 scripts/acceptance_conversations.py（切块段）")

    def check_noise_line(self) -> None:
        """decide_candidates：低于噪声线的候选不能当历史报出来。"""
        threshold = conv.CONVERSATION_MIN_SCORE
        noise = threshold * conv.CONVERSATION_CANDIDATE_FLOOR_RATIO
        margin = conv.CONVERSATION_AMBIGUOUS_MARGIN

        def ranked(scores):
            return [{"session_id": f"s{i}", "title": f"会话 {i}", "updated_at": "",
                     "score": score, "best_turn_id": f"s{i}:0", "primary_hit": True,
                     "hits": [{"turn_id": f"s{i}:0", "score": score, "primary_hit": True}]}
                    for i, score in enumerate(scores)]

        below = conv.decide_candidates(ranked([noise - 0.01, noise - 0.02]), threshold=threshold,
                                       noise=noise, margin=margin, max_sessions=3)
        between = conv.decide_candidates(ranked([0.845, 0.80]), threshold=threshold, noise=noise,
                                         margin=margin, max_sessions=3)
        found = conv.decide_candidates(ranked([0.92, 0.83]), threshold=threshold, noise=noise,
                                       margin=margin, max_sessions=3)
        tie = conv.decide_candidates(ranked([0.92, 0.92 - margin / 2]), threshold=threshold,
                                     noise=noise, margin=margin, max_sessions=3)
        just_clear = conv.decide_candidates(ranked([0.92, 0.92 - margin]), threshold=threshold,
                                            noise=noise, margin=margin, max_sessions=3)
        capped = conv.decide_candidates(ranked([0.95, 0.94, 0.93, 0.92]), threshold=threshold,
                                        noise=noise, margin=margin, max_sessions=2)
        ok = (below["status"] == "not_found" and below["candidates"] == []
              and between["status"] == "not_found" and len(between["candidates"]) == 2
              and found["status"] == "found" and len(found["candidates"]) == 1
              and tie["status"] == "ambiguous" and len(tie["candidates"]) == 2
              # 分差正好等于 margin 不算模糊（严格小于才算）：found，且过门槛的候选都报出来
              and just_clear["status"] == "found" and len(just_clear["candidates"]) == 2
              and len(capped["candidates"]) <= 2)
        print(f"  decide_candidates：噪声线={noise:.4f}（0.86×0.93）"
              f" 低于噪声→{below['status']}/{len(below['candidates'])}候选；"
              f"噪声与门槛之间→{between['status']}/{len(between['candidates'])}候选；"
              f"明显命中→{found['status']}/{len(found['candidates'])}候选；"
              f"分差<margin→{tie['status']}/{len(tie['candidates'])}候选；"
              f"分差==margin→{just_clear['status']}/{len(just_clear['candidates'])}候选（严格小于才算模糊）；"
              f"max_sessions=2→{len(capped['candidates'])}候选")
        self.report.metrics["noise_line"] = {"noise": noise, "below": below["status"],
                                      "between": between["status"], "found": found["status"],
                                      "tie": tie["status"], "just_clear": just_clear["status"],
                                      "capped": len(capped["candidates"]), "ok": ok}

    # ================================================================ 汇总

    def summary(self) -> int:
        self.report.print()
        metrics = dict(self.report.metrics)
        metrics.pop("positive_rows", None)
        print("\n" + "=" * 78)
        print("关键数字汇总（JSON）")
        print("=" * 78)
        print(json.dumps({k: v for k, v in metrics.items() if k != "standard11"}, ensure_ascii=False,
                         indent=2, default=str))
        print("\n" + "=" * 78)
        passed = sum(1 for item in self.report.standards if item["verdict"] == "PASS")
        failed = sum(1 for item in self.report.standards if item["verdict"] == "FAIL")
        skipped = sum(1 for item in self.report.standards if item["verdict"] == "SKIP")
        blocking = [item for item in self.report.defects if item["severity"] in ("中", "高")]
        print(f"结论：PASS {passed} / FAIL {failed} / SKIP {skipped}"
              f"（正样本严格 {self.report.metrics.get('positive_hits')}/"
              f"{self.report.metrics.get('positive_total')}、宽松 "
              f"{self.report.metrics.get('positive_relaxed_hits')}/"
              f"{self.report.metrics.get('positive_total')}，"
              f"无关误命中 {self.report.metrics.get('unrelated_false_hits')}/"
              f"{self.report.metrics.get('unrelated_total')}，"
              f"缺陷 {len(self.report.defects)} 项，其中中/高 {len(blocking)} 项）")
        print("=" * 78)
        return 1 if (failed or blocking) else 0


def check_preconditions() -> str | None:
    if not rag_store.model_ready():
        return f"真实 embedding 模型不完整：{rag_store.model_dir()}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跨会话回忆独立验收")
    parser.add_argument("--skip-regression", action="store_true", help="跳过 11 号标准的回归套件")
    parser.add_argument("--stub-summary", action="store_true", help="摘要不用真实 API（只用桩失败路径）")
    parser.add_argument("--keep", action="store_true", help="保留临时目录")
    args = parser.parse_args(argv)

    problem = check_preconditions()
    if problem:
        print(f"[SKIP] {problem}")
        return 2

    run = Acceptance(use_real_summary=not args.stub_summary, keep=args.keep)
    run.real_settings = lambda: None  # type: ignore[attr-defined]
    if run.use_real_summary:
        from api_setup import active_api_settings
        settings = active_api_settings()
        if not all(str(settings.get(name) or "").strip() for name in ("API_KEY", "MODEL", "BASE_URL")):
            print("[SKIP] 没有可用的模型配置，摘要真实性无法验证；请加 --stub-summary")
            return 2
        run.real_settings = lambda: dict(settings)
    try:
        run.setup()
        run.write_corpus()
        run.start_service()
        run.measure_quality()
        run.measure_threshold_sweep()
        ambiguous_info = run.check_ambiguous()
        run.standard_1()
        run.standard_2(ambiguous_info)
        run.standard_3()
        run.standard_4()
        run.standard_5()
        run.standard_6()
        run.standard_7(ambiguous_info)
        run.standard_8()
        run.standard_9()
        run.standard_10()
        run.standard_11(args.skip_regression)
        run.check_exclusion_and_target()
        run.check_tool_layer_exclusion()
        run.check_sessions_dir_isolation()
        run.check_summary_traceability()
        run.check_summary_stale()
        run.check_no_credential_leak()
        run.check_active_version_only()
        run.check_chunking()
        run.check_noise_line()
    except Exception:
        traceback.print_exc()
        run.report.add(0, "验收脚本自身运行", False, "脚本中断，见上方 traceback")
        return_code = 2
    else:
        return_code = run.summary()
    finally:
        if not run.keep:
            run.teardown()
        else:
            print(f"[keep] 临时目录：{run.root}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
