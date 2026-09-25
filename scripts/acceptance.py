"""端到端验收：真实模型 + 真实子进程执行（会访问网络）。用法：python _acceptance.py [场景名...]

任何一项检查不过就返回非 0 退出码，并把失败原因打出来。
"""
import json
import sys
import time
from pathlib import Path

import G_agent as agent
from agent_tools import run_python, PYTHON_MAX_OUTPUT

EXCEL = "https://download.microsoft.com/download/1/4/E/14EDED28-6C58-4055-A65C-23B4DA81C4DE/Financial%20Sample.xlsx"
WIKI_100M = ("https://en.wikipedia.org/wiki/"
             "Athletics_at_the_2024_Summer_Olympics_%E2%80%93_Men%27s_100_metres")
BRIO_PDF = ("https://www.logitech.com/content/dam/logitech/en_us/video-collaboration/pdf/"
            "brio-505-datasheet.pdf")

CASES = {
    "excel": {
        "question": (
            "我记得微软有一份给 Power BI 教学用的财务样本，里面好像有不同国家和产品的销售、利润数据。"
            "帮我找到原始表格，看看哪种产品的总利润最高，以及哪个国家的总利润最低。\n"
            f"原始表格地址：{EXCEL}"),
        "expect": ["Paseo", "Mexico"],
        "report": "read_excel 预览给出 local_path 后，run_python 读取该文件算出结果",
    },
    "web": {
        "question": (
            "2024 年巴黎奥运会男子 100 米决赛，请用决赛全部 8 名选手的成绩算出：平均成绩（秒，保留 3 位小数）、"
            "冠军与亚军的差距（秒，保留 3 位小数），以及最后一名与第一名的差距（秒，保留 3 位小数）。\n"
            f"参考页面：{WIKI_100M}"),
        "expect": ["0.005", "0.12"],
    },
    "pdf": {
        "question": (
            "这份 Logitech Brio 505 数据手册里写了对角视场角（dFOV）有几种可选值。"
            "请用这些 dFOV 数值算出：最大 dFOV 与最小 dFOV 相差多少度、多少弧度（保留 4 位小数）；"
            "再按「成像圆的直径与 dFOV 成正比」这个模型近似，"
            "最大 dFOV 覆盖的图像面积是最小 dFOV 的多少倍（保留 2 位小数）。\n"
            f"参考 PDF：{BRIO_PDF}"),
        "expect": ["25", "0.4363", "1.92"],
    },
    "error": {
        "question": "请用 run_python 算出 /definitely/not/here.csv 里 amount 列的总和，然后如实给出结果。",
        "expect": [],
        "expect_failure": True,
        "forbid": ["总和是", "总额为", "合计为", "等于", "总数为"],   # 不许编出一个和
    },
    # 超时由 check_timeout_path() 直接验证：真实模型会为了"通过"把 sleep 改短，
    # 那种场景验证不了超时路径，所以不放进模型驱动的场景里。
}


def check_timeout_path() -> list:
    """确定性验证：代码超时和输出过多时能及时终止。"""
    problems = []
    started = time.monotonic()
    result = run_python.invoke({"code": "import time; time.sleep(30)", "timeout": 2})
    if result.get("status") != "timeout":
        problems.append(f"超时场景状态是 {result.get('status')!r}，期望 'timeout'")
    if time.monotonic() - started > 20:
        problems.append("超时场景没有及时终止子进程")
    if not result.get("error"):
        problems.append("超时场景没有 error 说明")
    overflow = run_python.invoke({"code": "print('x' * 10000)"})
    if overflow.get("status") != "output_limit":
        problems.append(f"输出超限场景状态是 {overflow.get('status')!r}，期望 'output_limit'")
    if len(str(overflow.get("error") or "")) > PYTHON_MAX_OUTPUT + 100:
        problems.append("输出超限时回传内容没有被截断")
    return problems
# 已下载到本地的表格：read_excel 会复用本地副本，用来验证"预览 → 本地文件交接"这条链
_SEEDED_EXCEL = Path(__file__).resolve().parents[1] / "downloads" / "excel" / "Financial Sample.xlsx"
CASES["handoff"] = {
    "question": (
        "本地已经有一份微软 Financial Sample.xlsx（来源：" + EXCEL + "）。"
        "请先用 read_excel 看它的结构，再算出总利润最高的产品和总利润最低的国家。"),
    "expect": ["Paseo", "Mexico"],
    "report": "read_excel 预览给出 local_path，run_python 读同一个本地文件算出结果",
}
CASES["timeout_path"] = {"question": "", "expect": [], "report": "确定性检查超时与输出上限"}

FAILURE_WORDS = ("失败", "未能", "没有", "无法", "超时", "error")


def run_python_calls(transcript: str) -> list:
    """只从真正的 run_python 工具结果提取执行状态，不误认提示词。"""
    payloads = []
    for line in transcript.splitlines():
        if not line.startswith("tool run_python ["):
            continue
        try:
            payload = json.loads(line.split(": ", 1)[1])
        except (IndexError, ValueError):
            continue
        if isinstance(payload, dict) and "success" in payload:
            payloads.append(payload)
    return payloads


def check(name: str, answer: str, calls: list) -> list:
    """核对最终回答和实际代码执行结果。"""
    case = CASES[name]
    problems = []
    if not answer.strip():
        problems.append("没有最终回答")
    for token in case["expect"]:
        if token not in answer:
            problems.append(f"回答里缺少预期值 {token!r}")
    for token in case.get("forbid", []):
        if token in answer:
            problems.append(f"回答里出现了不该有的 {token!r}")
    if not calls:
        problems.append("没有任何 run_python 结果：无法确认实际执行过计算")
    elif case.get("expect_failure"):
        if calls[-1].get("success") is not False:
            problems.append("最后一次 run_python 居然成功了，但场景要求它失败")
        if not calls[-1].get("error"):
            problems.append("失败结果里没有 error 说明")
        if answer.strip() and not any(word in answer for word in FAILURE_WORDS):
            problems.append("模型没有如实说明失败")
    else:
        if calls[-1].get("success") is not True:
            problems.append(f"最后一次 run_python 没有成功：{calls[-1].get('error') or calls[-1].get('status')}")
        elif not str(calls[-1].get("output") or "").strip():
            problems.append("最后一次 run_python 没有输出计算结果")
    return problems

def seed_handoff() -> str | None:
    """handoff 场景需要本地已有一份工作簿：没有就用 httpx 下一份放进 downloads/。"""
    if _SEEDED_EXCEL.is_file() and _SEEDED_EXCEL.stat().st_size > 1000:
        return None
    try:
        import httpx
        _SEEDED_EXCEL.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(EXCEL, follow_redirects=True, timeout=60, verify=False,
                             headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        _SEEDED_EXCEL.write_bytes(response.content)
    except Exception as exc:
        return f"准备本地工作簿失败：{type(exc).__name__}: {exc}"
    return None


def run(name: str) -> dict:
    case = CASES[name]
    seed_error = seed_handoff() if name == "handoff" else None
    if seed_error:
        return {"case": name, "answer": "", "run_python": [], "passed": False, "problems": [seed_error]}
    instance = agent.BasicAgent()
    started = time.monotonic()
    answer = instance(case["question"])
    calls = run_python_calls(instance.last_transcript)
    problems = check(name, answer, calls)
    return {"case": name, "seconds": round(time.monotonic() - started, 1), "answer": answer,
            "run_python": calls, "problems": problems, "passed": not problems}


def main() -> int:
    names = sys.argv[1:] or list(CASES)
    results = []
    if "timeout" in names:                     # 兼容旧写法：timeout 现在是确定性检查
        names = [name for name in names if name != "timeout"] + ["timeout_path"]
    for name in names:
        if name == "timeout_path":
            print("\n===== 检查 timeout 路径（确定性）=====", flush=True)
            try:
                problems = check_timeout_path()
            except Exception as exc:
                problems = [f"检查抛异常：{type(exc).__name__}: {exc}"]
            results.append({"case": "timeout_path", "answer": "", "run_python": [],
                            "problems": problems, "passed": not problems})
            print(f"  -> {'PASS' if not problems else 'FAIL'} {problems}", flush=True)
            continue
        print(f"\n===== 场景 {name} =====", flush=True)
        try:
            results.append(run(name))
        except Exception as exc:  # 单个场景崩溃也要继续跑完剩下的，并记为失败
            results.append({"case": name, "answer": "", "run_python": [], "passed": False,
                            "problems": [f"场景抛异常：{type(exc).__name__}: {exc}"]})
        print(f"----- {name} 结果 -----\n{results[-1].get('answer')}", flush=True)
    target = Path(__file__).resolve().parent / "acceptance_result.json"
    target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [item for item in results if not item.get("passed")]
    print("\n===== 验收汇总 =====", flush=True)
    for item in results:
        mark = "PASS" if item.get("passed") else "FAIL"
        print(f"[{mark}] {item['case']} ({item.get('seconds')}s) " + "; ".join(item.get("problems") or []),
              flush=True)
    print(f"通过 {len(results) - len(failed)}/{len(results)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
