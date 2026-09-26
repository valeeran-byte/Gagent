"""Gagent 终端聊天界面：单行输入、滚动输出、多会话切换。

会话历史来自 sessions/ 下按 ID 存放的 JSON；本进程只顺序执行一个会话轮次。
详细过程写进 logs/ 的运行日志，控制台只显示工具进度和最终回答。
入库完成/失败提示来自本地文件服务的事件接口，退出本程序不影响服务继续处理任务。
"""
from datetime import datetime
import argparse
import os
from pathlib import Path
import sys
import threading

from api_setup import ensure_api_config, prompt_api_settings, save_active_api_settings
from rag import client as rag_client
from session_store import (SESSIONS_DIR, UNTITLED_TITLE, SessionCorrupted, SessionNotFound,
                           SessionStore, clean_title)


PROMPT = "❯ "
TOOL_LABELS = {
    "web_search": "正在搜索……",
    "visit_webpage": "正在读取网页……",
    "search_and_read": "正在搜索并读取网页……",
    "read_html": "正在读取网页……",
    "find_in_page": "正在定位网页内容……",
    "read_pdf": "正在读取 PDF……",
    "find_in_pdf": "正在定位 PDF 页码……",
    "read_excel": "正在读取 Excel……",
    "read_webpage_tables": "正在读取网页表格……",
    "read_text_file": "正在读取文本文件……",
    "run_python": "正在计算……",
}
HELP_TEXT = """命令：
  /help             显示本帮助
  /api              输入并切换模型接口
  /new [标题]       创建并进入一个新的空会话；不写标题时用第一个提问作会话名
  /sessions         列出已保存的会话，* 表示当前会话
  /switch           列出会话供选择，输入编号进入
  /switch <id>      直接进入指定会话，id 取自 /sessions
  /memory           查看长期记忆（文件入库）状态、最近事件和失败原因
  /retry <job_id>   重新尝试一个已失败的入库任务
  /exit             退出 Gagent
直接输入文字就是向当前会话提问。
同一个会话保留上下文：追问会带上本会话已完成的问答，模型据此接着回答。
/new 创建独立会话，不共享其他会话的历史；/switch 恢复已有会话，只加载该会话已完成的
问答，不会重放工具过程和中间结果。
工具调用、工具返回内容和运行详情只写 logs/ 日志，不进入会话历史。
read_pdf/read_excel 读过的文件会由本地服务在后台切块入库，完成后弹提示；
入库中或失败时用 /memory 查看进度和原因，退出 Gagent 不影响后台处理。"""


def _label(tool: str) -> str:
    return TOOL_LABELS.get(tool, "正在调用工具……")


class RagNotifier:
    """把文件服务的入库完成/失败事件显示成一行提示。

    提示属于界面，不进会话历史也不发给模型；正在等待输入时先攒着，等这行输入
    被提交后再显示，避免打断用户正在编辑的内容。已消费到的事件位置记在本地，
    关闭期间完成的任务下次打开会补提示，但不会重复弹。
    """

    def __init__(self, emit, poll_seconds: float | None = None, client=None):
        self.emit = emit
        self.client = client if client is not None else rag_client
        self.poll = float(poll_seconds if poll_seconds is not None
                          else os.environ.get("GAGENT_RAG_POLL_SECONDS", "4"))
        self.cursor = self.client.read_cursor()
        self._pending: list[str] = []
        self._typing = False
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None or self.poll <= 0:
            return
        self._thread = threading.Thread(target=self._loop, name="rag-notices", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.poll):
            try:
                self.poll_once()
            except Exception:  # 提示失败不该影响输入循环
                continue

    def poll_once(self) -> list[str]:
        """取新事件并显示；服务没在运行时安静跳过，不为此拉起服务。"""
        try:
            data = self.client.events_after(self.cursor)
        except Exception:
            return []
        events = data.get("events") or []
        if not events:
            return []
        self.cursor = int(data.get("cursor") or self.cursor)
        self.client.write_cursor(self.cursor)
        notices = [_notice_text(event) for event in events]
        self.deliver(notices)
        return notices

    def deliver(self, notices: list[str]) -> None:
        lines = [line for line in notices if line]
        with self._lock:
            if self._typing:
                self._pending.extend(lines)
                return
        self._emit(lines)

    def _emit(self, lines: list[str]) -> None:
        for line in lines:
            self.emit(line)

    def begin_input(self) -> None:
        with self._lock:
            self._typing = True

    def end_input(self) -> None:
        """这一行输入被提交后才补显示攒下的提示，避免打断正在编辑的内容。"""
        with self._lock:
            self._typing = False
            lines, self._pending = self._pending, []
        self._emit(lines)

    def flush(self) -> None:
        with self._lock:
            lines, self._pending = self._pending, []
        self._emit(lines)


def _notice_text(event: dict) -> str:
    message = str(event.get("message") or "").strip()
    if not message:
        return ""
    if str(event.get("kind")) == "failed":
        return f"{message}（用 /memory 查看原因）"
    return message


def _title(session: dict) -> str:
    return session["title"] or UNTITLED_TITLE


def _when(stamp: str) -> str:
    try:
        return datetime.fromisoformat(stamp).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return stamp or "未知时间"


class Chat:
    def __init__(self, store: SessionStore, agent, out=sys.stdout, input_fn=input, notifier=None):
        self.store = store
        self.agent = agent
        self.out = out
        self.input_fn = input_fn
        self.session = None
        self.unsaved: list[dict] = []
        self._showed_progress = False
        self.notifier = notifier if notifier is not None else RagNotifier(self.notify)

    # ---------- 输出 ----------

    def line(self, text: str = "") -> None:
        print(text, file=self.out, flush=True)

    def say(self, text: str) -> None:
        self.line(f"！{text}")

    def notify(self, text: str) -> None:
        """轻量提示：一行一个事件，与工具进度同样的显示密度，不占用输入行。"""
        self.line(f"● {text}")

    # ---------- 启动 ----------

    def start(self) -> int:
        self.line("Gagent")
        self.notifier.start()
        self.notifier.poll_once()  # 打开时先补一次提示，不靠轮询撞上时间
        notice = self._open_session()
        if notice:
            self.say(notice)
            self.show_sessions()
            self.line("用 /switch 选择一个会话，或 /new 新建一个。")
        else:
            self.show_current()
        self.line("输入 /help 查看命令。")
        self.line()
        return self.loop()

    def _open_session(self) -> str | None:
        """选定启动时的会话；返回需要提示用户的原因，不静默改动旧数据。"""
        metas, _ = self.store.scan()
        try:
            active = self.store.active_id()
        except SessionCorrupted as exc:
            return f"上次会话标记读不出来：{exc}"
        if not active:
            if not metas:
                self.session = self.store.create()
                self.store.set_active(self.session["id"])
                return None
            return "没有找到上次会话标记。"
        try:
            self.session = self.store.read(active)
        except SessionNotFound:
            return f"上次会话 {active} 的文件不存在。"
        except SessionCorrupted as exc:
            return f"上次会话 {active} 读不出来：{exc}"
        return None

    def show_current(self) -> None:
        if self.session is None:
            self.line("当前会话：未选定")
            return
        turns = len(self.session["turns"])
        self.line(f"当前会话：{_title(self.session)} [{self.session['id']}]（{turns} 轮）")

    # ---------- 输入循环 ----------

    def loop(self) -> int:
        while True:
            try:
                self.notifier.begin_input()
                try:
                    text = self.input_fn(PROMPT)
                finally:
                    self.notifier.end_input()
            except KeyboardInterrupt:
                self.line("\n")
                return self._shutdown()
            except EOFError:
                self.line("\n")
                return self._shutdown()
            try:
                if not self.handle(text):
                    return self._shutdown()
            except Exception as exc:  # 单次轮次出错不退出 CLI
                self.say(f"出现异常：{type(exc).__name__}: {exc}")

    def handle(self, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return True
        if not stripped.startswith("/"):
            self.ask(stripped)
            return True
        name, _, argument = stripped[1:].partition(" ")
        name, argument = name.lower(), argument.strip()
        if name == "exit":
            return False
        if name == "help":
            self.line(HELP_TEXT)
        elif name == "api":
            if argument:
                self.line("用法：/api")
            else:
                self.switch_api()
        elif name == "new":
            self.new_session(argument)
        elif name == "sessions":
            self.show_sessions()
        elif name == "switch":
            self.switch_session(argument)
        elif name == "memory":
            self.show_memory()
        elif name == "retry":
            self.retry_ingestion(argument)
        else:
            self.line(f"未知命令：/{name}，输入 /help 查看全部命令。")
        return True

    # ---------- 长期记忆 ----------

    def show_memory(self) -> None:
        """入库状态：待办/失败任务与最近事件，失败原因在这里可查。"""
        try:
            data = rag_client.state(20)
        except Exception as exc:
            self.say(f"读不到长期记忆状态：{exc}")
            return
        info = rag_client.health() or {}
        self.line(f"文件服务：{'运行中' if info else '未运行'}"
                  f"  资料 {info.get('documents', 0)} 个  向量 {info.get('vectors', 0)} 条"
                  f"  待处理 {info.get('jobs_pending', 0)}  失败 {info.get('jobs_failed', 0)}")
        if not data.get("model_ready", True):
            self.say("embedding 模型还没下载：运行 python file_service.py --prepare")
        jobs = data.get("jobs") or []
        if jobs:
            self.line("入库任务：")
            for job in jobs:
                note = f"：{str(job.get('last_error') or '')[:120]}" if job.get("last_error") else ""
                self.line(f"  {job['status']:<11} 已试 {job['attempts']}/{job['max_attempts']}"
                          f"  {job.get('file_name') or ''}  [{job['job_id']}]{note}")
                if job["status"] == "failed":
                    self.line(f"    重试用 /retry {job['job_id']}")
        else:
            self.line("入库任务：没有待办或失败的任务")
        events = data.get("events") or []
        if events:
            self.line("最近事件：")
            for event in events[:10]:
                detail = f"  {str(event.get('detail'))[:120]}" if event.get("detail") else ""
                self.line(f"  {_when(event.get('created_at'))}  {event.get('message') or ''}{detail}")

    def retry_ingestion(self, job_id: str) -> None:
        if not job_id or len(job_id.split()) > 1:
            self.line("用法：/retry <job_id>（job_id 见 /memory）")
            return
        try:
            result = rag_client.retry_job(job_id)
        except Exception as exc:
            self.say(f"重试未能提交：{exc}")
            return
        self.line(f"已提交重试：新任务 {result['job_id']}（第 {result['round']} 轮，最多 3 次尝试）")

    # ---------- 命令 ----------

    def switch_api(self) -> None:
        self.line("请输入新的模型接口配置：")
        try:
            settings = prompt_api_settings()
            import G_agent
            agent = G_agent.BasicAgent(api_settings=settings)
            save_active_api_settings(settings)
        except (KeyboardInterrupt, EOFError):
            self.line("\n已取消，当前模型接口不变。")
            return
        except OSError as exc:
            self.say(f"保存模型接口失败：{exc}。当前配置不变。")
            return
        except Exception as exc:
            self.say(f"切换模型接口失败：{type(exc).__name__}。当前配置不变。")
            return
        self.agent = agent
        self.line(f"已切换模型接口：{settings['MODEL']} ({settings['BASE_URL']})")

    def new_session(self, title: str) -> None:
        try:
            session = self.store.create(title or None)
        except OSError as exc:
            self.say(f"创建会话失败：{type(exc).__name__}: {exc}")
            return
        suffix = "" if session["title"] else "（第一个提问会成为会话名）"
        self.activate(session, f"已创建并切换到：{_title(session)} [{session['id']}]{suffix}")

    def switch_session(self, argument: str) -> None:
        if not argument:
            self.pick_session()
            return
        if len(argument.split()) > 1:
            self.line("用法：/switch 选择会话，或 /switch <id>（id 见 /sessions）。")
            return
        try:
            session = self.store.read(argument)
        except SessionNotFound:
            self.say(f"找不到会话 {argument}，用 /sessions 查看会话编号。当前会话不变。")
            return
        except SessionCorrupted as exc:
            self.say(f"会话 {argument} 读不出来：{exc}。当前会话不变。")
            return
        self.activate(session, f"已切换到：{_title(session)} [{session['id']}]")

    def pick_session(self) -> None:
        """列出会话供选择：用户输入编号，不用记 ID。"""
        metas, _broken = self.store.scan()
        if not metas:
            self.line("还没有已保存的会话，用 /new 创建一个。")
            return
        current = self.session["id"] if self.session else None
        self.line("选择会话（输入编号，直接回车取消）：")
        for index, meta in enumerate(metas, start=1):
            mark = "*" if meta["id"] == current else " "
            self.line(f"{mark} {index}) {_title(meta)} [{meta['id']}]"
                      f"  更新于 {_when(meta['updated_at'])}  {meta['turns']} 轮")
        try:
            answer = (self.input_fn("编号 ❯ ") or "").strip()
        except (KeyboardInterrupt, EOFError):
            self.line("已取消。")
            return
        if not answer:
            self.line("已取消，保持当前会话。")
            return
        if not answer.isdigit() or not 1 <= int(answer) <= len(metas):
            self.line(f"没有第 {answer} 个会话，已取消。")
            return
        chosen = metas[int(answer) - 1]
        if chosen["id"] == current:
            self.line(f"已经在：{_title(chosen)} [{chosen['id']}]")
            return
        self.activate(self.store.read(chosen["id"]),
                      f"已切换到：{_title(chosen)} [{chosen['id']}]")

    def show_sessions(self) -> None:
        metas, broken = self.store.scan()
        if not metas:
            self.line("还没有已保存的会话，用 /new 创建一个。")
        else:
            self.line("会话列表：")
            current = self.session["id"] if self.session else None
            for meta in metas:
                mark = "*" if meta["id"] == current else " "
                self.line(f"{mark} {meta['id']}  {_title(meta)}  更新于 {_when(meta['updated_at'])}"
                          f"  {meta['turns']} 轮")
        for name in broken:
            self.say(f"会话文件读不出来：{name}（未被覆盖，请手动处理）")

    def activate(self, session: dict, message: str) -> None:
        self.flush_unsaved()
        self.session = session
        try:
            self.store.set_active(session["id"])
        except OSError as exc:
            self.say(f"活动会话标记未能写入：{type(exc).__name__}: {exc}；本次仍使用该会话。")
        self.line(message)

    # ---------- 提问 ----------

    def ask(self, question: str) -> None:
        if self.session is None:
            self.line("尚未选定会话：用 /new 新建，或用 /switch 选择已有会话。")
            return
        self.flush_unsaved()
        if not self.session["title"]:
            self.session["title"] = clean_title(question)
        history = self.session["turns"]
        self._showed_progress = False
        try:
            result = self.agent.run(question, history=history, progress=self.progress,
                                    session_id=self.session["id"], echo=False)
        except KeyboardInterrupt:
            self.line("（本轮已中止，未保存。）")
            return
        except Exception as exc:
            self.say(f"本轮未完成：{type(exc).__name__}，未保存。")
            self.context_hint(history)
            return
        if self._showed_progress:
            self.line()
        self.line(result.answer.strip() or "（本轮没有回答内容）")
        if not result.ok:
            self.say("本轮未完成，未保存。")
            self.context_hint(history)
            return
        self.save_turn(result.user_prompt, result.answer)

    def context_hint(self, history: list) -> None:
        if history:
            self.line("若失败原因是上下文过长，用 /new 开一个新会话再继续，历史不会被自动删减。")

    def progress(self, phase: str, tool: str, note: str) -> None:
        if phase == "tool_call":
            self._showed_progress = True
            self.line(f"● {_label(tool)}")
        elif phase == "tool_error":
            self._showed_progress = True
            self.line(f"● 工具执行失败：{note or _label(tool)}")

    # ---------- 保存 ----------

    def save_turn(self, user_prompt: str, answer: str) -> None:
        try:
            self.store.save_turn(self.session, user_prompt, answer)
        except OSError as exc:
            if not any(session is self.session for session in self.unsaved):
                self.unsaved.append(self.session)
            self.say(f"本轮未保存：{type(exc).__name__}: {exc}。结果仍在内存中，稍后会重试。")

    def flush_unsaved(self) -> None:
        remaining = []
        for session in self.unsaved:
            try:
                self.store.write(session)
            except OSError as exc:
                self.say(f"会话 {session['id']} 的未保存轮次仍写入失败：{type(exc).__name__}: {exc}")
                remaining.append(session)
            else:
                self.line(f"（已补存会话 {session['id']} 的未保存轮次）")
        self.unsaved = remaining

    def _shutdown(self) -> int:
        self.notifier.stop()
        self.flush_unsaved()
        self.line("再见。")
        return 0


def _ensure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", "") or "").lower()
        try:
            if encoding in ("", "ansi_x3.4-1968", "us-ascii"):
                stream.reconfigure(encoding="utf-8", errors="replace")
            else:
                stream.reconfigure(errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def main(argv=None, *, store: SessionStore | None = None, agent=None, out=None,
         input_fn=input, notifier=None) -> int:
    parser = argparse.ArgumentParser(prog="Gagent", description="Gagent 终端聊天")
    parser.add_argument("--sessions-dir", default=str(SESSIONS_DIR),
                        help="会话目录，默认项目下的 sessions/")
    args = parser.parse_args(argv)
    _ensure_utf8_output()
    if agent is None:
        try:
            ensure_api_config()
        except (EOFError, KeyboardInterrupt):
            print("\nAPI 配置已取消，未保存。", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"API 配置保存失败：{exc}", file=sys.stderr)
            return 1
        import G_agent
        agent = G_agent.BasicAgent()
    chat = Chat(store or SessionStore(Path(args.sessions_dir)),
                agent,
                out=out or sys.stdout, input_fn=input_fn, notifier=notifier)
    return chat.start()


if __name__ == "__main__":
    raise SystemExit(main())
