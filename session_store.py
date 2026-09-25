"""CLI 的本地会话存储：sessions/state.json 记录活动会话，sessions/<id>.json 保存单个会话。

只用标准库和 JSON，不引入数据库。位置由本文件位置决定，跟启动目录无关。
写入走同目录临时文件再替换，中途退出不破坏原文件；读不出来的会话文件按错误上报，
不会被当成空会话覆盖。跨轮历史只保存每轮的 user 和 final_answer；标题允许为空，
由调用方在首轮问答后填成用户的第一个问题。
"""
from datetime import datetime
import json
import os
from pathlib import Path
import re
import tempfile
import uuid


SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"
STATE_FILENAME = "state.json"
ACTIVE_KEY = "active_session"
ID_CHARS = 6
ID_RE = re.compile(r"\A[A-Za-z0-9_-]{1,32}\Z")
TITLE_MAX_CHARS = 40
UNTITLED_TITLE = "未命名会话"


class SessionNotFound(Exception):
    """按 ID 找不到会话文件。"""


class SessionCorrupted(Exception):
    """会话文件在但读不出合法内容；调用方只能提示，不能覆盖。"""


def new_session_id() -> str:
    return uuid.uuid4().hex[:ID_CHARS]


def _stamp() -> str:
    return datetime.now().astimezone().isoformat()


def _text(value) -> str:
    return value if isinstance(value, str) else "" if value is None else str(value)


class SessionStore:
    def __init__(self, root: Path | str = SESSIONS_DIR):
        self.root = Path(root)

    # ---------- 路径 ----------

    def path_of(self, session_id: str) -> Path:
        if not ID_RE.match(_text(session_id)):
            raise SessionNotFound(_text(session_id))
        return self.root / f"{session_id}.json"

    @property
    def state_path(self) -> Path:
        return self.root / STATE_FILENAME

    def _ensure_dir(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _write_json(self, path: Path, payload: dict) -> None:
        """同目录临时文件 + os.replace，避免半截 JSON。"""
        self._ensure_dir()
        handle_fd, temp_name = tempfile.mkstemp(dir=str(self.root), prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    # ---------- 会话读写 ----------

    def read(self, session_id: str) -> dict:
        path = self.path_of(session_id)
        if not path.is_file():
            raise SessionNotFound(str(path))
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SessionCorrupted(f"{path.name} 读取失败：{type(exc).__name__}") from exc
        return self._validate(raw, expected_id=session_id, source=path.name)

    def write(self, session: dict) -> None:
        session = self._validate(session, expected_id=session.get("id"), source="会话")
        self._write_json(self.path_of(session["id"]), session)

    def create(self, title: str | None = None) -> dict:
        """新建空会话文件；不影响其他会话，也不改活动标记。"""
        self._ensure_dir()
        stamp = _stamp()
        for _ in range(50):
            session_id = new_session_id()
            if not self.path_of(session_id).exists():
                break
        else:  # pragma: no cover - 6 位 ID 连续撞 50 次
            raise OSError("无法生成唯一的会话 ID")
        session = {"id": session_id, "title": clean_title(title),
                   "created_at": stamp, "updated_at": stamp, "turns": []}
        self.write(session)
        return session

    def scan(self) -> tuple[list[dict], list[str]]:
        """返回 (按更新时间倒序的会话元数据, 读不出来的文件名)。"""
        if not self.root.is_dir():
            return [], []
        metas, broken = [], []
        for path in sorted(self.root.glob("*.json")):
            if path.name == STATE_FILENAME or not path.is_file():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                session = self._validate(raw, expected_id=path.stem, source=path.name)
            except (OSError, ValueError, SessionCorrupted, SessionNotFound) as exc:
                broken.append(f"{path.name}（{type(exc).__name__}: {exc}）")
                continue
            metas.append({"id": session["id"], "title": session["title"],
                          "created_at": session["created_at"], "updated_at": session["updated_at"],
                          "turns": len(session["turns"])})
        metas.sort(key=lambda item: (item["updated_at"], item["created_at"]), reverse=True)
        return metas, broken

    def _validate(self, raw, expected_id: str | None, source: str) -> dict:
        if not isinstance(raw, dict):
            raise SessionCorrupted(f"{source} 内容不是对象")
        session_id = _text(raw.get("id"))
        if not session_id:
            raise SessionCorrupted(f"{source} 缺少 id")
        if expected_id and session_id != expected_id:
            raise SessionCorrupted(f"{source} 的 id 与文件名不一致")
        turns_raw = raw.get("turns", [])
        if not isinstance(turns_raw, list):
            raise SessionCorrupted(f"{source} 的 turns 不是列表")
        turns = []
        for item in turns_raw:
            if not isinstance(item, dict):
                raise SessionCorrupted(f"{source} 的某一轮记录不是对象")
            turns.append({"user": _text(item.get("user")), "final_answer": _text(item.get("final_answer"))})
        return {"id": session_id, "title": clean_title(raw.get("title")),
                "created_at": _text(raw.get("created_at")), "updated_at": _text(raw.get("updated_at")),
                "turns": turns}

    # ---------- 活动会话 ----------

    def active_id(self) -> str | None:
        if not self.state_path.is_file():
            return None
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SessionCorrupted(f"{STATE_FILENAME} 读取失败：{type(exc).__name__}") from exc
        if not isinstance(raw, dict):
            raise SessionCorrupted(f"{STATE_FILENAME} 内容不是对象")
        return _text(raw.get(ACTIVE_KEY)) or None

    def set_active(self, session_id: str) -> None:
        self._write_json(self.state_path, {ACTIVE_KEY: session_id})

    def save_turn(self, session: dict, user: str, final_answer: str) -> None:
        """先更新内存再落盘：写失败时调用方仍持有这轮结果，可以再试。"""
        session["turns"].append({"user": user, "final_answer": final_answer})
        session["updated_at"] = _stamp()
        self.write(session)


def clean_title(title) -> str:
    """标题压成单行并限长；空标题返回空串，由调用方决定默认名称。"""
    collapsed = " ".join(_text(title).replace("\t", " ").split())
    return collapsed[:TITLE_MAX_CHARS].strip()
