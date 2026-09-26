"""Collect API settings and persist the selected model endpoint."""

from getpass import getpass
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit


API_FILE = Path(__file__).resolve().parent / "api.py"
ACTIVE_API_FILE = API_FILE.with_name("api_active.json")


def _required(prompt: str, *, secret: bool = False) -> str:
    ask = getpass if secret else input
    while True:
        value = ask(prompt).strip()
        if value:
            return value
        print("不能为空，请重新输入。")


def _base_url() -> str:
    while True:
        value = _required("Base URL: ")
        try:
            parts = urlsplit(value)
            valid = parts.scheme in {"http", "https"} and bool(parts.hostname)
        except ValueError:
            valid = False
        if valid:
            return value
        print("请输入完整的 http:// 或 https:// 地址。")


def prompt_api_settings() -> dict[str, str]:
    return {
        "API_KEY": _required("API Key: ", secret=True),
        "MODEL": _required("Model: "),
        "BASE_URL": _base_url(),
    }


def active_api_settings(path: Path = ACTIVE_API_FILE) -> dict[str, str]:
    if path.is_file():
        settings = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(settings, dict) or any(
            not isinstance(settings.get(name), str) or not settings[name]
            for name in ("API_KEY", "MODEL", "BASE_URL")
        ):
            raise ValueError(f"invalid API configuration: {path}")
        return settings

    import api
    return {"API_KEY": api.API_KEY, "MODEL": api.MODEL, "BASE_URL": api.BASE_URL}


def save_active_api_settings(settings: dict[str, str], path: Path = ACTIVE_API_FILE) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".api_active.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(settings, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def ensure_api_config(path: Path = API_FILE) -> bool:
    """Create api.py on first use; never replace an existing configuration."""
    if path.is_file():
        return False

    print("首次使用 Gagent，请填写模型接口配置。")
    settings = prompt_api_settings()
    content = "".join(f"{name} = {settings[name]!r}\n" for name in ("API_KEY", "MODEL", "BASE_URL"))
    created = False
    try:
        with path.open("x", encoding="utf-8") as stream:
            created = True
            stream.write(content)
    except FileExistsError:
        return False
    except OSError:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise
    print(f"配置已保存到 {path}")
    return True
