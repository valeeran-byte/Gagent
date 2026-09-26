"""First-run API configuration tests; no real credentials are used."""

import ast
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import api_setup
import cli


class ApiSetupTests(unittest.TestCase):
    def test_first_run_saves_values_and_later_run_leaves_them_alone(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "api.py"
            with patch.object(api_setup, "getpass", return_value="key'with-quote") as hidden, \
                 patch("builtins.input", side_effect=["model-a", "not-a-url", "http://[bad", "http://localhost:8000/v1"]):
                self.assertTrue(api_setup.ensure_api_config(path))
            hidden.assert_called_once_with("API Key: ")

            assignments = ast.parse(path.read_text(encoding="utf-8")).body
            values = {item.targets[0].id: ast.literal_eval(item.value) for item in assignments}
            self.assertEqual(values, {
                "API_KEY": "key'with-quote",
                "MODEL": "model-a",
                "BASE_URL": "http://localhost:8000/v1",
            })
            saved = path.read_bytes()
            with patch.object(api_setup, "getpass", side_effect=AssertionError("prompted again")), \
                 patch("builtins.input", side_effect=AssertionError("prompted again")):
                self.assertFalse(api_setup.ensure_api_config(path))
            self.assertEqual(path.read_bytes(), saved)

    def test_cancel_does_not_create_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "api.py"
            with patch.object(api_setup, "getpass", side_effect=EOFError):
                with self.assertRaises(EOFError):
                    api_setup.ensure_api_config(path)
            self.assertFalse(path.exists())

    def test_saved_selection_is_loaded_on_next_start(self):
        original = SimpleNamespace(API_KEY="old", MODEL="old-model", BASE_URL="https://old.example/v1")
        selection = {"API_KEY": "new", "MODEL": "new-model", "BASE_URL": "https://new.example/v1"}
        with TemporaryDirectory() as directory, patch.dict(sys.modules, {"api": original}):
            path = Path(directory) / "api_active.json"
            self.assertEqual(api_setup.active_api_settings(path)["MODEL"], "old-model")
            api_setup.save_active_api_settings(selection, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), selection)
            self.assertEqual(api_setup.active_api_settings(path), selection)

    def test_agent_uses_selected_settings_at_startup(self):
        import G_agent

        selection = {"API_KEY": "new", "MODEL": "new-model", "BASE_URL": "https://new.example/v1"}
        with patch.object(G_agent, "active_api_settings", return_value=selection), \
             patch.object(G_agent, "ChatOpenAI") as model:
            G_agent.BasicAgent()
        self.assertEqual(model.call_args.kwargs["api_key"], selection["API_KEY"])
        self.assertEqual(model.call_args.kwargs["model"], selection["MODEL"])
        self.assertEqual(model.call_args.kwargs["base_url"], selection["BASE_URL"])

    def test_api_command_switches_agent_and_help_lists_it(self):
        previous = object()
        replacement = object()
        selection = {"API_KEY": "new", "MODEL": "new-model", "BASE_URL": "https://new.example/v1"}
        events = []
        def make_agent(*, api_settings):
            events.append(("agent", api_settings))
            return replacement
        def save(settings):
            events.append(("save", settings))
        chat = cli.Chat(object(), previous, out=io.StringIO(), notifier=object())
        with patch.object(cli, "prompt_api_settings", return_value=selection), \
             patch.object(cli, "save_active_api_settings", side_effect=save), \
             patch.dict(sys.modules, {"G_agent": SimpleNamespace(BasicAgent=make_agent)}):
            self.assertTrue(chat.handle("/api"))
        self.assertEqual(events, [("agent", selection), ("save", selection)])
        self.assertIs(chat.agent, replacement)
        self.assertIn("/api", cli.HELP_TEXT)
        self.assertIn("已切换模型接口", chat.out.getvalue())

    def test_api_command_cancel_and_save_failure_keep_current_agent(self):
        previous = object()
        chat = cli.Chat(object(), previous, out=io.StringIO(), notifier=object())
        with patch.object(cli, "prompt_api_settings", side_effect=EOFError), \
             patch.object(cli, "save_active_api_settings") as save:
            self.assertTrue(chat.handle("/api"))
            save.assert_not_called()
        self.assertIs(chat.agent, previous)

        selection = {"API_KEY": "new", "MODEL": "new-model", "BASE_URL": "https://new.example/v1"}
        with patch.object(cli, "prompt_api_settings", return_value=selection), \
             patch.object(cli, "save_active_api_settings", side_effect=OSError("disk full")), \
             patch.dict(sys.modules, {"G_agent": SimpleNamespace(BasicAgent=lambda **kwargs: object())}):
            self.assertTrue(chat.handle("/api"))
        self.assertIs(chat.agent, previous)

    def test_cli_collects_config_before_creating_agent(self):
        events = []
        fake_agent = SimpleNamespace(BasicAgent=lambda: events.append("agent") or object())
        with patch.object(cli, "ensure_api_config", side_effect=lambda: events.append("config")), \
             patch.dict(sys.modules, {"G_agent": fake_agent}), \
             patch.object(cli, "Chat") as chat:
            chat.return_value.start.return_value = 0
            self.assertEqual(cli.main([], store=object()), 0)
        self.assertEqual(events, ["config", "agent"])

    def test_cli_cancel_exits_before_creating_agent(self):
        with patch.object(cli, "ensure_api_config", side_effect=EOFError), \
             patch.object(cli, "Chat") as chat:
            self.assertEqual(cli.main([], store=object()), 1)
            chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
