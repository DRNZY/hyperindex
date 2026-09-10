"""Tests for HyperIndex CLI entrypoint and interactive TUI."""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.syntax import Syntax
from textual.widgets import Input, ListView, Static
from typer.testing import CliRunner

from hyperindex.cli import app
from hyperindex.search import SearchResult
from hyperindex.tui.app import (
    HyperIndexApp,
    copy_to_clipboard,
    get_editor_command,
    launch_editor,
)

runner = CliRunner()


def make_sample_results():
    return [
        SearchResult(
            chunk_id=1,
            path=Path("/app/auth.py"),
            start_line=15,
            end_line=25,
            symbol="verify_token",
            content="def verify_token(token: str) -> bool:\n    return token == 'secret'",
            score=0.9876,
        ),
        SearchResult(
            chunk_id=2,
            path=Path("/app/session.py"),
            start_line=40,
            end_line=55,
            symbol="SessionStore",
            content="class SessionStore:\n    def __init__(self):\n        self.sessions = {}",
            score=0.8765,
        ),
    ]


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "HyperIndex" in result.stdout
    assert "search" in result.stdout
    assert "tui" in result.stdout
    assert "index" in result.stdout
    assert "watch" in result.stdout


def test_cli_search_help():
    result = runner.invoke(app, ["search", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.stdout
    assert "--plain" in result.stdout
    assert "--paths-only" in result.stdout
    assert "--limit" in result.stdout
    assert "--tui" in result.stdout


def test_cli_search_default_and_plain():
    with patch("hyperindex.cli.HybridSearchEngine") as MockEngine:
        mock_instance = MagicMock()
        MockEngine.return_value = mock_instance
        mock_instance.search.return_value = make_sample_results()

        result = runner.invoke(app, ["search", "verify_token"])
        assert result.exit_code == 0
        assert "auth.py:15" in result.stdout
        assert "[verify_token]" in result.stdout
        assert "0.9876" in result.stdout
        assert "def verify_token" in result.stdout


def test_cli_search_plain_flag_strips_ansi():
    with patch("hyperindex.cli.get_engine") as mock_get_engine:
        mock_instance = MagicMock()
        mock_get_engine.return_value = mock_instance
        mock_instance.search.return_value = make_sample_results()

        # Run with --plain: clean text without ANSI escape sequences
        result_plain = runner.invoke(app, ["search", "verify_token", "--plain"], color=True)
        assert result_plain.exit_code == 0
        assert "auth.py:15" in result_plain.stdout
        assert "\x1b[" not in result_plain.stdout

        # Run without --plain (with color enabled in terminal): contains ANSI escape sequences
        result_color = runner.invoke(app, ["search", "verify_token"], color=True)
        assert result_color.exit_code == 0
        assert "auth.py:15" in result_color.stdout
        assert "\x1b[" in result_color.stdout


def test_cli_search_json_output():
    with patch("hyperindex.cli.HybridSearchEngine") as MockEngine:
        mock_instance = MagicMock()
        MockEngine.return_value = mock_instance
        mock_instance.search.return_value = make_sample_results()

        result = runner.invoke(app, ["search", "verify_token", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["path"] == "/app/auth.py"
        assert data[0]["start_line"] == 15
        assert data[0]["end_line"] == 25
        assert data[0]["symbol"] == "verify_token"
        assert data[0]["score"] == 0.9876
        assert "def verify_token" in data[0]["content"]


def test_cli_search_paths_only():
    with patch("hyperindex.cli.HybridSearchEngine") as MockEngine:
        mock_instance = MagicMock()
        MockEngine.return_value = mock_instance
        mock_instance.search.return_value = make_sample_results()

        result = runner.invoke(app, ["search", "verify_token", "--paths-only"])
        assert result.exit_code == 0
        paths = [p.strip() for p in result.stdout.splitlines() if p.strip()]
        assert paths == ["/app/auth.py", "/app/session.py"]


def test_cli_search_no_results():
    with patch("hyperindex.cli.HybridSearchEngine") as MockEngine:
        mock_instance = MagicMock()
        MockEngine.return_value = mock_instance
        mock_instance.search.return_value = []

        result = runner.invoke(app, ["search", "nonexistent"])
        assert result.exit_code == 0
        assert "No matches found for 'nonexistent'" in result.stdout


def test_cli_search_limit_option():
    with patch("hyperindex.cli.HybridSearchEngine") as MockEngine:
        mock_instance = MagicMock()
        MockEngine.return_value = mock_instance
        mock_instance.search.return_value = make_sample_results()[:1]

        result = runner.invoke(app, ["search", "verify_token", "--limit", "5"])
        assert result.exit_code == 0
        mock_instance.search.assert_called_once_with("verify_token", limit=5)


def test_cli_search_tui_flag():
    with patch("hyperindex.cli.HybridSearchEngine") as MockEngine, patch(
        "hyperindex.tui.app.launch_tui"
    ) as mock_launch:
        mock_instance = MagicMock()
        MockEngine.return_value = mock_instance

        result = runner.invoke(app, ["search", "verify_token", "--tui", "--limit", "25"])
        assert result.exit_code == 0
        mock_launch.assert_called_once_with(mock_instance, "verify_token", limit=25)


def test_cli_tui_command():
    with patch("hyperindex.cli.get_engine") as mock_get_engine, patch(
        "hyperindex.tui.app.launch_tui"
    ) as mock_launch:
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        result = runner.invoke(app, ["tui", "find_symbol"])
        assert result.exit_code == 0
        mock_launch.assert_called_once_with(mock_engine, "find_symbol", limit=50)


def test_cli_index_and_watch_commands():
    res_index_help = runner.invoke(app, ["index", "--help"])
    assert res_index_help.exit_code == 0
    assert "Index code files" in res_index_help.stdout

    res_index = runner.invoke(app, ["index", "/tmp/sample"])
    assert res_index.exit_code == 0
    assert "Indexing paths: /tmp/sample" in res_index.stdout

    res_watch_help = runner.invoke(app, ["watch", "--help"])
    assert res_watch_help.exit_code == 0
    assert "Start filesystem watcher daemon" in res_watch_help.stdout

    res_watch = runner.invoke(app, ["watch", "/tmp/sample"])
    assert res_watch.exit_code == 0
    assert "Watching paths for changes: /tmp/sample" in res_watch.stdout


def test_get_editor_command_env():
    with patch.dict(os.environ, {"EDITOR": "code -w"}):
        assert get_editor_command("/path/to/code.py", 42) == [
            "code",
            "-w",
            "-g",
            "/path/to/code.py:42",
        ]

    with patch.dict(os.environ, {"EDITOR": "nvim"}):
        assert get_editor_command("/path/to/code.py", 42) == [
            "nvim",
            "+42",
            "/path/to/code.py",
        ]

    with patch.dict(os.environ, {"EDITOR": "nano"}):
        assert get_editor_command("/path/to/code.py", 42) == [
            "nano",
            "+42",
            "/path/to/code.py",
        ]

    with patch.dict(os.environ, {"EDITOR": "gedit"}):
        assert get_editor_command("/path/to/code.py", 42) == [
            "gedit",
            "/path/to/code.py",
        ]


def test_get_editor_command_fallback():
    with patch.dict(os.environ, {}, clear=True), patch(
        "shutil.which",
        side_effect=lambda x: f"/bin/{x}" if x == "nvim" else None,
    ):
        assert get_editor_command("/path/to/file.py", 10) == [
            "nvim",
            "+10",
            "/path/to/file.py",
        ]

    with patch.dict(os.environ, {}, clear=True), patch("shutil.which", return_value=None):
        assert get_editor_command("/path/to/file.py", 10) == [
            "vi",
            "+10",
            "/path/to/file.py",
        ]


def test_launch_editor():
    with patch("subprocess.run") as mock_run, patch.dict(
        os.environ, {"EDITOR": "nvim"}
    ):
        success = launch_editor("/path/to/file.py", 10)
        assert success is True
        mock_run.assert_called_once_with(["nvim", "+10", "/path/to/file.py"], check=False)

    with patch("subprocess.run", side_effect=OSError("Command failed")), patch.dict(
        os.environ, {"EDITOR": "nvim"}
    ):
        assert launch_editor("/path/to/file.py", 10) is False


def test_copy_to_clipboard():
    with patch(
        "shutil.which", side_effect=lambda x: "/bin/wl-copy" if x == "wl-copy" else None
    ), patch("subprocess.run") as mock_run:
        assert copy_to_clipboard("/path/to/file.py") is True
        mock_run.assert_called_once_with(["wl-copy"], input=b"/path/to/file.py", check=True)

    with patch(
        "shutil.which", side_effect=lambda x: "/bin/xclip" if x == "xclip" else None
    ), patch("subprocess.run") as mock_run:
        assert copy_to_clipboard("/path/to/file.py") is True
        mock_run.assert_called_once_with(
            ["xclip", "-selection", "clipboard"],
            input=b"/path/to/file.py",
            check=True,
        )


@pytest.mark.anyio
async def test_tui_app_navigation_and_actions():
    engine = MagicMock()
    sample_results = make_sample_results()
    engine.search.return_value = sample_results

    app = HyperIndexApp(engine=engine, initial_query="verify", limit=10)

    with patch("hyperindex.tui.app.copy_to_clipboard", return_value=True) as mock_copy, patch(
        "hyperindex.tui.app.launch_editor", return_value=True
    ) as mock_editor:
        async with app.run_test() as pilot:
            # 1. Verify initial state and search execution
            assert len(app.current_results) == 2
            cur = app.get_current_result()
            assert cur is not None
            assert cur.chunk_id == 1

            # 2. Test j (down) navigation
            await pilot.press("j")
            cur = app.get_current_result()
            assert cur is not None
            assert cur.chunk_id == 2

            # 3. Test k (up) navigation
            await pilot.press("k")
            cur = app.get_current_result()
            assert cur is not None
            assert cur.chunk_id == 1

            # 4. Test c (copy path)
            await pilot.press("c")
            mock_copy.assert_called_once_with("/app/auth.py")

            # 5. Test enter (launch editor)
            await pilot.press("enter")
            mock_editor.assert_called_once_with(Path("/app/auth.py"), 15)

            # 6. Test q (quit)
            await pilot.press("q")
            assert not pilot.app.is_running


@pytest.mark.anyio
async def test_tui_app_search_input_and_escape_exit():
    engine = MagicMock()
    sample_results = make_sample_results()

    def mock_search(query, limit=50):
        if query == "session":
            return [sample_results[1]]
        elif query == "none":
            return []
        return sample_results

    engine.search.side_effect = mock_search

    app = HyperIndexApp(engine=engine, initial_query="verify")

    async with app.run_test() as pilot:
        assert len(app.current_results) == 2

        search_input = pilot.app.query_one("#search-input", Input)
        search_input.value = "session"
        await pilot.pause()
        assert len(app.current_results) == 1
        assert app.current_results[0].chunk_id == 2

        search_input.value = "none"
        await pilot.pause()
        assert len(app.current_results) == 0

        # Test escape exit
        await pilot.press("escape")
        assert not pilot.app.is_running


@pytest.mark.anyio
async def test_tui_app_preview_syntax_highlighting(tmp_path):
    # Create an actual test file with 60 lines
    code_file = tmp_path / "service.py"
    lines = [f"def function_{i}():\n    return {i}" for i in range(1, 61)]
    code_file.write_text("\n".join(lines))

    engine = MagicMock()
    engine.search.return_value = [
        SearchResult(
            chunk_id=10,
            path=code_file,
            start_line=25,
            end_line=28,
            symbol="function_13",
            content="def function_13():\n    return 13",
            score=0.999,
        )
    ]

    app = HyperIndexApp(engine=engine, initial_query="function_13")

    async with app.run_test() as pilot:
        preview = pilot.app.query_one("#preview-content", Static)
        renderable = preview.render()._renderable
        assert isinstance(renderable, Syntax)
        # Verify 15 lines context above (25-15 = 10) and below (28+15 = 43)
        assert renderable.line_range == (10, 43)
        # Verify target lines are highlighted
        assert renderable.highlight_lines == {25, 26, 27, 28}
