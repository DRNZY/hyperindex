"""Interactive split-pane Terminal User Interface (TUI) for HyperIndex."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import List, Optional, Union

from rich.syntax import Syntax
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, Static

from hyperindex.search import HybridSearchEngine, SearchResult


def get_editor_command(path: Union[str, Path], line: int = 1) -> List[str]:
    """Resolve the command line arguments to launch an editor at a specific line."""
    path_str = str(path)
    editor_env = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if editor_env:
        parts = shlex.split(editor_env)
        editor_bin = Path(parts[0]).name.lower()
        if "code" in editor_bin or "cursor" in editor_bin:
            return parts + ["-g", f"{path_str}:{line}"]
        elif any(ed in editor_bin for ed in ("nvim", "vim", "vi", "nano", "emacs")):
            return parts + [f"+{line}", path_str]
        else:
            return parts + [path_str]

    # Auto-detect default installed editors
    if shutil.which("code"):
        return ["code", "-g", f"{path_str}:{line}"]
    if shutil.which("nvim"):
        return ["nvim", f"+{line}", path_str]
    if shutil.which("vim"):
        return ["vim", f"+{line}", path_str]
    if shutil.which("nano"):
        return ["nano", f"+{line}", path_str]
    return ["vi", f"+{line}", path_str]


def launch_editor(path: Union[str, Path], line: int = 1) -> bool:
    """Launch editor on target path and line number."""
    cmd = get_editor_command(path, line)
    try:
        subprocess.run(cmd, check=False)
        return True
    except Exception:
        return False


def copy_to_clipboard(text: str) -> bool:
    """Copy given text to clipboard using available system clipboard tools."""
    if shutil.which("wl-copy"):
        try:
            subprocess.run(["wl-copy"], input=text.encode("utf-8"), check=True)
            return True
        except Exception:
            pass
    if shutil.which("xclip"):
        try:
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode("utf-8"),
                check=True,
            )
            return True
        except Exception:
            pass
    if shutil.which("pbcopy"):
        try:
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
            return True
        except Exception:
            pass
    try:
        import pyperclip

        pyperclip.copy(text)
        return True
    except Exception:
        pass
    return False


class ResultItem(ListItem):
    """List item representing a single ranked search result."""

    def __init__(self, result: SearchResult, index: int):
        super().__init__()
        self.result = result
        self.index = index

    def compose(self) -> ComposeResult:
        sym_str = f" [{self.result.symbol}]" if self.result.symbol else ""
        yield Label(f"{self.result.path.name}:{self.result.start_line}{sym_str}", classes="res-header")
        yield Label(
            f"Score: {self.result.score:.4f}  •  {self.result.path}",
            classes="res-meta",
        )


class HyperIndexApp(App):
    """Interactive split-pane TUI for HyperIndex."""

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    #search-box {
        dock: top;
        padding: 1 2;
        height: auto;
    }

    #search-input {
        width: 100%;
        border: tall $primary;
    }

    #main-split {
        height: 1fr;
        layout: horizontal;
    }

    #results-pane {
        width: 45%;
        border-right: solid $primary;
        height: 100%;
        layout: vertical;
    }

    #results-header {
        dock: top;
        background: $panel;
        padding: 0 1;
        text-style: bold;
        color: $accent;
    }

    #results-list {
        height: 1fr;
        overflow-y: auto;
    }

    .res-header {
        text-style: bold;
        color: $accent;
    }

    .res-meta {
        color: $text-muted;
    }

    #preview-pane {
        width: 55%;
        height: 100%;
        layout: vertical;
    }

    #preview-header {
        dock: top;
        background: $panel;
        padding: 0 1;
        text-style: bold;
        color: $accent;
    }

    #preview-scroll {
        height: 1fr;
        overflow-y: auto;
        padding: 0 1;
    }

    #preview-content {
        width: 100%;
    }

    #footer {
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("escape", "quit", "Quit", show=True),
        Binding("j", "cursor_down", "Down", show=True),
        Binding("k", "cursor_up", "Up", show=True),
        Binding("c", "copy_path", "Copy Path", show=True),
        Binding("enter", "open_editor", "Open in Editor", show=True),
        Binding("slash", "focus_search", "Search", show=True),
        Binding("tab", "switch_focus", "Switch Focus", show=True),
    ]

    def __init__(
        self,
        engine: HybridSearchEngine,
        initial_query: str = "",
        limit: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.engine = engine
        self.initial_query = initial_query
        self.limit = limit
        self.current_results: List[SearchResult] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="search-box"):
            yield Input(
                value=self.initial_query,
                placeholder="Type search query (e.g. symbol, function, concept)...",
                id="search-input",
            )
        with Horizontal(id="main-split"):
            with Vertical(id="results-pane"):
                yield Label("Results", id="results-header")
                yield ListView(id="results-list")
            with Vertical(id="preview-pane"):
                yield Label("Preview", id="preview-header")
                with VerticalScroll(id="preview-scroll"):
                    yield Static("Enter a search query to view results.", id="preview-content")
        yield Footer(id="footer")

    def on_mount(self) -> None:
        if self.initial_query:
            self.perform_search(self.initial_query)
            self.query_one("#results-list", ListView).focus()
        else:
            self.query_one("#search-input", Input).focus()

    @on(Input.Changed, "#search-input")
    def on_search_changed(self, event: Input.Changed) -> None:
        self.perform_search(event.value)

    @on(Input.Submitted, "#search-input")
    def on_search_submitted(self, event: Input.Submitted) -> None:
        self.query_one("#results-list", ListView).focus()

    def perform_search(self, query: str) -> None:
        query = query.strip()
        results_list = self.query_one("#results-list", ListView)
        results_list.clear()
        self.current_results = []

        if not query:
            preview = self.query_one("#preview-content", Static)
            preview.update("Enter a search query above...")
            header = self.query_one("#preview-header", Label)
            header.update("Preview")
            res_header = self.query_one("#results-header", Label)
            res_header.update("Results (0)")
            return

        results = self.engine.search(query, limit=self.limit)
        self.current_results = results

        res_header = self.query_one("#results-header", Label)
        res_header.update(f"Results ({len(results)})")

        if not results:
            preview = self.query_one("#preview-content", Static)
            preview.update(f"No results found for '{query}'.")
            header = self.query_one("#preview-header", Label)
            header.update("Preview: No matches")
            return

        for idx, res in enumerate(results):
            results_list.append(ResultItem(res, idx))

        results_list.index = 0
        self.update_preview(results[0])

    @on(ListView.Highlighted, "#results-list")
    def on_result_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None and isinstance(event.item, ResultItem):
            self.update_preview(event.item.result)

    @on(ListView.Selected, "#results-list")
    def on_result_selected(self, event: ListView.Selected) -> None:
        if event.item is not None and isinstance(event.item, ResultItem):
            self.launch_editor_for_result(event.item.result)

    def update_preview(self, result: SearchResult) -> None:
        preview = self.query_one("#preview-content", Static)
        header = self.query_one("#preview-header", Label)
        sym = f" [{result.symbol}]" if result.symbol else ""
        header.update(f"Preview: {result.path.name}:{result.start_line}{sym}")

        target_path = Path(result.path)
        syntax = None
        if target_path.exists() and target_path.is_file():
            try:
                with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                start_line = max(1, result.start_line - 15)
                end_line = min(len(lines), result.end_line + 15)
                highlight = set(range(result.start_line, result.end_line + 1))
                syntax = Syntax.from_path(
                    str(target_path),
                    line_numbers=True,
                    line_range=(start_line, end_line),
                    highlight_lines=highlight,
                    theme="monokai",
                )
            except Exception:
                pass

        if syntax is None:
            try:
                lexer = Syntax.guess_lexer(str(result.path), result.content)
            except Exception:
                lexer = "python"
            syntax = Syntax(
                result.content,
                lexer=lexer or "python",
                line_numbers=True,
                start_line=result.start_line,
                theme="monokai",
            )

        preview.update(syntax)

    def get_current_result(self) -> Optional[SearchResult]:
        results_list = self.query_one("#results-list", ListView)
        if results_list.highlighted_child is not None and isinstance(
            results_list.highlighted_child, ResultItem
        ):
            return results_list.highlighted_child.result
        return None

    def action_cursor_down(self) -> None:
        results_list = self.query_one("#results-list", ListView)
        results_list.action_cursor_down()

    def action_cursor_up(self) -> None:
        results_list = self.query_one("#results-list", ListView)
        results_list.action_cursor_up()

    def action_focus_search(self) -> None:
        self.query_one("#search-input", Input).focus()

    def action_switch_focus(self) -> None:
        search_input = self.query_one("#search-input", Input)
        results_list = self.query_one("#results-list", ListView)
        if search_input.has_focus:
            results_list.focus()
        else:
            search_input.focus()

    def action_copy_path(self) -> None:
        res = self.get_current_result()
        if res:
            copied = copy_to_clipboard(str(res.path))
            if copied:
                self.notify(f"Copied: {res.path}", title="Clipboard")
            else:
                self.notify(f"Could not copy {res.path}", severity="warning")

    def launch_editor_for_result(self, result: SearchResult) -> None:
        try:
            with self.suspend():
                launch_editor(result.path, result.start_line)
        except Exception:
            launch_editor(result.path, result.start_line)

    def action_open_editor(self) -> None:
        res = self.get_current_result()
        if res:
            self.launch_editor_for_result(res)


def launch_tui(
    engine: HybridSearchEngine,
    initial_query: str = "",
    limit: int = 50,
) -> None:
    """Launch the interactive split-pane TUI."""
    app = HyperIndexApp(engine=engine, initial_query=initial_query, limit=limit)
    app.run()
