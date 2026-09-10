"""End-to-end integration tests for HyperIndex global entrypoint, indexing, and search."""

import json
import shutil
import subprocess
import time
from pathlib import Path
import pytest
from typer.testing import CliRunner

from hyperindex.cli import app, get_engine
from hyperindex.config import get_config
from hyperindex.db import Database
from hyperindex.tui.app import HyperIndexApp

runner = CliRunner()


def test_cli_entrypoint_executable():
    """Verify that hindex executable runs --help cleanly."""
    hindex_bin = shutil.which("hindex")
    assert hindex_bin is not None, "hindex executable not found in PATH"

    result = subprocess.run(
        [hindex_bin, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "HyperIndex: GPU-accelerated hybrid local search" in result.stdout
    assert "search" in result.stdout
    assert "index" in result.stdout
    assert "watch" in result.stdout
    assert "tui" in result.stdout


def test_e2e_index_and_search_workflow(tmp_path, monkeypatch):
    """End-to-end test indexing a sample codebase, searching, and validating ranking and latency."""
    # 1. Setup isolated XDG directories
    share_dir = tmp_path / "share"
    config_dir = tmp_path / "config"
    cache_dir = tmp_path / "cache"

    monkeypatch.setenv("XDG_DATA_HOME", str(share_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_dir))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_dir))

    cfg = get_config()

    # 2. Create sample source files
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    auth_file = src_dir / "auth.py"
    auth_file.write_text(
        "def authenticate_user(username: str, password_hash: str) -> bool:\n"
        "    \"\"\"Validate user credentials against authentication database.\"\"\"\n"
        "    return username == 'admin' and password_hash == 'secret'\n"
    )

    session_file = src_dir / "session.py"
    session_file.write_text(
        "def revoke_session_token(token_id: str) -> None:\n"
        "    \"\"\"Revoke active user session token immediately.\"\"\"\n"
        "    pass\n"
    )

    db_file = src_dir / "database.py"
    db_file.write_text(
        "class ConnectionPool:\n"
        "    \"\"\"Manage pool of persistent SQLite connections.\"\"\"\n"
        "    def get_connection(self):\n"
        "        pass\n"
    )

    ignored_dir = src_dir / "node_modules"
    ignored_dir.mkdir()
    (ignored_dir / "dummy.js").write_text("console.log('should be ignored');")

    # 3. Run indexing via CLI
    index_res = runner.invoke(app, ["index", str(src_dir)])
    assert index_res.exit_code == 0
    assert "Indexing completed: 3 files processed." in index_res.stdout

    # Verify database contents
    db = Database(cfg.db_path)
    db.initialize()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM files;")
        assert cursor.fetchone()[0] == 3
        cursor.execute("SELECT count(*) FROM chunks;")
        assert cursor.fetchone()[0] >= 3

    # 4. Perform search via CLI
    search_res = runner.invoke(app, ["search", "authenticate_user"])
    assert search_res.exit_code == 0
    assert "auth.py" in search_res.stdout
    assert "authenticate_user" in search_res.stdout

    # JSON search output
    json_res = runner.invoke(app, ["search", "revoke_session_token", "--json"])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert len(data) >= 1
    assert data[0]["symbol"] == "revoke_session_token"
    assert "session.py" in data[0]["path"]

    # Paths only output
    paths_res = runner.invoke(app, ["search", "ConnectionPool", "--paths-only"])
    assert paths_res.exit_code == 0
    assert str(db_file) in paths_res.stdout

    # 5. Measure query latency and verify sub-5ms response
    engine = get_engine()

    # Warmup
    engine.search("authenticate_user")

    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        results = engine.search("authenticate_user", limit=5)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)
        assert len(results) >= 1
        assert results[0].symbol == "authenticate_user"

    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 5.0, f"Query latency exceeded 5ms: {avg_latency:.2f}ms"


@pytest.mark.anyio
async def test_e2e_tui_rendering_on_real_index(tmp_path, monkeypatch):
    """Verify interactive split-pane TUI rendering and keyboard navigation against live index."""
    share_dir = tmp_path / "share"
    config_dir = tmp_path / "config"
    cache_dir = tmp_path / "cache"

    monkeypatch.setenv("XDG_DATA_HOME", str(share_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_dir))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_dir))

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    sample_file = src_dir / "service.py"
    sample_file.write_text(
        "def compute_hash(data: bytes) -> str:\n"
        "    \"\"\"Compute SHA-256 hash.\"\"\"\n"
        "    return 'hash'\n"
    )

    runner.invoke(app, ["index", str(src_dir)])

    engine = get_engine()
    tui_app = HyperIndexApp(engine=engine, initial_query="compute_hash", limit=10)
    async with tui_app.run_test() as pilot:
        assert len(tui_app.current_results) == 1
        cur = tui_app.get_current_result()
        assert cur is not None
        assert cur.symbol == "compute_hash"
        await pilot.press("q")
        assert not pilot.app.is_running
