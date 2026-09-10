"""Tests for inotify filesystem watcher daemon and incremental indexer."""

import os
from pathlib import Path
import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileSystemMovedEvent,
)

from hyperindex.chunker import Chunk
from hyperindex.db import Database
from hyperindex.embedder import Embedder
from hyperindex.search import HybridSearchEngine
from hyperindex.watcher import (
    DebouncedEventHandler,
    IndexWatcher,
    compute_file_hash,
    load_vectors_file,
    match_gitignore_rule,
    parse_gitignore_content,
    save_vectors_file,
    should_ignore_path,
)


def test_should_ignore_common_build_artifacts():
    assert (
        should_ignore_path(Path("/home/darnell/Projects/myrepo/node_modules/pkg/index.js"))
        is True
    )
    assert should_ignore_path(Path("/home/darnell/Projects/myrepo/.git/HEAD")) is True
    assert should_ignore_path(Path("/home/darnell/Projects/myrepo/__pycache__/mod.pyc")) is True
    assert should_ignore_path(Path("/home/darnell/Projects/myrepo/.venv/bin/python")) is True
    assert should_ignore_path(Path("/home/darnell/Projects/myrepo/dist/bundle.js")) is True
    assert should_ignore_path(Path("/home/darnell/Projects/myrepo/target/debug/app")) is True
    assert should_ignore_path(Path("/home/darnell/Projects/myrepo/src/main.py")) is False
    assert should_ignore_path(Path("/home/darnell/Projects/myrepo/index.ts")) is False


def test_should_ignore_binary_extensions():
    assert should_ignore_path(Path("/path/to/asset.png")) is True
    assert should_ignore_path(Path("/path/to/asset.jpg")) is True
    assert should_ignore_path(Path("/path/to/asset.jpeg")) is True
    assert should_ignore_path(Path("/path/to/archive.tar.gz")) is True
    assert should_ignore_path(Path("/path/to/archive.zip")) is True
    assert should_ignore_path(Path("/path/to/video.mp4")) is True
    assert should_ignore_path(Path("/path/to/binary.so")) is True
    assert should_ignore_path(Path("/path/to/database.sqlite3")) is True
    assert should_ignore_path(Path("/path/to/script.py")) is False
    assert should_ignore_path(Path("/path/to/document.md")) is False


def test_parse_gitignore_and_match_rules():
    raw = """
    # Comments should be ignored
    *.log
    !keep.log
    temp/
    /root_only.txt
    docs/*.draft
    """
    rules = parse_gitignore_content(raw)
    assert len(rules) == 5

    # *.log
    assert match_gitignore_rule("debug.log", "*.log", False, False, False) is True
    assert match_gitignore_rule("sub/debug.log", "*.log", False, False, False) is True
    # temp/ (dir only)
    assert match_gitignore_rule("temp/file.txt", "temp", True, False, False) is True
    assert match_gitignore_rule("sub/temp/file.txt", "temp", True, False, False) is True
    # root_only.txt (anchored)
    assert match_gitignore_rule("root_only.txt", "root_only.txt", False, True, False) is True
    assert match_gitignore_rule("sub/root_only.txt", "root_only.txt", False, True, False) is False


def test_should_ignore_with_real_gitignore(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".gitignore").write_text("*.log\n!important.log\nbuild/\n/secret.txt\n")

    # Inside repo
    assert should_ignore_path(root / "app.log", root=root) is True
    assert should_ignore_path(root / "important.log", root=root) is False
    assert should_ignore_path(root / "secret.txt", root=root) is True
    assert should_ignore_path(root / "subdir" / "secret.txt", root=root) is False
    assert should_ignore_path(root / "build" / "app.js", root=root) is True
    assert should_ignore_path(root / "src" / "index.py", root=root) is False


def test_compute_file_hash(tmp_path):
    file_a = tmp_path / "a.py"
    file_a.write_text("print('hello world')")

    file_b = tmp_path / "b.py"
    file_b.write_text("print('hello world')")

    file_c = tmp_path / "c.py"
    file_c.write_text("print('different content')")

    hash_a = compute_file_hash(file_a)
    hash_b = compute_file_hash(file_b)
    hash_c = compute_file_hash(file_c)

    assert hash_a is not None
    assert hash_a == hash_b
    assert hash_a != hash_c

    # Non-existent file returns None
    assert compute_file_hash(tmp_path / "missing.py") is None


def test_save_and_load_vectors_file(tmp_path):
    vec_path = tmp_path / "vectors.bin"

    # 1. Non-existent returns empty
    vecs, cids = load_vectors_file(vec_path)
    assert len(vecs) == 0
    assert cids == []

    # 2. Save populated vectors
    data = np.random.randn(5, 384).astype(np.float32)
    chunk_ids = [101, 102, 103, 104, 105]
    save_vectors_file(vec_path, data, chunk_ids)

    loaded_vecs, loaded_cids = load_vectors_file(vec_path)
    assert loaded_cids == chunk_ids
    assert loaded_vecs.shape == (5, 384)
    assert np.allclose(data, loaded_vecs, atol=1e-6)

    # 3. Save empty vectors
    save_vectors_file(vec_path, np.empty((0, 384), dtype=np.float32), [])
    loaded_empty_vecs, loaded_empty_cids = load_vectors_file(vec_path)
    assert len(loaded_empty_vecs) == 0
    assert loaded_empty_cids == []


def test_index_file_lifecycle_and_hashing(tmp_path):
    db_path = tmp_path / "test.db"
    vec_path = tmp_path / "vectors.bin"
    db = Database(db_path)
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    watcher = IndexWatcher(
        db=db,
        embedder=embedder,
        watch_paths=[tmp_path],
        debounce_delay=0.1,
        vectors_path=vec_path,
    )

    code_file = tmp_path / "module.py"
    code_file.write_text("def calculate_tax(subtotal):\n    return subtotal * 0.2\n")

    # 1. First index creates DB entries and vectors
    indexed = watcher.index_file(code_file)
    assert indexed is True

    # Verify DB FTS
    results = db.search_fts("calculate_tax")
    assert len(results) == 1
    assert "calculate_tax" in results[0]["content"]

    # Verify vector index
    assert len(watcher.chunk_ids) == 1
    assert watcher.vectors.shape == (1, 384)
    assert vec_path.exists()

    # 2. Re-indexing unchanged file skips analysis
    skipped = watcher.index_file(code_file)
    assert skipped is False

    # 3. Modifying file content triggers re-index
    code_file.write_text(
        "def calculate_tax(subtotal):\n    # Updated algorithm\n    return subtotal * 0.25\n"
    )
    reindexed = watcher.index_file(code_file)
    assert reindexed is True

    results_updated = db.search_fts("algorithm")
    assert len(results_updated) == 1
    assert "Updated algorithm" in results_updated[0]["content"]
    assert len(watcher.chunk_ids) == 1

    # 4. Removing file cleans up DB and vectors
    removed = watcher.remove_file(code_file)
    assert removed is True
    assert len(db.search_fts("calculate_tax")) == 0
    assert len(watcher.chunk_ids) == 0
    assert len(watcher.vectors) == 0


def test_index_directory_and_ignored_filtering(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    watcher = IndexWatcher(
        db=db,
        embedder=embedder,
        watch_paths=[tmp_path],
    )

    # Valid files
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "service.py").write_text("def start_service(): pass")
    (src_dir / "client.ts").write_text("export function connect() {}")

    # Ignored directory & files
    nm = tmp_path / "node_modules" / "express"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = {}")

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]")

    (src_dir / "banner.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    count = watcher.index_directory(tmp_path)
    assert count == 2

    assert len(db.search_fts("start_service")) == 1
    assert len(db.search_fts("connect")) == 1
    assert len(db.search_fts("express")) == 0


def test_remove_directory_cascades(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    watcher = IndexWatcher(db=db, embedder=embedder, watch_paths=[tmp_path])

    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "mod1.py").write_text("def function_one(): pass")
    (pkg_dir / "mod2.py").write_text("def function_two(): pass")
    watcher.index_directory(pkg_dir)

    assert len(watcher.chunk_ids) == 2
    assert len(db.search_fts("function_one")) == 1
    assert len(db.search_fts("function_two")) == 1

    # Remove the entire directory
    deleted_count = watcher.remove_directory(pkg_dir)
    assert deleted_count == 2
    assert len(watcher.chunk_ids) == 0
    assert len(db.search_fts("function_one")) == 0
    assert len(db.search_fts("function_two")) == 0


def test_debounced_event_handler_queueing(tmp_path):
    db = MagicMock()
    embedder = MagicMock()
    watcher = IndexWatcher(db=db, embedder=embedder, watch_paths=[tmp_path], debounce_delay=1.0)
    handler = DebouncedEventHandler(watcher)

    file_a = tmp_path / "a.py"
    handler.on_created(FileCreatedEvent(str(file_a)))
    assert file_a in watcher._pending_changes

    handler.on_modified(FileModifiedEvent(str(file_a)))
    assert file_a in watcher._pending_changes

    file_b = tmp_path / "b.py"
    handler.on_deleted(FileDeletedEvent(str(file_b)))
    assert file_b in watcher._pending_deletions

    # Moving a file
    file_src = tmp_path / "src.py"
    file_dest = tmp_path / "dest.py"
    handler.on_moved(FileSystemMovedEvent(str(file_src), str(file_dest)))
    assert file_src in watcher._pending_deletions
    assert file_dest in watcher._pending_changes

    # Ignored paths are not queued
    ignored_file = tmp_path / "node_modules" / "foo.js"
    handler.on_created(FileCreatedEvent(str(ignored_file)))
    assert ignored_file not in watcher._pending_changes


def test_debounce_cooldown_timer_and_batching(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    # Use a 0.2s debounce delay
    watcher = IndexWatcher(
        db=db,
        embedder=embedder,
        watch_paths=[tmp_path],
        debounce_delay=0.2,
    )

    doc = tmp_path / "draft.txt"
    doc.write_text("version 1")

    # Queue changes in rapid succession
    watcher.enqueue_change(doc)
    time.sleep(0.05)
    doc.write_text("version 2")
    watcher.enqueue_change(doc)
    time.sleep(0.05)
    doc.write_text("version 3 (final)")
    watcher.enqueue_change(doc)

    # Before debounce expires (elapsed ~0.1s < 0.2s)
    indexed, deleted = watcher.process_pending_debounced()
    assert indexed == 0
    assert len(db.search_fts("version")) == 0

    # Wait for debounce cooldown to pass (> 0.2s)
    time.sleep(0.25)
    indexed, deleted = watcher.process_pending_debounced()
    assert indexed == 1
    assert deleted == 0

    # Verify only final version was indexed
    results = db.search_fts("version")
    assert len(results) == 1
    assert "version 3 (final)" in results[0]["content"]


def test_flush_immediately_processes_all_pending(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    # Large debounce delay (5s)
    watcher = IndexWatcher(
        db=db,
        embedder=embedder,
        watch_paths=[tmp_path],
        debounce_delay=5.0,
    )

    file_a = tmp_path / "a.py"
    file_a.write_text("def task_a(): pass")
    file_b = tmp_path / "b.py"
    file_b.write_text("def task_b(): pass")

    watcher.enqueue_change(file_a)
    watcher.enqueue_change(file_b)

    # flush() processes immediately without waiting 5 seconds
    indexed, deleted = watcher.flush()
    assert indexed == 2
    assert deleted == 0
    assert len(db.search_fts("task_a")) == 1
    assert len(db.search_fts("task_b")) == 1


def test_hybrid_search_engine_sync_during_incremental_updates(tmp_path):
    db_path = tmp_path / "test.db"
    vec_path = tmp_path / "vectors.bin"
    db = Database(db_path)
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    engine = HybridSearchEngine(db, embedder)

    watcher = IndexWatcher(
        db=db,
        embedder=embedder,
        watch_paths=[tmp_path],
        debounce_delay=0.05,
        vectors_path=vec_path,
        engine=engine,
    )

    code_file = tmp_path / "handler.py"
    code_file.write_text("def authenticate_jwt_token(token):\n    return True\n")
    watcher.index_file(code_file)

    # Engine should see the file both lexically and semantically
    res = engine.search("authenticate_jwt_token", limit=5)
    assert len(res) >= 1
    assert res[0].path == code_file
    assert res[0].symbol == "authenticate_jwt_token"

    # Delete file
    watcher.remove_file(code_file)
    res_after = engine.search("authenticate_jwt_token", limit=5)
    assert len(res_after) == 0


def test_real_inotify_filesystem_watcher_integration(tmp_path):
    db_path = tmp_path / "test.db"
    vec_path = tmp_path / "vectors.bin"
    db = Database(db_path)
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    watch_dir = tmp_path / "watched_repo"
    watch_dir.mkdir()

    # Short debounce delay for snappy test
    watcher = IndexWatcher(
        db=db,
        embedder=embedder,
        watch_paths=[watch_dir],
        debounce_delay=0.1,
        vectors_path=vec_path,
    )

    with watcher:
        # 1. Create a new file in watched directory
        new_file = watch_dir / "worker.py"
        new_file.write_text("def process_queue_jobs():\n    return 42\n")

        # Allow inotify event delivery and debounce delay
        time.sleep(0.35)

        results = db.search_fts("process_queue_jobs")
        assert len(results) == 1

        # 2. Modify existing file
        new_file.write_text("def process_queue_jobs():\n    # Extended pipeline\n    return 99\n")
        time.sleep(0.35)

        results_mod = db.search_fts("Extended")
        assert len(results_mod) == 1

        # 3. Delete file
        new_file.unlink()
        time.sleep(0.35)

        results_del = db.search_fts("process_queue_jobs")
        assert len(results_del) == 0


def test_watcher_run_timeout_and_stop(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    watcher = IndexWatcher(
        db=db,
        embedder=embedder,
        watch_paths=[tmp_path],
        debounce_delay=0.1,
    )

    start = time.time()
    # Run for 0.3s timeout and cleanly exit
    watcher.run(timeout=0.3)
    elapsed = time.time() - start
    assert elapsed >= 0.25
    assert watcher._is_running is False
