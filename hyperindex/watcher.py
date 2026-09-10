"""Inotify-backed filesystem watcher daemon and incremental indexer."""

import fnmatch
import hashlib
import os
from pathlib import Path
import signal
import threading
import time
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from watchdog.events import FileSystemEvent, FileSystemEventHandler, FileSystemMovedEvent
from watchdog.observers import Observer

from hyperindex.chunker import chunk_file
from hyperindex.db import Database
from hyperindex.embedder import Embedder
from hyperindex.search import HybridSearchEngine

IGNORED_DIRS: Set[str] = {
    "node_modules",
    ".git",
    "__pycache__",
    "dist",
    "build",
    "target",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".tox",
    "env",
}

IGNORED_EXTENSIONS: Set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".svg",
    ".mp4",
    ".webm",
    ".mkv",
    ".avi",
    ".mov",
    ".tar",
    ".gz",
    ".zip",
    ".iso",
    ".7z",
    ".rar",
    ".bz2",
    ".xz",
    ".pdf",
    ".bin",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".pyc",
    ".wasm",
    ".whl",
    ".lock",
    ".sqlite3",
    ".db",
}

# Cache for loaded gitignore patterns: (path_str, mtime) -> rules
_GITIGNORE_CACHE: Dict[Tuple[str, float], List[Tuple[str, bool, bool, bool]]] = {}


def parse_gitignore_content(content: str) -> List[Tuple[str, bool, bool, bool]]:
    """Parse gitignore content into a list of (pattern, is_dir_only, is_negated, is_anchored)."""
    rules: List[Tuple[str, bool, bool, bool]] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        is_negated = False
        if line.startswith("!"):
            is_negated = True
            line = line[1:].strip()
            if not line:
                continue

        is_dir_only = False
        if line.endswith("/"):
            is_dir_only = True
            line = line[:-1].rstrip()
            if not line:
                continue

        is_anchored = False
        if line.startswith("/"):
            is_anchored = True
            line = line[1:].lstrip()
        elif "/" in line:
            # Pattern containing slash is relative to base dir
            is_anchored = True

        rules.append((line, is_dir_only, is_negated, is_anchored))
    return rules


def load_gitignore_rules(gitignore_file: Path) -> List[Tuple[str, bool, bool, bool]]:
    """Load and parse gitignore rules with mtime-based caching."""
    try:
        stat = gitignore_file.stat()
        cache_key = (str(gitignore_file), stat.st_mtime)
        if cache_key in _GITIGNORE_CACHE:
            return _GITIGNORE_CACHE[cache_key]

        content = gitignore_file.read_text(encoding="utf-8", errors="replace")
        rules = parse_gitignore_content(content)
        _GITIGNORE_CACHE[cache_key] = rules
        return rules
    except Exception:
        return []


def match_gitignore_rule(
    rel_path_str: str,
    pattern: str,
    is_dir_only: bool,
    is_anchored: bool,
    is_dir: bool = False,
) -> bool:
    """Check if a relative path matches a specific gitignore rule."""
    parts = rel_path_str.split("/")
    if is_dir_only and not is_dir:
        # Check if any parent directory component matches
        if is_anchored:
            return rel_path_str.startswith(pattern + "/")
        return pattern in parts[:-1]

    if is_anchored or "/" in pattern:
        return fnmatch.fnmatch(rel_path_str, pattern) or fnmatch.fnmatch(
            rel_path_str, f"{pattern}/*"
        )

    filename = parts[-1]
    return (
        fnmatch.fnmatch(filename, pattern)
        or fnmatch.fnmatch(rel_path_str, pattern)
        or fnmatch.fnmatch(rel_path_str, f"*/{pattern}")
        or fnmatch.fnmatch(rel_path_str, f"*/{pattern}/*")
    )


def is_gitignored(path: Path, root: Optional[Path] = None) -> bool:
    """Check if path matches any .gitignore files located up to root or repository root."""
    try:
        resolved_path = path.resolve()
    except Exception:
        resolved_path = path

    # Determine directories to check for .gitignore
    target_dir = resolved_path if resolved_path.is_dir() else resolved_path.parent
    check_dirs: List[Path] = []
    curr = target_dir

    limit_dir = root.resolve() if root is not None else None

    while True:
        check_dirs.append(curr)
        if limit_dir is not None and curr == limit_dir:
            break
        if (curr / ".git").exists():
            break
        if curr.parent == curr:
            break
        curr = curr.parent

    # Evaluate rules from highest root to deepest child
    check_dirs.reverse()
    is_ignored = False

    for base_dir in check_dirs:
        gi_path = base_dir / ".gitignore"
        if not gi_path.is_file():
            continue
        rules = load_gitignore_rules(gi_path)
        try:
            rel = resolved_path.relative_to(base_dir)
            rel_str = str(rel).replace("\\", "/")
        except ValueError:
            continue

        for pattern, is_dir_only, is_negated, is_anchored in rules:
            if match_gitignore_rule(
                rel_str,
                pattern,
                is_dir_only=is_dir_only,
                is_anchored=is_anchored,
                is_dir=resolved_path.is_dir(),
            ):
                is_ignored = not is_negated

    return is_ignored


def should_ignore_path(
    path: Path,
    root: Optional[Path] = None,
    check_gitignore: bool = True,
) -> bool:
    """Determine whether a path should be ignored by the indexer/watcher."""
    # 1. Ignored directory names in path parts
    for part in path.parts:
        if part in IGNORED_DIRS:
            return True

    # 2. Ignored binary file extensions
    if path.suffix.lower() in IGNORED_EXTENSIONS:
        return True

    # 3. .gitignore awareness
    if check_gitignore:
        try:
            if is_gitignored(path, root=root):
                return True
        except Exception:
            pass

    return False


def compute_file_hash(path: Path) -> Optional[str]:
    """Compute fast cryptographic hash of file content for change detection."""
    try:
        h = hashlib.blake2b()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, PermissionError, FileNotFoundError):
        return None


def save_vectors_file(
    path: Union[str, Path],
    vectors: np.ndarray,
    chunk_ids: List[int],
) -> None:
    """Persist dense float32 vector array and chunk ID lookup to binary file atomically."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    num_chunks = len(chunk_ids)
    dim = (
        vectors.shape[1]
        if (vectors is not None and vectors.ndim == 2 and vectors.shape[0] > 0)
        else 384
    )
    tmp_path = p.with_name(f".{p.name}.tmp")
    try:
        with open(tmp_path, "wb") as f:
            np.array([num_chunks, dim], dtype=np.int32).tofile(f)
            if num_chunks > 0:
                np.array(chunk_ids, dtype=np.int64).tofile(f)
                vectors.astype(np.float32).tofile(f)
        os.replace(tmp_path, p)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def load_vectors_file(path: Union[str, Path]) -> Tuple[np.ndarray, List[int]]:
    """Load dense float32 vector array and chunk IDs from binary file."""
    p = Path(path)
    if not p.exists():
        return np.empty((0, 384), dtype=np.float32), []

    try:
        if p.stat().st_size < 8:
            return np.empty((0, 384), dtype=np.float32), []

        with open(p, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=2)
            if len(header) < 2:
                return np.empty((0, 384), dtype=np.float32), []
            num_chunks, dim = int(header[0]), int(header[1])
            if num_chunks <= 0:
                return np.empty((0, dim), dtype=np.float32), []

            chunk_ids = np.fromfile(f, dtype=np.int64, count=num_chunks).tolist()
            vectors = np.fromfile(f, dtype=np.float32, count=num_chunks * dim).reshape(
                (num_chunks, dim)
            )
            return vectors, chunk_ids
    except Exception:
        return np.empty((0, 384), dtype=np.float32), []


class DebouncedEventHandler(FileSystemEventHandler):
    """Watches filesystem events and queues them with debouncing to avoid redundant indexing."""

    def __init__(self, watcher: "IndexWatcher"):
        super().__init__()
        self.watcher = watcher

    def on_created(self, event: FileSystemEvent) -> None:
        path = Path(event.src_path)
        if event.is_directory:
            return
        if not self.watcher.should_ignore_path(path):
            self.watcher.enqueue_change(path)

    def on_modified(self, event: FileSystemEvent) -> None:
        path = Path(event.src_path)
        if event.is_directory:
            return
        if not self.watcher.should_ignore_path(path):
            self.watcher.enqueue_change(path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        path = Path(event.src_path)
        if event.is_directory:
            if not self.watcher.should_ignore_path(path):
                self.watcher.enqueue_directory_deletion(path)
            return
        if not self.watcher.should_ignore_path(path):
            self.watcher.enqueue_deletion(path)

    def on_moved(self, event: FileSystemMovedEvent) -> None:
        src = Path(event.src_path)
        dest = Path(event.dest_path)
        if event.is_directory:
            if not self.watcher.should_ignore_path(src):
                self.watcher.enqueue_directory_deletion(src)
            if not self.watcher.should_ignore_path(dest):
                self.watcher.enqueue_change(dest)
            return
        if not self.watcher.should_ignore_path(src):
            self.watcher.enqueue_deletion(src)
        if not self.watcher.should_ignore_path(dest):
            self.watcher.enqueue_change(dest)


class IndexWatcher:
    """Inotify-backed filesystem watcher daemon with debounce timer and incremental indexer."""

    def __init__(
        self,
        db: Database,
        embedder: Optional[Embedder] = None,
        watch_paths: Optional[List[Union[str, Path]]] = None,
        debounce_delay: float = 1.5,
        vectors_path: Optional[Union[str, Path]] = None,
        engine: Optional[HybridSearchEngine] = None,
    ):
        self.db = db
        self.embedder = embedder or Embedder.create_mock_or_real()
        self.debounce_delay = debounce_delay
        self.engine = engine

        if watch_paths is not None:
            self.watch_paths = [Path(p).resolve() for p in watch_paths]
        else:
            try:
                from hyperindex.config import get_config

                self.watch_paths = [Path(p).resolve() for p in get_config().watch_paths]
            except Exception:
                self.watch_paths = []

        if vectors_path is not None:
            self.vectors_path: Optional[Path] = Path(vectors_path).resolve()
        else:
            self.vectors_path = None

        self.dim = getattr(self.embedder, "dim", 384)
        if self.vectors_path and self.vectors_path.exists():
            loaded_vecs, loaded_cids = load_vectors_file(self.vectors_path)
            conn = self.db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM chunks")
                existing_cids = {row["id"] for row in cursor.fetchall()}
            finally:
                conn.close()

            if existing_cids and len(loaded_cids) > 0:
                valid_indices = [
                    idx for idx, cid in enumerate(loaded_cids) if cid in existing_cids
                ]
                if len(valid_indices) == len(loaded_cids):
                    self.vectors = loaded_vecs
                    self.chunk_ids = loaded_cids
                elif len(valid_indices) > 0:
                    self.vectors = loaded_vecs[valid_indices]
                    self.chunk_ids = [loaded_cids[i] for i in valid_indices]
                else:
                    self.vectors = np.empty((0, self.dim), dtype=np.float32)
                    self.chunk_ids = []
            else:
                self.vectors = np.empty((0, self.dim), dtype=np.float32)
                self.chunk_ids = []
        else:
            self.vectors = np.empty((0, self.dim), dtype=np.float32)
            self.chunk_ids = []

        if self.engine is not None:
            self.engine.update_index(self.vectors, self.chunk_ids)

        self._lock = threading.Lock()
        self._pending_changes: Dict[Path, float] = {}
        self._pending_deletions: Dict[Path, float] = {}
        self._pending_dir_deletions: Dict[Path, float] = {}

        self._observer: Optional[Observer] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False

    def should_ignore_path(self, path: Path) -> bool:
        """Check if path is ignored using global rules and watcher watch roots."""
        for root in self.watch_paths:
            try:
                path.relative_to(root)
                return should_ignore_path(path, root=root)
            except ValueError:
                continue
        return should_ignore_path(path)

    def _remove_file_from_vectors(self, path: Union[Path, str], save: bool = True) -> List[int]:
        """Remove in-memory and persisted vectors associated with chunks of path."""
        path_str = str(path)
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM chunks WHERE path = ?", (path_str,))
            old_cids = {row["id"] for row in cursor.fetchall()}
        finally:
            conn.close()

        if not old_cids or len(self.chunk_ids) == 0:
            return []

        keep_indices = [idx for idx, cid in enumerate(self.chunk_ids) if cid not in old_cids]
        if len(keep_indices) == len(self.chunk_ids):
            return []

        if len(keep_indices) == 0:
            self.vectors = np.empty((0, self.dim), dtype=np.float32)
            self.chunk_ids = []
        else:
            self.vectors = self.vectors[keep_indices]
            self.chunk_ids = [self.chunk_ids[i] for i in keep_indices]

        if save and self.vectors_path:
            save_vectors_file(self.vectors_path, self.vectors, self.chunk_ids)

        if self.engine is not None:
            self.engine.update_index(self.vectors, self.chunk_ids)

        return list(old_cids)

    def remove_file(self, path: Path, save: bool = True) -> bool:
        """Remove a file from SQLite chunks, FTS5 index, and vector embeddings."""
        path_str = str(path.resolve() if path.is_absolute() else path)
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM files WHERE path = ?", (path_str,))
            row = cursor.fetchone()
            if row is None:
                cursor.execute("SELECT id FROM chunks WHERE path = ? LIMIT 1", (path_str,))
                if cursor.fetchone() is None:
                    return False
        finally:
            conn.close()

        self._remove_file_from_vectors(path_str, save=save)
        self.db.remove_file(path_str)
        return True

    def remove_directory(self, dir_path: Path, save: bool = True) -> int:
        """Remove all indexed files residing within the given directory."""
        dir_str = str(dir_path.resolve() if dir_path.is_absolute() else dir_path)
        if not dir_str.endswith("/"):
            dir_prefix = dir_str + "/"
        else:
            dir_prefix = dir_str

        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT path FROM files WHERE path = ? OR path LIKE ?",
                (dir_str, f"{dir_prefix}%"),
            )
            paths = {Path(row["path"]) for row in cursor.fetchall()}
            cursor.execute(
                "SELECT DISTINCT path FROM chunks WHERE path = ? OR path LIKE ?",
                (dir_str, f"{dir_prefix}%"),
            )
            for row in cursor.fetchall():
                paths.add(Path(row["path"]))
        finally:
            conn.close()

        count = 0
        for p in paths:
            if self.remove_file(p, save=False):
                count += 1

        if save and count > 0 and self.vectors_path:
            save_vectors_file(self.vectors_path, self.vectors, self.chunk_ids)

        return count

    def index_file(self, path: Path, save: bool = True) -> bool:
        """Index or incrementally re-index a single file into SQLite FTS5 and vector index."""
        try:
            resolved_path = path.resolve()
        except Exception:
            resolved_path = path

        if self.should_ignore_path(resolved_path):
            return False

        if not resolved_path.is_file():
            return False

        file_hash = compute_file_hash(resolved_path)
        if file_hash is None:
            return False

        path_str = str(resolved_path)

        # Check existing hash in files table for early exit
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, hash, mtime FROM files WHERE path = ?", (path_str,))
            row = cursor.fetchone()
            if row is not None and row["hash"] == file_hash:
                # Content has not changed
                return False
        finally:
            conn.close()

        # Remove previous chunks & vectors before re-indexing
        self._remove_file_from_vectors(path_str, save=save)
        self.db.remove_file(path_str)

        chunks = chunk_file(resolved_path)
        try:
            mtime = resolved_path.stat().st_mtime
        except OSError:
            mtime = time.time()

        conn = self.db.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO files (path, mtime, hash) VALUES (?, ?, ?)",
                    (path_str, mtime, file_hash),
                )
        finally:
            conn.close()

        if not chunks:
            return True

        chunk_ids = self.db.index_chunks(chunks)

        if self.embedder is not None and chunk_ids:
            texts = [c.content for c in chunks]
            new_vecs = self.embedder.embed_texts(texts)
            if len(new_vecs) > 0 and len(chunk_ids) > 0:
                if len(self.vectors) == 0:
                    self.vectors = np.asarray(new_vecs, dtype=np.float32)
                    self.chunk_ids = list(chunk_ids)
                else:
                    self.vectors = np.vstack([self.vectors, new_vecs])
                    self.chunk_ids.extend(chunk_ids)

                if save and self.vectors_path:
                    save_vectors_file(self.vectors_path, self.vectors, self.chunk_ids)

                if self.engine is not None:
                    self.engine.update_index(self.vectors, self.chunk_ids)

        return True

    def index_directory(
        self,
        dir_path: Path,
        save: bool = True,
        on_progress: Optional[Callable[[Path, int], None]] = None,
    ) -> int:
        """Recursively scan and index non-ignored files in directory. Returns count of files indexed."""
        try:
            resolved_dir = dir_path.resolve()
        except Exception:
            resolved_dir = dir_path

        if not resolved_dir.exists() or not resolved_dir.is_dir():
            return 0

        count = 0
        for root, dirs, files in os.walk(resolved_dir):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if not self.should_ignore_path(root_path / d)]

            for file_name in files:
                file_path = root_path / file_name
                if not self.should_ignore_path(file_path):
                    if self.index_file(file_path, save=False):
                        count += 1
                    if on_progress:
                        on_progress(file_path, count)

        if save and count > 0 and self.vectors_path:
            save_vectors_file(self.vectors_path, self.vectors, self.chunk_ids)

        return count

    def enqueue_change(self, path: Path) -> None:
        """Queue a file creation or modification event with timestamp."""
        with self._lock:
            self._pending_changes[path] = time.time()
            self._pending_deletions.pop(path, None)

    def enqueue_deletion(self, path: Path) -> None:
        """Queue a file deletion event with timestamp."""
        with self._lock:
            self._pending_deletions[path] = time.time()
            self._pending_changes.pop(path, None)

    def enqueue_directory_deletion(self, dir_path: Path) -> None:
        """Queue a directory deletion event."""
        resolved_dir = dir_path.resolve() if dir_path.is_absolute() else dir_path
        with self._lock:
            self._pending_dir_deletions[resolved_dir] = time.time()
            for p in list(self._pending_changes.keys()):
                try:
                    if p.is_relative_to(resolved_dir):
                        self._pending_changes.pop(p, None)
                except (ValueError, AttributeError):
                    dir_str = str(resolved_dir)
                    if not dir_str.endswith("/"):
                        dir_prefix = dir_str + "/"
                    else:
                        dir_prefix = dir_str
                    if str(p).startswith(dir_prefix):
                        self._pending_changes.pop(p, None)

    def process_pending_debounced(self) -> Tuple[int, int]:
        """Process pending changes and deletions whose debounce timer has elapsed."""
        now = time.time()
        changes_to_process: List[Path] = []
        deletions_to_process: List[Path] = []
        dir_deletions_to_process: List[Path] = []

        with self._lock:
            for path, ts in list(self._pending_changes.items()):
                if now - ts >= self.debounce_delay:
                    changes_to_process.append(path)
                    del self._pending_changes[path]

            for path, ts in list(self._pending_deletions.items()):
                if now - ts >= self.debounce_delay:
                    deletions_to_process.append(path)
                    del self._pending_deletions[path]

            for path, ts in list(self._pending_dir_deletions.items()):
                if now - ts >= self.debounce_delay:
                    dir_deletions_to_process.append(path)
                    del self._pending_dir_deletions[path]

        num_deleted = 0
        for dir_path in dir_deletions_to_process:
            num_deleted += self.remove_directory(dir_path, save=False)

        for path in deletions_to_process:
            if self.remove_file(path, save=False):
                num_deleted += 1

        num_indexed = 0
        for path in changes_to_process:
            if path.exists():
                if path.is_file():
                    if self.index_file(path, save=False):
                        num_indexed += 1
                elif path.is_dir():
                    num_indexed += self.index_directory(path, save=False)
            else:
                if self.remove_file(path, save=False):
                    num_deleted += 1

        if (num_deleted > 0 or num_indexed > 0) and self.vectors_path:
            save_vectors_file(self.vectors_path, self.vectors, self.chunk_ids)

        return num_indexed, num_deleted

    def flush(self) -> Tuple[int, int]:
        """Immediately process all pending events without waiting for debounce timer."""
        with self._lock:
            changes_to_process = list(self._pending_changes.keys())
            self._pending_changes.clear()
            deletions_to_process = list(self._pending_deletions.keys())
            self._pending_deletions.clear()
            dir_deletions_to_process = list(self._pending_dir_deletions.keys())
            self._pending_dir_deletions.clear()

        num_deleted = 0
        for dir_path in dir_deletions_to_process:
            num_deleted += self.remove_directory(dir_path, save=False)

        for path in deletions_to_process:
            if self.remove_file(path, save=False):
                num_deleted += 1

        num_indexed = 0
        for path in changes_to_process:
            if path.exists():
                if path.is_file():
                    if self.index_file(path, save=False):
                        num_indexed += 1
                elif path.is_dir():
                    num_indexed += self.index_directory(path, save=False)
            else:
                if self.remove_file(path, save=False):
                    num_deleted += 1

        if (num_deleted > 0 or num_indexed > 0) and self.vectors_path:
            save_vectors_file(self.vectors_path, self.vectors, self.chunk_ids)

        return num_indexed, num_deleted

    def _worker_loop(self) -> None:
        """Background thread worker to periodically process debounced events."""
        poll_interval = max(0.05, min(0.2, self.debounce_delay / 4))
        while not self._stop_event.is_set():
            try:
                self.process_pending_debounced()
            except Exception:
                pass
            self._stop_event.wait(poll_interval)

    def start(self) -> None:
        """Start watchdog filesystem observers and debounce processing thread."""
        if self._is_running:
            return

        self._stop_event.clear()
        self._observer = Observer()
        handler = DebouncedEventHandler(self)

        scheduled_any = False
        for watch_path in self.watch_paths:
            try:
                p = Path(watch_path).resolve()
                if p.exists() and p.is_dir():
                    self._observer.schedule(handler, str(p), recursive=True)
                    scheduled_any = True
            except Exception:
                continue

        if scheduled_any:
            self._observer.start()

        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="HyperIndexWatcherWorker"
        )
        self._worker_thread.start()
        self._is_running = True

    def stop(self) -> None:
        """Stop watcher observers and worker thread, flushing pending edits."""
        if not self._is_running:
            return

        self._stop_event.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=2.0)
            except Exception:
                pass
            self._observer = None

        if self._worker_thread is not None:
            try:
                self._worker_thread.join(timeout=2.0)
            except Exception:
                pass
            self._worker_thread = None

        self._is_running = False
        # Commit any unhandled pending events
        self.flush()

    def run(self, timeout: Optional[float] = None) -> None:
        """Block execution running the daemon until KeyboardInterrupt, SIGTERM, or timeout."""
        orig_sigterm = None
        try:
            def _handle_sigterm(signum, frame):
                self.stop()
                raise SystemExit(0)

            orig_sigterm = signal.signal(signal.SIGTERM, _handle_sigterm)
        except (ValueError, AttributeError):
            orig_sigterm = None

        self.start()
        start_time = time.time()
        try:
            while self._is_running and not self._stop_event.is_set():
                if timeout is not None and (time.time() - start_time) >= timeout:
                    break
                self._stop_event.wait(0.2)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            self.stop()
            if orig_sigterm is not None:
                try:
                    signal.signal(signal.SIGTERM, orig_sigterm)
                except (ValueError, AttributeError):
                    pass

    def __enter__(self) -> "IndexWatcher":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
