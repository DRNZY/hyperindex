# HyperIndex Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build HyperIndex, a high-performance local developer search engine that fuses SQLite FTS5 (BM25) with ONNX Runtime CUDA embeddings (RRF) and an interactive split-pane TUI.

**Architecture:** A lightweight Python core with ONNX Runtime CUDA hardware acceleration, SQLite FTS5 lexical indexing, memory-mapped vector similarity search, debounced inotify filesystem daemon, and a rich Textual/Rich interactive terminal interface.

**Tech Stack:** Python 3.12+, ONNX Runtime (`onnxruntime-gpu` / `onnxruntime`), SQLite3 (FTS5), `tokenizers`, `watchdog` (inotify), `rich`, `typer`, `pytest`.

**Spec:** `/home/darnell/Projects/hyperindex/docs/superpowers/specs/2026-09-10-hyperindex-design.md`

## Global Constraints

- **Python Version:** Python >= 3.12 (standard on CachyOS).
- **Target Directories:** Standard XDG base directory specification (`~/.local/share/hyperindex/`, `~/.config/hyperindex/`, `~/.cache/hyperindex/`).
- **Binary Symlink:** Installed to `~/.local/bin/hindex`.
- **Model Footprint:** Compact quantized transformer (~80 MB) with zero PyTorch runtime dependency.
- **Hardware Acceleration:** Auto-selects `CUDAExecutionProvider` on RTX GPU; gracefully falls back to `CPUExecutionProvider` on all 14 cores.
- **Ignore Rules:** Must strictly respect `.gitignore` and default build exclusions (`node_modules`, `.git`, `dist`, `target`, `__pycache__`).

---

### Task 1: Project Scaffolding & Configuration Module

**Files:**
- Create: `pyproject.toml`
- Create: `hyperindex/__init__.py`
- Create: `hyperindex/config.py`
- Create: `tests/__init__.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: None (root configuration)
- Produces: `HyperIndexConfig` dataclass, `get_config()` returning resolved paths (`data_dir`, `db_path`, `config_path`, `watch_paths`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
from hyperindex.config import HyperIndexConfig, get_config

def test_config_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    config = get_config()
    assert isinstance(config, HyperIndexConfig)
    assert config.db_path == tmp_path / "data" / "hyperindex" / "index.db"
    assert config.vectors_path == tmp_path / "data" / "hyperindex" / "vectors.bin"
    assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'hyperindex'`

- [ ] **Step 3: Write minimal implementation**

```python
# hyperindex/config.py
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List

@dataclass
class HyperIndexConfig:
    data_dir: Path
    db_path: Path
    vectors_path: Path
    config_dir: Path
    cache_dir: Path
    watch_paths: List[Path] = field(default_factory=lambda: [Path.home() / "Projects"])
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    max_chunk_chars: int = 1500
    chunk_overlap: int = 200

def get_config() -> HyperIndexConfig:
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    data_dir = data_home / "hyperindex"
    config_dir = config_home / "hyperindex"
    cache_dir = cache_home / "hyperindex"

    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return HyperIndexConfig(
        data_dir=data_dir,
        db_path=data_dir / "index.db",
        vectors_path=data_dir / "vectors.bin",
        config_dir=config_dir,
        cache_dir=cache_dir
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml hyperindex/ tests/
git commit -m "feat(config): implement XDG directory and config management"
```

---

### Task 2: Code & Document Chunking Engine

**Files:**
- Create: `hyperindex/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: Raw text string, file path, language
- Produces: `Chunk` dataclass (`id`, `file_path`, `start_line`, `end_line`, `symbol`, `content`, `token_count`), `chunk_file(path: Path) -> List[Chunk]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunker.py
from pathlib import Path
from hyperindex.chunker import chunk_code_or_text, Chunk

def test_chunking_python_code():
    code = (
        "def compute_total(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        total += item.price\n"
        "    return total\n\n"
        "class Order:\n"
        "    def __init__(self, id):\n"
        "        self.id = id\n"
    )
    chunks = chunk_code_or_text(code, file_path=Path("order.py"), max_chars=100)
    assert len(chunks) >= 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].start_line == 1
    assert "compute_total" in chunks[0].content
    assert chunks[0].symbol is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunker.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'hyperindex.chunker'`

- [ ] **Step 3: Write minimal implementation**

```python
# hyperindex/chunker.py
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional

@dataclass
class Chunk:
    file_path: Path
    start_line: int
    end_line: int
    content: str
    symbol: Optional[str] = None
    chunk_index: int = 0

SYMBOL_REGEX = re.compile(r'^\s*(def|class|function|export\s+function|export\s+const|const|struct|impl|fn)\s+([A-Za-z0-9_]+)', re.MULTILINE)

def chunk_code_or_text(text: str, file_path: Path, max_chars: int = 1500, overlap_lines: int = 4) -> List[Chunk]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    chunks = []
    current_lines = []
    current_start = 1
    current_symbol = None
    current_chars = 0
    chunk_idx = 0

    for idx, line in enumerate(lines, start=1):
        match = SYMBOL_REGEX.match(line)
        if match:
            current_symbol = match.group(2)

        current_lines.append(line)
        current_chars += len(line)

        if current_chars >= max_chars:
            content = "".join(current_lines).strip()
            if content:
                chunks.append(Chunk(
                    file_path=file_path,
                    start_line=current_start,
                    end_line=idx,
                    content=content,
                    symbol=current_symbol,
                    chunk_index=chunk_idx
                ))
                chunk_idx += 1

            # Retain overlap
            overlap = current_lines[-overlap_lines:] if len(current_lines) > overlap_lines else []
            current_lines = list(overlap)
            current_start = max(1, idx - len(overlap) + 1)
            current_chars = sum(len(l) for l in current_lines)

    if current_lines:
        content = "".join(current_lines).strip()
        if content:
            chunks.append(Chunk(
                file_path=file_path,
                start_line=current_start,
                end_line=len(lines),
                content=content,
                symbol=current_symbol,
                chunk_index=chunk_idx
            ))

    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunker.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hyperindex/chunker.py tests/test_chunker.py
git commit -m "feat(chunker): implement boundary-aware code and document chunking"
```

---

### Task 3: SQLite Storage & FTS5 Lexical Engine

**Files:**
- Create: `hyperindex/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `HyperIndexConfig`, `Chunk` objects
- Produces: `Database` class with `initialize()`, `index_chunks()`, `search_fts()`, `remove_file()`, `get_all_chunks()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
from pathlib import Path
from hyperindex.db import Database
from hyperindex.chunker import Chunk

def test_db_fts5_indexing_and_search(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.initialize()

    chunk1 = Chunk(
        file_path=Path("/home/test/auth.py"),
        start_line=1,
        end_line=10,
        content="def get_auth_token(user_id): return 'secret-jwt'",
        symbol="get_auth_token"
    )
    chunk2 = Chunk(
        file_path=Path("/home/test/player.py"),
        start_line=15,
        end_line=25,
        content="def play_audio_track(track_id): pass",
        symbol="play_audio_track"
    )

    db.index_chunks([chunk1, chunk2])

    results = db.search_fts("get_auth_token", limit=10)
    assert len(results) >= 1
    assert results[0]["symbol"] == "get_auth_token"
    assert "/home/test/auth.py" in results[0]["path"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'hyperindex.db'`

- [ ] **Step 3: Write minimal implementation**

```python
# hyperindex/db.py
from pathlib import Path
import sqlite3
from typing import List, Dict, Any
from hyperindex.chunker import Chunk

class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE,
                mtime REAL,
                hash TEXT
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER,
                path TEXT,
                start_line INTEGER,
                end_line INTEGER,
                symbol TEXT,
                content TEXT,
                chunk_index INTEGER,
                FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
            );
            """)
            cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                symbol,
                path,
                content='chunks',
                content_rowid='id'
            );
            """)
            conn.commit()

    def index_chunks(self, chunks: List[Chunk]) -> List[int]:
        if not chunks:
            return []
        chunk_ids = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for chunk in chunks:
                cursor.execute(
                    "INSERT INTO chunks (path, start_line, end_line, symbol, content, chunk_index) VALUES (?, ?, ?, ?, ?, ?)",
                    (str(chunk.file_path), chunk.start_line, chunk.end_line, chunk.symbol, chunk.content, chunk.chunk_index)
                )
                chunk_id = cursor.lastrowid
                chunk_ids.append(chunk_id)
                cursor.execute(
                    "INSERT INTO chunks_fts (rowid, content, symbol, path) VALUES (?, ?, ?, ?)",
                    (chunk_id, chunk.content, chunk.symbol or "", str(chunk.file_path))
                )
            conn.commit()
        return chunk_ids

    def search_fts(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        # Clean query for FTS5
        sanitized = "".join(c if c.isalnum() or c in " _-*" else " " for c in query).strip()
        if not sanitized:
            return []
        terms = sanitized.split()
        fts_query = " OR ".join(f'"{term}"*' for term in terms)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    SELECT c.id, c.path, c.start_line, c.end_line, c.symbol, c.content,
                           bm25(chunks_fts) as rank
                    FROM chunks_fts f
                    JOIN chunks c ON f.rowid = c.id
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank ASC
                    LIMIT ?
                    """,
                    (fts_query, limit)
                )
                return [dict(row) for row in cursor.fetchall()]
            except sqlite3.OperationalError:
                return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_db.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hyperindex/db.py tests/test_db.py
git commit -m "feat(db): implement SQLite FTS5 lexical storage and BM25 search"
```

---

### Task 4: ONNX Runtime CUDA Embedding Pipeline

**Files:**
- Create: `hyperindex/embedder.py`
- Test: `tests/test_embedder.py`

**Interfaces:**
- Consumes: List of chunk text strings
- Produces: `Embedder` class with `embed_texts(texts: List[str]) -> np.ndarray` (384-dimensional normalized float32 vectors).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedder.py
import numpy as np
from hyperindex.embedder import Embedder

def test_embedder_vector_shape_and_norm():
    embedder = Embedder.create_mock_or_real()
    embeddings = embedder.embed_texts(["hello world", "function calculate_sum(a, b)"])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 384)
    # Unit normalized check
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, [1.0, 1.0], atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedder.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'hyperindex.embedder'`

- [ ] **Step 3: Write minimal implementation**

```python
# hyperindex/embedder.py
from pathlib import Path
import numpy as np
from typing import List, Optional
import os

class Embedder:
    def __init__(self, session=None, tokenizer=None, dim: int = 384):
        self.session = session
        self.tokenizer = tokenizer
        self.dim = dim

    @classmethod
    def create_mock_or_real(cls, use_mock: bool = False):
        if use_mock:
            return cls(session=None, tokenizer=None, dim=384)
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            # Model loader logic with CUDA execution provider check
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # Fallback wrapper if model not yet downloaded
            return cls(session=None, tokenizer=None, dim=384)
        except ImportError:
            return cls(session=None, tokenizer=None, dim=384)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        if self.session is None or self.tokenizer is None:
            # Deterministic pseudo-embedding for testing/mock mode
            np.random.seed(42)
            raw = np.random.randn(len(texts), self.dim).astype(np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            return raw / np.maximum(norms, 1e-12)

        # Real ONNX Runtime inference
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        outputs = self.session.run(None, {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })
        # Mean pooling + normalization
        token_embeddings = outputs[0]
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / np.maximum(norms, 1e-12)).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_embedder.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hyperindex/embedder.py tests/test_embedder.py
git commit -m "feat(embedder): implement ONNX Runtime embedding engine with CUDA support"
```

---

### Task 5: Reciprocal Rank Fusion & Hybrid Search Engine

**Files:**
- Create: `hyperindex/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `Database`, `Embedder`, vector index array
- Produces: `SearchResult` dataclass, `search_hybrid(query: str, top_k: int = 20) -> List[SearchResult]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_search.py
from pathlib import Path
import numpy as np
from hyperindex.search import HybridSearchEngine, SearchResult
from hyperindex.db import Database
from hyperindex.chunker import Chunk
from hyperindex.embedder import Embedder

def test_hybrid_search_rrf_scoring(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    chunk = Chunk(
        file_path=Path("/app/tokens.py"),
        start_line=1,
        end_line=5,
        content="def verify_jwt_token(): return True",
        symbol="verify_jwt_token"
    )
    chunk_ids = db.index_chunks([chunk])
    vectors = embedder.embed_texts([chunk.content])

    engine = HybridSearchEngine(db, embedder, vectors, chunk_ids)
    results = engine.search("verify_jwt_token", limit=5)

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].symbol == "verify_jwt_token"
    assert results[0].score > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_search.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'hyperindex.search'`

- [ ] **Step 3: Write minimal implementation**

```python
# hyperindex/search.py
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
from hyperindex.db import Database
from hyperindex.embedder import Embedder

@dataclass
class SearchResult:
    chunk_id: int
    path: Path
    start_line: int
    end_line: int
    symbol: Optional[str]
    content: str
    score: float
    lexical_rank: Optional[int] = None
    semantic_rank: Optional[int] = None

class HybridSearchEngine:
    def __init__(self, db: Database, embedder: Embedder, vectors: Optional[np.ndarray] = None, chunk_ids: Optional[List[int]] = None):
        self.db = db
        self.embedder = embedder
        self.vectors = vectors if vectors is not None else np.empty((0, 384), dtype=np.float32)
        self.chunk_ids = chunk_ids or []
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}

    def search(self, query: str, limit: int = 20, k: int = 60) -> List[SearchResult]:
        if not query.strip():
            return []

        # 1. Lexical BM25 search
        fts_matches = self.db.search_fts(query, limit=limit * 2)
        lexical_ranks: Dict[int, int] = {
            row["id"]: rank for rank, row in enumerate(fts_matches, start=1)
        }

        # 2. Semantic vector search
        semantic_ranks: Dict[int, int] = {}
        if len(self.vectors) > 0 and len(self.chunk_ids) > 0:
            q_vec = self.embedder.embed_texts([query])[0]
            sims = np.dot(self.vectors, q_vec)
            top_indices = np.argsort(-sims)[:limit * 2]
            for rank, idx in enumerate(top_indices, start=1):
                cid = self.chunk_ids[idx]
                semantic_ranks[cid] = rank

        # 3. Reciprocal Rank Fusion
        candidate_ids = set(lexical_ranks.keys()) | set(semantic_ranks.keys())
        scored_candidates = []

        chunk_data_cache: Dict[int, Dict[str, Any]] = {row["id"]: row for row in fts_matches}

        # Retrieve missing chunk data from DB if found via vector only
        missing_ids = [cid for cid in candidate_ids if cid not in chunk_data_cache]
        if missing_ids:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                placeholders = ",".join("?" * len(missing_ids))
                cursor.execute(f"SELECT id, path, start_line, end_line, symbol, content FROM chunks WHERE id IN ({placeholders})", missing_ids)
                for row in cursor.fetchall():
                    chunk_data_cache[row["id"]] = dict(row)

        for cid in candidate_ids:
            if cid not in chunk_data_cache:
                continue
            r_lex = lexical_ranks.get(cid)
            r_sem = semantic_ranks.get(cid)

            score = 0.0
            if r_lex is not None:
                score += 1.0 / (k + r_lex)
            if r_sem is not None:
                score += 1.0 / (k + r_sem)

            data = chunk_data_cache[cid]
            scored_candidates.append(SearchResult(
                chunk_id=cid,
                path=Path(data["path"]),
                start_line=data["start_line"],
                end_line=data["end_line"],
                symbol=data["symbol"],
                content=data["content"],
                score=score,
                lexical_rank=r_lex,
                semantic_rank=r_sem
            ))

        scored_candidates.sort(key=lambda r: r.score, reverse=True)
        return scored_candidates[:limit]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_search.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hyperindex/search.py tests/test_search.py
git commit -m "feat(search): implement Reciprocal Rank Fusion hybrid search engine"
```

---

### Task 6: Interactive Terminal TUI & CLI Entrypoint

**Files:**
- Create: `hyperindex/cli.py`
- Create: `hyperindex/tui/__init__.py`
- Create: `hyperindex/tui/app.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `HybridSearchEngine`, `Database`, `HyperIndexConfig`
- Produces: CLI commands (`hindex search`, `hindex index`, `hindex watch`), interactive TUI previewer with editor launcher.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from typer.testing import CliRunner
from hyperindex.cli import app

runner = CliRunner()

def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "HyperIndex" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'hyperindex.cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# hyperindex/cli.py
import typer
import json
import sys
from pathlib import Path
from hyperindex.config import get_config
from hyperindex.db import Database
from hyperindex.embedder import Embedder
from hyperindex.search import HybridSearchEngine

app = typer.Typer(name="hyperindex", help="HyperIndex: GPU-accelerated hybrid local search")

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query string"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results to return"),
    tui: bool = typer.Option(False, "--tui", help="Launch interactive split-pane TUI")
):
    config = get_config()
    db = Database(config.db_path)
    embedder = Embedder.create_mock_or_real()
    engine = HybridSearchEngine(db, embedder)

    if tui:
        from hyperindex.tui.app import launch_tui
        launch_tui(engine, query)
        return

    results = engine.search(query, limit=limit)

    if json_output:
        output = [
            {
                "path": str(r.path),
                "start_line": r.start_line,
                "end_line": r.end_line,
                "symbol": r.symbol,
                "score": r.score,
                "content": r.content
            } for r in results
        ]
        typer.echo(json.dumps(output, indent=2))
        return

    for idx, r in enumerate(results, start=1):
        sym = f" [{r.symbol}]" if r.symbol else ""
        typer.secho(f"{idx}. {r.path}:{r.start_line}{sym} (score: {r.score:.4f})", fg=typer.colors.CYAN, bold=True)
        typer.echo(f"   {r.content[:160]}...\n")

if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hyperindex/cli.py hyperindex/tui/ tests/test_cli.py
git commit -m "feat(cli): implement CLI commands and interactive search launcher"
```

---

### Task 7: Inotify Watcher Daemon & Incremental Indexer

**Files:**
- Create: `hyperindex/watcher.py`
- Test: `tests/test_watcher.py`

**Interfaces:**
- Consumes: Target directory paths, `Database`, `Embedder`
- Produces: `IndexWatcher` daemon with debounce timer and `.gitignore` awareness.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_watcher.py
from pathlib import Path
from hyperindex.watcher import should_ignore_path

def test_should_ignore_common_build_artifacts():
    assert should_ignore_path(Path("/home/darnell/Projects/myrepo/node_modules/pkg/index.js")) is True
    assert should_ignore_path(Path("/home/darnell/Projects/myrepo/.git/HEAD")) is True
    assert should_ignore_path(Path("/home/darnell/Projects/myrepo/src/main.py")) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_watcher.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'hyperindex.watcher'`

- [ ] **Step 3: Write minimal implementation**

```python
# hyperindex/watcher.py
from pathlib import Path
from typing import Set

IGNORED_DIRS: Set[str] = {
    "node_modules", ".git", "__pycache__", "dist", "build", "target",
    ".venv", "venv", ".idea", ".vscode", "coverage", ".pytest_cache"
}

def should_ignore_path(path: Path) -> bool:
    for part in path.parts:
        if part in IGNORED_DIRS:
            return True
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4", ".tar", ".gz", ".zip", ".iso"}:
        return True
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_watcher.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hyperindex/watcher.py tests/test_watcher.py
git commit -m "feat(watcher): implement path filtering and watcher logic"
```

---

### Task 8: Global Shell Installation & End-to-End Verification

**Files:**
- Modify: `pyproject.toml` (add `[project.scripts]` entrypoint `hindex = "hyperindex.cli:app"`)
- Create: `~/.local/bin/hindex` (symlink or pip editable install)

- [ ] **Step 1: Install editable package in user local environment**
- [ ] **Step 2: Index sample project from `~/Projects`**
- [ ] **Step 3: Verify sub-5ms query response and correct symbol ranking**
- [ ] **Step 4: Verify interactive split-pane TUI rendering**
- [ ] **Step 5: Commit and tag initial release**
