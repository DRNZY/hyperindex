from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Union

from hyperindex.chunker import Chunk


class Database:
    def __init__(self, db_path: Union[Path, str]):
        self.db_path = Path(db_path)

    def get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def initialize(self) -> None:
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode = WAL;")
                cursor.execute("PRAGMA foreign_keys = ON;")
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        path TEXT UNIQUE,
                        mtime REAL,
                        hash TEXT
                    );
                    """
                )
                cursor.execute(
                    """
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
                    """
                )
                cursor.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        content,
                        symbol,
                        path,
                        content='chunks',
                        content_rowid='id'
                    );
                    """
                )
        finally:
            conn.close()

    def index_chunks(self, chunks: List[Chunk]) -> List[int]:
        if not chunks:
            return []
        chunk_ids: List[int] = []
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                for chunk in chunks:
                    cursor.execute(
                        """
                        INSERT INTO chunks (path, start_line, end_line, symbol, content, chunk_index)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(chunk.file_path),
                            chunk.start_line,
                            chunk.end_line,
                            chunk.symbol,
                            chunk.content,
                            chunk.chunk_index,
                        ),
                    )
                    chunk_id = cursor.lastrowid
                    if chunk_id is not None:
                        chunk_ids.append(chunk_id)
                        cursor.execute(
                            """
                            INSERT INTO chunks_fts (rowid, content, symbol, path)
                            VALUES (?, ?, ?, ?)
                            """,
                            (chunk_id, chunk.content, chunk.symbol or "", str(chunk.file_path)),
                        )
            return chunk_ids
        finally:
            conn.close()

    def search_fts(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        # Clean query for FTS5
        sanitized = "".join(c if c.isalnum() or c in " _-*" else " " for c in query).strip()
        if not sanitized:
            return []
        terms = [t for t in sanitized.split() if t.strip("*")]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term}"*' for term in terms)

        conn = self.get_connection()
        try:
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
                    (fts_query, limit),
                )
                return [dict(row) for row in cursor.fetchall()]
            except sqlite3.OperationalError:
                return []
        finally:
            conn.close()

    def remove_file(self, file_path: Union[Path, str]) -> None:
        path_str = str(file_path)
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, content, symbol, path FROM chunks WHERE path = ?",
                    (path_str,),
                )
                rows = cursor.fetchall()
                for row in rows:
                    cursor.execute(
                        """
                        INSERT INTO chunks_fts (chunks_fts, rowid, content, symbol, path)
                        VALUES ('delete', ?, ?, ?, ?)
                        """,
                        (
                            row["id"],
                            row["content"],
                            row["symbol"] or "",
                            row["path"],
                        ),
                    )
                cursor.execute("DELETE FROM chunks WHERE path = ?", (path_str,))
                cursor.execute("DELETE FROM files WHERE path = ?", (path_str,))
        finally:
            conn.close()

    def get_all_chunks(self) -> List[Chunk]:
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, path, start_line, end_line, symbol, content, chunk_index
                FROM chunks
                ORDER BY id ASC
                """
            )
            return [
                Chunk(
                    id=row["id"],
                    file_path=Path(row["path"]),
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    symbol=row["symbol"],
                    content=row["content"],
                    chunk_index=row["chunk_index"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()
