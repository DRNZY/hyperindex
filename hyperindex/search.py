"""Reciprocal Rank Fusion & Hybrid Search Engine combining lexical and semantic search."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from hyperindex.db import Database
from hyperindex.embedder import Embedder


@dataclass
class SearchResult:
    """Individual search result item with ranked fusion metadata."""

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
    """Combines SQLite FTS5 lexical matching with dense vector similarity via RRF."""

    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        vectors: Optional[np.ndarray] = None,
        chunk_ids: Optional[List[int]] = None,
    ):
        self.db = db
        self.embedder = embedder
        self.dim = getattr(embedder, "dim", 384)
        if vectors is not None and len(vectors) > 0:
            self.vectors = np.asarray(vectors, dtype=np.float32)
        else:
            self.vectors = np.empty((0, self.dim), dtype=np.float32)
        self.chunk_ids: List[int] = list(chunk_ids) if chunk_ids is not None else []
        self.id_to_idx: Dict[int, int] = {cid: idx for idx, cid in enumerate(self.chunk_ids)}

    def update_index(
        self,
        vectors: Optional[np.ndarray],
        chunk_ids: Optional[List[int]],
    ) -> None:
        """Update in-memory vector index and chunk ID mapping."""
        if vectors is not None and len(vectors) > 0:
            self.vectors = np.asarray(vectors, dtype=np.float32)
        else:
            self.vectors = np.empty((0, self.dim), dtype=np.float32)
        self.chunk_ids = list(chunk_ids) if chunk_ids is not None else []
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}

    def search(
        self,
        query: str,
        limit: int = 20,
        k: int = 60,
    ) -> List[SearchResult]:
        """Perform hybrid search over indexed chunks using Reciprocal Rank Fusion.

        Args:
            query: Natural language or code search string.
            limit: Maximum number of ranked results to return.
            k: RRF smoothing constant (standard default is 60).
        """
        if not query or not query.strip() or limit <= 0:
            return []

        # 1. Lexical BM25 search
        fts_matches = self.db.search_fts(query, limit=limit * 2)
        lexical_ranks: Dict[int, int] = {
            row["id"]: rank for rank, row in enumerate(fts_matches, start=1)
        }

        # 2. Semantic vector search
        semantic_ranks: Dict[int, int] = {}
        if len(self.vectors) > 0 and len(self.chunk_ids) > 0:
            q_embeddings = self.embedder.embed_texts([query])
            if len(q_embeddings) > 0:
                q_vec = q_embeddings[0]
                sims = np.dot(self.vectors, q_vec)
                if sims.ndim > 1:
                    sims = sims.squeeze()
                top_indices = np.argsort(-sims)[: limit * 2]
                for rank, idx in enumerate(top_indices, start=1):
                    if idx < len(self.chunk_ids):
                        cid = self.chunk_ids[idx]
                        semantic_ranks[cid] = rank

        # 3. Reciprocal Rank Fusion
        candidate_ids = set(lexical_ranks.keys()) | set(semantic_ranks.keys())
        if not candidate_ids:
            return []

        chunk_data_cache: Dict[int, Dict[str, Any]] = {
            row["id"]: dict(row) for row in fts_matches
        }

        # Retrieve missing chunk data from DB if found via vector search only
        missing_ids = [cid for cid in candidate_ids if cid not in chunk_data_cache]
        if missing_ids:
            conn = self.db.get_connection()
            try:
                cursor = conn.cursor()
                batch_size = 500
                for i in range(0, len(missing_ids), batch_size):
                    batch = missing_ids[i : i + batch_size]
                    placeholders = ",".join("?" * len(batch))
                    cursor.execute(
                        f"SELECT id, path, start_line, end_line, symbol, content FROM chunks WHERE id IN ({placeholders})",
                        batch,
                    )
                    for row in cursor.fetchall():
                        chunk_data_cache[row["id"]] = dict(row)
            finally:
                conn.close()

        scored_candidates: List[SearchResult] = []
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
            scored_candidates.append(
                SearchResult(
                    chunk_id=cid,
                    path=Path(data["path"]),
                    start_line=int(data["start_line"]),
                    end_line=int(data["end_line"]),
                    symbol=data.get("symbol"),
                    content=str(data["content"]),
                    score=float(score),
                    lexical_rank=r_lex,
                    semantic_rank=r_sem,
                )
            )

        scored_candidates.sort(key=lambda r: r.score, reverse=True)
        return scored_candidates[:limit]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 20,
        k: int = 60,
    ) -> List[SearchResult]:
        """Convenience alias for search matching interface specification."""
        return self.search(query=query, limit=top_k, k=k)


def search_hybrid(
    engine: HybridSearchEngine,
    query: str,
    top_k: int = 20,
    k: int = 60,
) -> List[SearchResult]:
    """Helper function to perform hybrid search on an engine instance."""
    return engine.search(query=query, limit=top_k, k=k)
