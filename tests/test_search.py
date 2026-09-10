from pathlib import Path
import numpy as np
import pytest

from hyperindex.chunker import Chunk
from hyperindex.db import Database
from hyperindex.embedder import Embedder
from hyperindex.search import HybridSearchEngine, SearchResult, search_hybrid


def test_hybrid_search_rrf_scoring(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    chunk = Chunk(
        file_path=Path("/app/tokens.py"),
        start_line=1,
        end_line=5,
        content="def verify_jwt_token(): return True",
        symbol="verify_jwt_token",
    )
    chunk_ids = db.index_chunks([chunk])
    vectors = embedder.embed_texts([chunk.content])

    engine = HybridSearchEngine(db, embedder, vectors, chunk_ids)
    results = engine.search("verify_jwt_token", limit=5)

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].symbol == "verify_jwt_token"
    assert results[0].score > 0
    assert results[0].chunk_id == chunk_ids[0]
    assert results[0].path == Path("/app/tokens.py")
    assert results[0].start_line == 1
    assert results[0].end_line == 5
    assert results[0].lexical_rank == 1
    assert results[0].semantic_rank == 1


def test_hybrid_search_empty_query_and_zero_limit(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    engine = HybridSearchEngine(db, embedder)
    assert engine.search("") == []
    assert engine.search("   ") == []
    assert engine.search("test", limit=0) == []
    assert engine.search("test", limit=-5) == []


def test_hybrid_search_lexical_only(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    chunk = Chunk(
        file_path=Path("/app/utils.py"),
        start_line=10,
        end_line=20,
        content="def calculate_statistics(): return 42",
        symbol="calculate_statistics",
    )
    chunk_ids = db.index_chunks([chunk])

    # No vectors provided
    engine = HybridSearchEngine(db, embedder, vectors=None, chunk_ids=None)
    results = engine.search("calculate_statistics", limit=10, k=60)

    assert len(results) == 1
    assert results[0].chunk_id == chunk_ids[0]
    assert results[0].lexical_rank == 1
    assert results[0].semantic_rank is None
    # Score for rank 1 with k=60: 1 / (60 + 1)
    assert pytest.approx(results[0].score, rel=1e-5) == 1.0 / 61.0


def test_hybrid_search_semantic_only(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    chunk = Chunk(
        file_path=Path("/app/math.py"),
        start_line=1,
        end_line=10,
        content="x = 12345",
        symbol=None,
    )
    chunk_ids = db.index_chunks([chunk])

    # Provide vector that will match query
    dim = embedder.dim
    # Chunk vector
    vec = np.ones((1, dim), dtype=np.float32)
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)

    engine = HybridSearchEngine(db, embedder, vectors=vec, chunk_ids=chunk_ids)

    # Query terms that don't match FTS content
    results = engine.search("nonexistent_fts_term", limit=5, k=60)

    assert len(results) == 1
    assert results[0].chunk_id == chunk_ids[0]
    assert results[0].lexical_rank is None
    assert results[0].semantic_rank == 1
    assert pytest.approx(results[0].score, rel=1e-5) == 1.0 / 61.0
    assert results[0].content == "x = 12345"


def test_hybrid_search_rrf_ranking_combined(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    chunk1 = Chunk(
        file_path=Path("/app/auth.py"),
        start_line=1,
        end_line=10,
        content="def authenticate_user(): pass",
        symbol="authenticate_user",
    )
    chunk2 = Chunk(
        file_path=Path("/app/login.py"),
        start_line=1,
        end_line=10,
        content="def authenticate_admin(): pass",
        symbol="authenticate_admin",
    )
    chunk_ids = db.index_chunks([chunk1, chunk2])

    # Query vector
    q_vec = embedder.embed_texts(["authenticate"])[0]

    # chunk1 gets q_vec, chunk2 gets -q_vec
    vectors = np.vstack([q_vec, -q_vec])

    engine = HybridSearchEngine(db, embedder, vectors=vectors, chunk_ids=chunk_ids)
    results = engine.search("authenticate", limit=5, k=60)

    assert len(results) == 2
    # chunk1 should be rank 1 in semantic and in lexical
    assert results[0].chunk_id == chunk_ids[0]
    assert results[0].lexical_rank is not None
    assert results[0].semantic_rank == 1
    # chunk1 has higher combined score
    assert results[0].score > results[1].score


def test_hybrid_search_custom_k(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    chunk = Chunk(
        file_path=Path("/app/test.py"),
        start_line=1,
        end_line=3,
        content="foo bar baz",
        symbol=None,
    )
    chunk_ids = db.index_chunks([chunk])

    engine = HybridSearchEngine(db, embedder, vectors=None, chunk_ids=None)
    results = engine.search("foo", limit=5, k=10)

    assert len(results) == 1
    # r_lex = 1, k = 10 -> score = 1 / 11
    assert pytest.approx(results[0].score, rel=1e-5) == 1.0 / 11.0


def test_search_hybrid_alias_and_helper(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    chunk = Chunk(
        file_path=Path("/app/code.py"),
        start_line=1,
        end_line=5,
        content="class NetworkHandler: pass",
        symbol="NetworkHandler",
    )
    chunk_ids = db.index_chunks([chunk])
    vectors = embedder.embed_texts([chunk.content])

    engine = HybridSearchEngine(db, embedder, vectors, chunk_ids)

    # Method alias
    res1 = engine.search_hybrid("NetworkHandler", top_k=5)
    # Function alias
    res2 = search_hybrid(engine, "NetworkHandler", top_k=5)

    assert len(res1) == 1
    assert len(res2) == 1
    assert res1[0].chunk_id == chunk_ids[0]
    assert res2[0].chunk_id == chunk_ids[0]
    assert res1[0].score == res2[0].score


def test_hybrid_search_handles_missing_chunk_gracefully(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    embedder = Embedder.create_mock_or_real(use_mock=True)

    # chunk_id 99999 does not exist in DB
    dummy_vec = np.ones((1, embedder.dim), dtype=np.float32)
    engine = HybridSearchEngine(db, embedder, vectors=dummy_vec, chunk_ids=[99999])

    results = engine.search("something", limit=5)
    # Should not raise exception and missing ID should simply be skipped
    assert len(results) == 0
