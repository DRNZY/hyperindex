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
        symbol="get_auth_token",
    )
    chunk2 = Chunk(
        file_path=Path("/home/test/player.py"),
        start_line=15,
        end_line=25,
        content="def play_audio_track(track_id): pass",
        symbol="play_audio_track",
    )

    db.index_chunks([chunk1, chunk2])

    results = db.search_fts("get_auth_token", limit=10)
    assert len(results) >= 1
    assert results[0]["symbol"] == "get_auth_token"
    assert "/home/test/auth.py" in results[0]["path"]


def test_db_remove_file(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.initialize()

    chunk1 = Chunk(
        file_path=Path("/home/test/auth.py"),
        start_line=1,
        end_line=10,
        content="def get_auth_token(user_id): return 'secret-jwt'",
        symbol="get_auth_token",
    )
    chunk2 = Chunk(
        file_path=Path("/home/test/player.py"),
        start_line=15,
        end_line=25,
        content="def play_audio_track(track_id): pass",
        symbol="play_audio_track",
    )

    db.index_chunks([chunk1, chunk2])
    assert len(db.search_fts("get_auth_token")) == 1
    assert len(db.search_fts("play_audio_track")) == 1

    db.remove_file(Path("/home/test/auth.py"))

    assert len(db.search_fts("get_auth_token")) == 0
    assert len(db.search_fts("play_audio_track")) == 1


def test_db_get_all_chunks(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.initialize()

    assert db.get_all_chunks() == []

    chunk1 = Chunk(
        file_path=Path("/home/test/auth.py"),
        start_line=1,
        end_line=10,
        content="def get_auth_token(user_id): return 'secret-jwt'",
        symbol="get_auth_token",
        chunk_index=0,
    )
    ids = db.index_chunks([chunk1])
    assert len(ids) == 1

    chunks = db.get_all_chunks()
    assert len(chunks) == 1
    c = chunks[0]
    assert c.id == ids[0]
    assert c.file_path == Path("/home/test/auth.py")
    assert c.start_line == 1
    assert c.end_line == 10
    assert c.content == "def get_auth_token(user_id): return 'secret-jwt'"
    assert c.symbol == "get_auth_token"
    assert c.chunk_index == 0


def test_db_search_fts_empty_and_special_chars(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.initialize()

    chunk = Chunk(
        file_path=Path("/home/test/utils.py"),
        start_line=1,
        end_line=5,
        content="class TokenValidator:\n    def validate(token):\n        return True",
        symbol="TokenValidator",
    )
    db.index_chunks([chunk])

    # Empty / whitespace
    assert db.search_fts("") == []
    assert db.search_fts("   ") == []

    # Non-alphanumeric special characters
    assert db.search_fts("!@#$%^&()") == []
    assert db.search_fts("***") == []

    # Valid matches with punctuation
    results = db.search_fts("validate(token)")
    assert len(results) >= 1
    assert results[0]["symbol"] == "TokenValidator"


def test_db_nested_directory_auto_created(tmp_path):
    nested_db_path = tmp_path / "subdir" / "nested" / "test.db"
    db = Database(nested_db_path)
    db.initialize()
    assert nested_db_path.exists()


def test_db_wal_mode_and_pragmas(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.initialize()

    conn = db.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0]
        assert journal_mode.lower() == "wal"

        cursor.execute("PRAGMA foreign_keys;")
        foreign_keys = cursor.fetchone()[0]
        assert foreign_keys == 1
    finally:
        conn.close()
