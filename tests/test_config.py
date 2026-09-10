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


def test_config_directories_created(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    config = get_config()
    assert config.data_dir.is_dir()
    assert config.config_dir.is_dir()
    assert config.cache_dir.is_dir()


def test_config_fallback_when_unset(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    config = get_config()
    assert config.data_dir == fake_home / ".local" / "share" / "hyperindex"
    assert config.config_dir == fake_home / ".config" / "hyperindex"
    assert config.cache_dir == fake_home / ".cache" / "hyperindex"
    assert config.watch_paths == [fake_home / "Projects"]


def test_config_properties_and_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    config = get_config()

    assert config.batch_size == 64
    assert config.max_chunk_chars == 1500
    assert config.chunk_overlap == 200
    assert config.config_path == config.config_dir / "config.yaml"
