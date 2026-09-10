"""Configuration management and XDG directory resolution for HyperIndex."""

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

    @property
    def config_path(self) -> Path:
        return self.config_dir / "config.yaml"


def get_config() -> HyperIndexConfig:
    xdg_data = os.environ.get("XDG_DATA_HOME")
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    xdg_cache = os.environ.get("XDG_CACHE_HOME")

    data_home = Path(xdg_data) if xdg_data else Path.home() / ".local" / "share"
    config_home = Path(xdg_config) if xdg_config else Path.home() / ".config"
    cache_home = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"

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
        cache_dir=cache_dir,
    )
