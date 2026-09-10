"""CLI entrypoint and commands for HyperIndex."""

import json
from pathlib import Path
from typing import Optional
import typer

from hyperindex.config import get_config
from hyperindex.db import Database
from hyperindex.embedder import Embedder
from hyperindex.search import HybridSearchEngine

app = typer.Typer(
    name="hyperindex",
    help="HyperIndex: GPU-accelerated hybrid local search",
    no_args_is_help=True,
)


def get_engine() -> HybridSearchEngine:
    """Initialize and return a HybridSearchEngine instance with configured database and embedder."""
    config = get_config()
    db = Database(config.db_path)
    db.initialize()
    embedder = Embedder.create_mock_or_real()
    vectors, chunk_ids = None, None
    if config.vectors_path.exists():
        from hyperindex.watcher import load_vectors_file

        vectors, chunk_ids = load_vectors_file(config.vectors_path)
    return HybridSearchEngine(db, embedder, vectors=vectors, chunk_ids=chunk_ids)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query string"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
    plain: bool = typer.Option(False, "--plain", help="Output plain text format"),
    paths_only: bool = typer.Option(False, "--paths-only", help="Output matching file paths only"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results to return"),
    tui: bool = typer.Option(False, "--tui", help="Launch interactive split-pane TUI"),
):
    """Perform hybrid lexical and semantic search across indexed code."""
    engine = get_engine()

    if tui:
        from hyperindex.tui.app import launch_tui

        launch_tui(engine, query, limit=limit)
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
                "content": r.content,
            }
            for r in results
        ]
        typer.echo(json.dumps(output, indent=2))
        return

    if paths_only:
        seen = set()
        for r in results:
            p_str = str(r.path)
            if p_str not in seen:
                seen.add(p_str)
                typer.echo(p_str)
        return

    if not results:
        typer.echo(f"No matches found for '{query}'.")
        return

    for idx, r in enumerate(results, start=1):
        sym = f" [{r.symbol}]" if r.symbol else ""
        typer.secho(
            f"{idx}. {r.path}:{r.start_line}{sym} (score: {r.score:.4f})",
            fg=typer.colors.CYAN if not plain else None,
            bold=not plain,
            color=not plain,
        )
        content_preview = r.content[:160].strip()
        typer.echo(f"   {content_preview}...\n")


@app.command()
def tui(
    query: str = typer.Argument("", help="Initial search query string"),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum results to return"),
):
    """Launch the interactive split-pane terminal user interface (TUI)."""
    engine = get_engine()
    from hyperindex.tui.app import launch_tui

    launch_tui(engine, query, limit=limit)


@app.command()
def index(
    path: Optional[Path] = typer.Argument(
        None, help="Directory path to index (defaults to configured watch paths)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-indexing all files"),
):
    """Index code files in target directory for hybrid search."""
    config = get_config()
    target_paths = [path] if path is not None else config.watch_paths
    typer.echo(f"Indexing paths: {', '.join(str(p) for p in target_paths)}...")

    from hyperindex.watcher import IndexWatcher

    db = Database(config.db_path)
    db.initialize()
    if force:
        db.clear_all()
        if config.vectors_path.exists():
            try:
                config.vectors_path.unlink()
            except OSError:
                pass

    embedder = Embedder.create_mock_or_real()
    watcher = IndexWatcher(
        db=db,
        embedder=embedder,
        watch_paths=target_paths,
        vectors_path=config.vectors_path,
    )
    total_indexed = 0
    for p in target_paths:
        target = Path(p)
        if target.is_dir():
            total_indexed += watcher.index_directory(target)
        elif target.is_file():
            if watcher.index_file(target):
                total_indexed += 1
    typer.echo(f"Indexing completed: {total_indexed} files processed.")


@app.command()
def watch(
    path: Optional[Path] = typer.Argument(
        None, help="Directory path to watch for real-time indexing"
    ),
    continuous: bool = typer.Option(
        True, "--continuous/--no-continuous", help="Run continuous watcher daemon loop"
    ),
):
    """Start filesystem watcher daemon for automatic incremental indexing."""
    config = get_config()
    target_paths = [path] if path is not None else config.watch_paths
    typer.echo(f"Watching paths for changes: {', '.join(str(p) for p in target_paths)}...")

    from hyperindex.watcher import IndexWatcher

    db = Database(config.db_path)
    db.initialize()
    embedder = Embedder.create_mock_or_real()
    watcher = IndexWatcher(
        db=db,
        embedder=embedder,
        watch_paths=target_paths,
        vectors_path=config.vectors_path,
    )
    if continuous:
        watcher.run()


if __name__ == "__main__":
    app()
