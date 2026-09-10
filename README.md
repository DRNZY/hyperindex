# HyperIndex

GPU-accelerated local code and document search engine using hybrid Reciprocal Rank Fusion (SQLite FTS5 BM25 + ONNX Runtime CUDA embeddings) with an interactive terminal user interface.

## Installation

```bash
git clone https://github.com/DRNZY/hyperindex.git
cd hyperindex
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

To make `hindex` available globally in your shell:

```bash
ln -sf $(pwd)/.venv/bin/hindex ~/.local/bin/hindex
```

Ensure `~/.local/bin` is in your `PATH`.

## Usage

Index a directory:

```bash
hindex index ~/Projects
```

Search from the command line:

```bash
hindex search "query"
hindex search "query" --json
hindex search "query" --plain
```

Launch the interactive split-pane TUI:

```bash
hindex tui
# or
hindex search "query" --tui
```

Run the background file watcher daemon:

```bash
hindex watch ~/Projects
```

## Keybindings (TUI)

- Up / Down or j / k: Navigate search results
- Enter: Open file in editor at line ($EDITOR, code, nvim)
- c: Copy file path to clipboard
- Tab: Switch focus between search input and results list
- /: Jump to search input
- q / Esc: Exit
