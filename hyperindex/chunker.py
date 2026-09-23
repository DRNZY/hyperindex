import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional


@dataclass
class SymbolInfo:
    name: str
    qualified_name: str
    kind: str  # "function", "async_function", "class", "method"
    start_line: int
    end_line: int
    docstring: Optional[str] = None


@dataclass
class Chunk:
    file_path: Path
    start_line: int
    end_line: int
    content: str
    symbol: Optional[str] = None
    chunk_index: int = 0
    id: Optional[int] = None
    token_count: Optional[int] = None


SYMBOL_REGEX = re.compile(
    r"^\s*(def|class|function|export\s+function|export\s+const|const|struct|impl|fn)\s+([A-Za-z0-9_]+)",
    re.MULTILINE,
)


def extract_symbols_from_python_ast(code: str) -> List[SymbolInfo]:
    """Extract all functions, async functions, classes, and methods using Python's AST."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    symbols: List[SymbolInfo] = []

    def visit_node(node: ast.AST, parent_scope: str = ""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if parent_scope else ("async_function" if isinstance(child, ast.AsyncFunctionDef) else "function")
                qual = f"{parent_scope}.{child.name}" if parent_scope else child.name
                end_line = getattr(child, "end_lineno", child.lineno)
                doc = ast.get_docstring(child)
                symbols.append(SymbolInfo(
                    name=child.name,
                    qualified_name=qual,
                    kind=kind,
                    start_line=child.lineno,
                    end_line=end_line,
                    docstring=doc,
                ))
                visit_node(child, qual)
            elif isinstance(child, ast.ClassDef):
                qual = f"{parent_scope}.{child.name}" if parent_scope else child.name
                end_line = getattr(child, "end_lineno", child.lineno)
                doc = ast.get_docstring(child)
                symbols.append(SymbolInfo(
                    name=child.name,
                    qualified_name=qual,
                    kind="class",
                    start_line=child.lineno,
                    end_line=end_line,
                    docstring=doc,
                ))
                visit_node(child, qual)

    visit_node(tree)
    symbols.sort(key=lambda s: (s.start_line, -s.end_line))
    return symbols


def chunk_code_or_text(
    text: str,
    file_path: Path,
    max_chars: int = 1500,
    overlap_lines: int = 4,
) -> List[Chunk]:
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
                chunks.append(
                    Chunk(
                        file_path=file_path,
                        start_line=current_start,
                        end_line=idx,
                        content=content,
                        symbol=current_symbol,
                        chunk_index=chunk_idx,
                    )
                )
                chunk_idx += 1

            # Retain overlap
            overlap = (
                current_lines[-overlap_lines:]
                if overlap_lines > 0 and len(current_lines) > overlap_lines
                else []
            )
            current_lines = list(overlap)
            current_start = max(1, idx - len(overlap) + 1)
            current_chars = sum(len(l) for l in current_lines)

    if current_lines and not (chunks and chunks[-1].end_line == len(lines)):
        content = "".join(current_lines).strip()
        if content:
            chunks.append(
                Chunk(
                    file_path=file_path,
                    start_line=current_start,
                    end_line=len(lines),
                    content=content,
                    symbol=current_symbol,
                    chunk_index=chunk_idx,
                )
            )

    return chunks


def chunk_file(
    path: Path,
    max_chars: int = 1500,
    overlap_lines: int = 4,
) -> List[Chunk]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    return chunk_code_or_text(
        text, file_path=path, max_chars=max_chars, overlap_lines=overlap_lines
    )
