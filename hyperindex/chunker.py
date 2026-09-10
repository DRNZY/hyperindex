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
    id: Optional[int] = None
    token_count: Optional[int] = None


SYMBOL_REGEX = re.compile(
    r"^\s*(def|class|function|export\s+function|export\s+const|const|struct|impl|fn)\s+([A-Za-z0-9_]+)",
    re.MULTILINE,
)


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
                if len(current_lines) > overlap_lines
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
