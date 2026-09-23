from pathlib import Path
import pytest
from hyperindex.chunker import (
    chunk_code_or_text,
    chunk_file,
    Chunk,
    extract_symbols_from_python_ast,
    SymbolInfo,
)


def test_chunking_python_code():
    code = (
        "def compute_total(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        total += item.price\n"
        "    return total\n\n"
        "class Order:\n"
        "    def __init__(self, id):\n"
        "        self.id = id\n"
    )
    chunks = chunk_code_or_text(code, file_path=Path("order.py"), max_chars=100)
    assert len(chunks) >= 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].start_line == 1
    assert "compute_total" in chunks[0].content
    assert chunks[0].symbol is not None


def test_chunk_symbol_extraction():
    samples = [
        ("def calculate_tax():\n    return 0.1\n", "calculate_tax"),
        ("class InvoiceProcessor:\n    pass\n", "InvoiceProcessor"),
        ("function formatCurrency(val) {\n    return '$' + val;\n}\n", "formatCurrency"),
        ("export function parsePayload(data) {\n    return JSON.parse(data);\n}\n", "parsePayload"),
        ("export const API_BASE = 'https://api.example.com';\n", "API_BASE"),
        ("const MAX_RETRIES = 5;\n", "MAX_RETRIES"),
        ("struct UserRecord {\n    id: u64,\n}\n", "UserRecord"),
        ("impl UserRecord {\n}\n", "UserRecord"),
        ("fn process_event(ev: Event) -> Result<()> {}\n", "process_event"),
    ]

    for code, expected_sym in samples:
        chunks = chunk_code_or_text(code, file_path=Path("sample.src"), max_chars=500)
        assert len(chunks) >= 1
        assert chunks[0].symbol == expected_sym


def test_chunk_empty_input():
    chunks = chunk_code_or_text("", file_path=Path("empty.txt"))
    assert chunks == []

    chunks_whitespace = chunk_code_or_text("   \n\n\t  \n", file_path=Path("whitespace.txt"))
    assert chunks_whitespace == []


def test_chunk_small_text_single_chunk():
    text = "Line 1\nLine 2\nLine 3\n"
    chunks = chunk_code_or_text(text, file_path=Path("doc.md"), max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3
    assert chunks[0].chunk_index == 0
    assert chunks[0].symbol is None
    assert chunks[0].content == "Line 1\nLine 2\nLine 3"


def test_chunk_overlap():
    lines = [f"Line {i} with some additional content padding" for i in range(1, 21)]
    text = "\n".join(lines) + "\n"
    chunks = chunk_code_or_text(text, file_path=Path("padded.txt"), max_chars=120, overlap_lines=2)

    assert len(chunks) > 1
    # Check that overlap lines appear at the beginning of subsequent chunk content
    for i in range(len(chunks) - 1):
        c_curr = chunks[i]
        c_next = chunks[i + 1]
        assert c_next.chunk_index == c_curr.chunk_index + 1
        # start_line of next chunk should be before or equal to end_line of current chunk
        assert c_next.start_line <= c_curr.end_line


def test_chunk_zero_overlap():
    lines = [f"Line {i:02d}\n" for i in range(1, 21)]
    text = "".join(lines)
    chunks = chunk_code_or_text(text, file_path=Path("no_overlap.txt"), max_chars=30, overlap_lines=0)
    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        c_curr = chunks[i]
        c_next = chunks[i + 1]
        assert c_next.start_line == c_curr.end_line + 1



def test_chunk_file(tmp_path):
    p = tmp_path / "hello.py"
    p.write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    chunks = chunk_file(p, max_chars=100)
    assert len(chunks) == 1
    assert chunks[0].symbol == "hello"
    assert "def hello():" in chunks[0].content

    # Non-existent file
    missing = tmp_path / "missing.py"
    assert chunk_file(missing) == []


def test_chunk_dataclass_fields():
    chunk = Chunk(
        file_path=Path("foo.py"),
        start_line=10,
        end_line=20,
        content="code snippet",
        symbol="foo_func",
        chunk_index=3,
        id=42,
        token_count=15,
    )
    assert chunk.file_path == Path("foo.py")
    assert chunk.start_line == 10
    assert chunk.end_line == 20
    assert chunk.content == "code snippet"
    assert chunk.symbol == "foo_func"
    assert chunk.chunk_index == 3
    assert chunk.id == 42
    assert chunk.token_count == 15


def test_extract_symbols_from_python_ast():
    code = (
        "class TelemetryManager:\n"
        "    \"\"\"Handles system metrics collection.\"\"\"\n"
        "    def __init__(self, sample_rate: int):\n"
        "        self.rate = sample_rate\n\n"
        "    async def collect_metrics(self) -> dict:\n"
        "        \"\"\"Collect active CPU and GPU telemetry.\"\"\"\n"
        "        return {}\n\n"
        "def helper_func():\n"
        "    pass\n"
    )
    symbols = extract_symbols_from_python_ast(code)
    assert len(symbols) == 4

    class_sym = symbols[0]
    assert class_sym.name == "TelemetryManager"
    assert class_sym.qualified_name == "TelemetryManager"
    assert class_sym.kind == "class"
    assert class_sym.docstring == "Handles system metrics collection."

    init_sym = symbols[1]
    assert init_sym.name == "__init__"
    assert init_sym.qualified_name == "TelemetryManager.__init__"
    assert init_sym.kind == "method"

    async_sym = symbols[2]
    assert async_sym.name == "collect_metrics"
    assert async_sym.qualified_name == "TelemetryManager.collect_metrics"
    assert async_sym.kind == "method"
    assert async_sym.docstring == "Collect active CPU and GPU telemetry."

    helper_sym = symbols[3]
    assert helper_sym.name == "helper_func"
    assert helper_sym.kind == "function"


def test_extract_symbols_syntax_error():
    invalid_code = "def bad_syntax(:\n    pass\n"
    symbols = extract_symbols_from_python_ast(invalid_code)
    assert symbols == []

