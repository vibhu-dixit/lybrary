"""Tests for lybrary.indexer.ast_chunker."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lybrary.indexer.ast_chunker import (
    CodeChunk,
    _merge_intervals,
    _sanitize_name,
    chunk_file,
    chunk_source,
    detect_language,
)


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------


def test_detect_language_known_extensions():
    assert detect_language(Path("foo.py")) == "python"
    assert detect_language(Path("foo.js")) == "javascript"
    assert detect_language(Path("foo.ts")) == "typescript"
    assert detect_language(Path("foo.tsx")) == "tsx"
    assert detect_language(Path("foo.go")) == "go"
    assert detect_language(Path("foo.rs")) == "rust"
    assert detect_language(Path("foo.java")) == "java"
    assert detect_language(Path("foo.c")) == "c"
    assert detect_language(Path("foo.cpp")) == "cpp"


def test_detect_language_unknown():
    assert detect_language(Path("foo.txt")) is None
    assert detect_language(Path("Makefile")) is None


def test_detect_language_case_insensitive():
    assert detect_language(Path("FOO.PY")) == "python"
    assert detect_language(Path("bar.TS")) == "typescript"


# ---------------------------------------------------------------------------
# _merge_intervals
# ---------------------------------------------------------------------------


def test_merge_intervals_empty():
    assert _merge_intervals([]) == []


def test_merge_intervals_no_overlap():
    assert _merge_intervals([(0, 10), (20, 30)]) == [(0, 10), (20, 30)]


def test_merge_intervals_adjacent():
    assert _merge_intervals([(0, 10), (10, 20)]) == [(0, 20)]


def test_merge_intervals_overlap():
    assert _merge_intervals([(0, 20), (10, 30)]) == [(0, 30)]


def test_merge_intervals_nested():
    assert _merge_intervals([(0, 100), (10, 50)]) == [(0, 100)]


def test_merge_intervals_unsorted():
    result = _merge_intervals([(20, 30), (0, 10), (5, 25)])
    assert result == [(0, 30)]


# ---------------------------------------------------------------------------
# _sanitize_name
# ---------------------------------------------------------------------------


def test_sanitize_name_strips_quotes():
    assert '"' not in _sanitize_name('foo"bar')


def test_sanitize_name_strips_newlines():
    assert "\n" not in _sanitize_name("foo\nbar")


def test_sanitize_name_strips_double_colon():
    assert "::" not in _sanitize_name("foo::bar")


def test_sanitize_name_truncates():
    assert len(_sanitize_name("x" * 200)) <= 80


# ---------------------------------------------------------------------------
# chunk_source — Python
# ---------------------------------------------------------------------------

_PYTHON_SIMPLE = textwrap.dedent("""\
    import os

    def greet(name: str) -> str:
        return f"Hello, {name}"

    def farewell(name: str) -> str:
        return f"Goodbye, {name}"
""")

_PYTHON_CLASS = textwrap.dedent("""\
    class Greeter:
        def __init__(self, prefix: str) -> None:
            self.prefix = prefix

        def greet(self, name: str) -> str:
            return f"{self.prefix}, {name}"
""")


def test_chunk_source_python_functions():
    chunks = chunk_source(_PYTHON_SIMPLE, "test.py", "python")
    symbols = {c.symbol for c in chunks}
    assert "greet" in symbols
    assert "farewell" in symbols


def test_chunk_source_python_module_level():
    # Pass min_chunk_tokens=1 so even small import blocks produce a module_level chunk
    chunks = chunk_source(_PYTHON_SIMPLE, "test.py", "python", min_chunk_tokens=1)
    kinds = {c.kind for c in chunks}
    assert "module_level" in kinds
    module_chunk = next(c for c in chunks if c.kind == "module_level")
    assert "import os" in module_chunk.content


def test_chunk_source_python_class_and_methods():
    chunks = chunk_source(_PYTHON_CLASS, "test.py", "python")
    symbols = {c.symbol for c in chunks}
    assert "Greeter" in symbols


def test_chunk_source_unique_ids():
    chunks = chunk_source(_PYTHON_SIMPLE, "test.py", "python")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_source_sorted_by_start_line():
    chunks = chunk_source(_PYTHON_SIMPLE, "test.py", "python")
    lines = [c.start_line for c in chunks]
    assert lines == sorted(lines)


def test_chunk_source_content_matches_source():
    chunks = chunk_source(_PYTHON_SIMPLE, "test.py", "python")
    greet = next(c for c in chunks if c.symbol == "greet")
    assert "def greet" in greet.content
    assert "Hello" in greet.content


def test_chunk_source_correct_language_stored():
    chunks = chunk_source(_PYTHON_SIMPLE, "test.py", "python")
    for c in chunks:
        assert c.language == "python"


def test_chunk_source_empty_source():
    chunks = chunk_source("", "empty.py", "python")
    assert isinstance(chunks, list)


def test_chunk_source_nested_class_not_fully_duplicated():
    """Bug #3: method chunk content must differ from the class chunk content."""
    chunks = chunk_source(_PYTHON_CLASS, "test.py", "python")
    class_chunk = next((c for c in chunks if c.symbol == "Greeter"), None)
    if class_chunk:
        method_chunks = [c for c in chunks if c.symbol == "greet"]
        for mc in method_chunks:
            assert mc.content != class_chunk.content


def test_chunk_source_tsx_language():
    """Bug #1: tsx files must keep language='tsx', not be converted to typescript."""
    tsx_source = (
        "import React from 'react';\n"
        "function Button({ label }) {\n"
        "  return <button>{label}</button>;\n"
        "}\n"
        "export default Button;\n"
    )
    chunks = chunk_source(tsx_source, "Button.tsx", "tsx")
    assert isinstance(chunks, list)
    for c in chunks:
        assert c.language == "tsx"


# ---------------------------------------------------------------------------
# chunk_source — fallback (unknown language)
# ---------------------------------------------------------------------------


def test_fallback_line_chunks():
    source = "\n".join(f"line {i}" for i in range(200))
    chunks = chunk_source(source, "file.xyz", "unknown_lang")
    assert len(chunks) >= 1
    for c in chunks:
        assert c.kind == "block"


def test_fallback_unique_ids():
    source = "\n".join(f"line {i}" for i in range(300))
    chunks = chunk_source(source, "file.xyz", "unknown_lang")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# chunk_file
# ---------------------------------------------------------------------------


def test_chunk_file_returns_empty_for_unknown_ext(tmp_path):
    f = tmp_path / "script.sh"
    f.write_text("echo hello\n")
    assert chunk_file(f, tmp_path) == []


def test_chunk_file_python(tmp_path):
    f = tmp_path / "hello.py"
    f.write_text(_PYTHON_SIMPLE)
    chunks = chunk_file(f, tmp_path)
    assert len(chunks) > 0
    assert "greet" in {c.symbol for c in chunks}


def test_chunk_file_relative_path_uses_root(tmp_path):
    sub = tmp_path / "src"
    sub.mkdir()
    f = sub / "utils.py"
    f.write_text(_PYTHON_SIMPLE)
    chunks = chunk_file(f, tmp_path)
    for c in chunks:
        assert not Path(c.path).is_absolute()
        assert "src" in c.path


def test_chunk_file_missing_returns_empty(tmp_path):
    assert chunk_file(tmp_path / "nonexistent.py", tmp_path) == []


def test_chunk_file_tsx_language_preserved(tmp_path):
    """Bug #1: chunk_file must not strip tsx to typescript."""
    f = tmp_path / "App.tsx"
    f.write_text(
        "import React from 'react';\n"
        "function App() { return <div>Hello</div>; }\n"
        "export default App;\n"
    )
    chunks = chunk_file(f, tmp_path)
    assert isinstance(chunks, list)
    for c in chunks:
        assert c.language == "tsx"


# ---------------------------------------------------------------------------
# CodeChunk.embedding_text
# ---------------------------------------------------------------------------


def test_embedding_text_includes_header_and_content():
    c = CodeChunk(
        id="f::foo:1:0",
        path="f.py",
        symbol="foo",
        kind="function_definition",
        start_line=1,
        end_line=3,
        content="def foo():\n    pass\n",
        language="python",
        context_header="// File: f.py → function_definition foo",
    )
    text = c.embedding_text()
    assert "// File: f.py" in text
    assert "def foo()" in text
