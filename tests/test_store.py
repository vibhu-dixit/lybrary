"""Tests for lybrary.indexer.store.IndexStore (SQLite + numpy backend)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from lybrary.indexer.ast_chunker import CodeChunk
from lybrary.indexer.store import IndexStore

_DIM = 384  # MiniLM vector dimension


def _zero_vec() -> list[float]:
    return [0.0] * _DIM


def _unit_vec() -> list[float]:
    return [1.0 / (_DIM ** 0.5)] * _DIM


def _make_chunk(
    path: str = "src/foo.py",
    symbol: str = "foo",
    kind: str = "function_definition",
    start_line: int = 1,
    content: str = "def foo(): pass",
    language: str = "python",
) -> CodeChunk:
    return CodeChunk(
        id=f"{path}::{symbol}:{start_line}:0",
        path=path,
        symbol=symbol,
        kind=kind,
        start_line=start_line,
        end_line=start_line + 4,
        content=content,
        language=language,
        context_header=f"// File: {path} → {kind} {symbol}",
    )


@pytest.fixture()
def store(tmp_path: Path) -> IndexStore:
    return IndexStore(tmp_path)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_store_creates_lybrary_dir(tmp_path):
    IndexStore(tmp_path)
    assert (tmp_path / ".lybrary").exists()


def test_store_creates_db_file(tmp_path):
    IndexStore(tmp_path)
    assert (tmp_path / ".lybrary" / "index.db").exists()


def test_read_only_store_uses_existing_index_without_setup(tmp_path):
    writer = IndexStore(tmp_path)
    writer.upsert_chunks([_make_chunk()], [_unit_vec()])
    writer.close()
    with patch("lybrary.indexer.store.ensure_dir", side_effect=AssertionError("unexpected write")):
        reader = IndexStore(tmp_path, read_only=True)
        assert reader.count() == 1
        reader.close()


def test_initial_count_is_zero(store):
    assert store.count() == 0


# ---------------------------------------------------------------------------
# Hash tracking
# ---------------------------------------------------------------------------


def test_load_hashes_empty_when_no_file(store):
    assert store.load_hashes() == {}


def test_save_and_load_hashes(store):
    hashes = {"src/a.py": "abc123", "src/b.py": "def456"}
    store.save_hashes(hashes)
    assert store.load_hashes() == hashes


def test_get_changed_files_all_new(store):
    current = {"a.py": "hash1", "b.py": "hash2"}
    changed, deleted = store.get_changed_files(current)
    assert set(changed) == {"a.py", "b.py"}
    assert deleted == []


def test_get_changed_files_some_changed(store):
    store.save_hashes({"a.py": "old", "b.py": "same"})
    changed, deleted = store.get_changed_files({"a.py": "new", "b.py": "same"})
    assert changed == ["a.py"]
    assert deleted == []


def test_get_changed_files_deleted(store):
    store.save_hashes({"a.py": "h1", "b.py": "h2"})
    changed, deleted = store.get_changed_files({"a.py": "h1"})
    assert changed == []
    assert deleted == ["b.py"]


# ---------------------------------------------------------------------------
# upsert_chunks
# ---------------------------------------------------------------------------


def test_upsert_adds_chunks(store):
    store.upsert_chunks([_make_chunk()], [_unit_vec()])
    assert store.count() == 1


def test_upsert_multiple_chunks(store):
    chunks = [
        _make_chunk(symbol="foo", start_line=1),
        _make_chunk(symbol="bar", start_line=10),
    ]
    store.upsert_chunks(chunks, [_unit_vec(), _zero_vec()])
    assert store.count() == 2


def test_upsert_replaces_existing_for_path(store):
    """Upserting the same path twice must not accumulate stale rows."""
    store.upsert_chunks([_make_chunk(symbol="foo")], [_unit_vec()])
    store.upsert_chunks([_make_chunk(symbol="foo_v2")], [_unit_vec()])
    assert store.count() == 1


def test_upsert_empty_is_noop(store):
    store.upsert_chunks([], [])
    assert store.count() == 0


def test_upsert_preserves_content(store):
    store.upsert_chunks([_make_chunk(content="def magic(): return 42")], [_unit_vec()])
    results = store.search_vector(_unit_vec(), limit=1)
    assert results[0]["content"] == "def magic(): return 42"


# ---------------------------------------------------------------------------
# delete_path
# ---------------------------------------------------------------------------


def test_delete_path_removes_chunks(store):
    store.upsert_chunks([_make_chunk(path="a.py")], [_unit_vec()])
    store.upsert_chunks([_make_chunk(path="b.py", symbol="bar")], [_unit_vec()])
    store.delete_path("a.py")
    assert store.count() == 1


def test_delete_nonexistent_is_noop(store):
    store.upsert_chunks([_make_chunk()], [_unit_vec()])
    store.delete_path("no_such_file.py")
    assert store.count() == 1


def test_delete_path_with_quote_does_not_raise(store):
    """Parameterised query: paths with quotes must not raise or corrupt DB."""
    tricky = 'src/he said "hello".py'
    store.upsert_chunks([_make_chunk(path=tricky)], [_unit_vec()])
    store.delete_path(tricky)  # should not raise
    assert store.count() == 0


def test_delete_path_with_backslash(store):
    """Windows paths with backslashes must be handled correctly."""
    win_path = "src\\subdir\\module.py"
    store.upsert_chunks([_make_chunk(path=win_path)], [_unit_vec()])
    store.delete_path(win_path)
    assert store.count() == 0


# ---------------------------------------------------------------------------
# truncate
# ---------------------------------------------------------------------------


def test_truncate_clears_all(store):
    store.upsert_chunks(
        [_make_chunk(path="a.py"), _make_chunk(path="b.py", symbol="bar")],
        [_unit_vec(), _unit_vec()],
    )
    store.truncate()
    assert store.count() == 0


def test_truncate_then_upsert(store):
    store.upsert_chunks([_make_chunk()], [_unit_vec()])
    store.truncate()
    store.upsert_chunks([_make_chunk(symbol="fresh")], [_unit_vec()])
    assert store.count() == 1


# ---------------------------------------------------------------------------
# search_vector
# ---------------------------------------------------------------------------


def test_search_returns_results(store):
    store.upsert_chunks([_make_chunk()], [_unit_vec()])
    results = store.search_vector(_unit_vec(), limit=5)
    assert len(results) == 1


def test_search_no_vector_column(store):
    """Raw vector bytes must not leak into search results."""
    store.upsert_chunks([_make_chunk()], [_unit_vec()])
    row = store.search_vector(_unit_vec(), limit=1)[0]
    assert "vector" not in row


def test_search_has_expected_columns(store):
    store.upsert_chunks([_make_chunk()], [_unit_vec()])
    row = store.search_vector(_unit_vec(), limit=1)[0]
    for col in ("id", "path", "symbol", "kind", "content", "start_line", "end_line", "_distance"):
        assert col in row, f"missing column: {col}"


def test_search_distance_in_valid_range(store):
    """_distance must be in [0, 2] for unit-normalised vectors."""
    store.upsert_chunks([_make_chunk()], [_unit_vec()])
    row = store.search_vector(_unit_vec(), limit=1)[0]
    assert 0.0 <= row["_distance"] <= 2.0


def test_search_identical_vector_distance_near_zero(store):
    """Searching with the exact same vector should give distance ≈ 0."""
    store.upsert_chunks([_make_chunk()], [_unit_vec()])
    row = store.search_vector(_unit_vec(), limit=1)[0]
    assert row["_distance"] == pytest.approx(0.0, abs=1e-5)


def test_search_orthogonal_vector_distance(store):
    a = [1.0] + [0.0] * (_DIM - 1)
    b = [0.0, 1.0] + [0.0] * (_DIM - 2)
    store.upsert_chunks([_make_chunk()], [a])
    assert store.search_vector(b, limit=1)[0]["_distance"] == pytest.approx(2 ** 0.5)


def test_search_limit_respected(store):
    chunks = [_make_chunk(symbol=f"fn{i}", start_line=i * 10) for i in range(5)]
    store.upsert_chunks(chunks, [_unit_vec()] * 5)
    assert len(store.search_vector(_unit_vec(), limit=3)) <= 3


def test_search_empty_store(store):
    assert store.search_vector(_unit_vec(), limit=5) == []


def test_search_returns_most_similar_first(store):
    """Most similar chunk (unit vec) should rank above a zero-vec chunk."""
    close = _make_chunk(symbol="close", content="close match")
    far = _make_chunk(symbol="far", content="far away", start_line=20)
    # close gets the unit vector (perfect match); far gets zero vector (no match)
    store.upsert_chunks([close, far], [_unit_vec(), _zero_vec()])
    results = store.search_vector(_unit_vec(), limit=2)
    assert results[0]["symbol"] == "close"


def test_search_symbol_finds_exact_name_inside_export_declaration(store):
    store.upsert_chunks(
        [
            _make_chunk(symbol="export function formatIcsDate(d: Date)"),
            _make_chunk(symbol="buildIcs", start_line=20),
        ],
        [_unit_vec(), _unit_vec()],
    )
    results = store.search_symbol("formatIcsDate", limit=2)
    assert [row["symbol"] for row in results] == ["export function formatIcsDate(d: Date)"]
    assert results[0]["path"] == "src/foo.py"


def test_search_text_ranks_content_with_both_query_terms(store):
    store.upsert_chunks(
        [
            _make_chunk(path="a.ts", symbol="calendarEmail", content="interview.ics attachment"),
            _make_chunk(path="b.ts", symbol="other", start_line=20, content="interview"),
        ],
        [_zero_vec(), _unit_vec()],
    )
    results = store.search_text(["interview", "attachment"], limit=2)
    assert [row["path"] for row in results] == ["a.ts", "b.ts"]


def test_search_text_prefers_content_covering_more_terms_than_generic_symbol(store):
    store.upsert_chunks(
        [
            _make_chunk(
                path="generic.ts", symbol="sendInterviewEmail",
                content="interview email sent",
            ),
            _make_chunk(
                path="attachment.ts", symbol="sendInvitation", start_line=20,
                content="interview calendar attachment email sent",
            ),
        ],
        [_unit_vec(), _zero_vec()],
    )
    results = store.search_text(
        ["interview", "calendar", "attachment", "email", "sent"], limit=2
    )
    assert results[0]["path"] == "attachment.ts"


# ---------------------------------------------------------------------------
# all_paths
# ---------------------------------------------------------------------------


def test_all_paths(store):
    store.upsert_chunks([_make_chunk(path="a.py")], [_unit_vec()])
    store.upsert_chunks([_make_chunk(path="b.py", symbol="bar")], [_unit_vec()])
    assert store.all_paths() == {"a.py", "b.py"}


def test_all_paths_empty_store(store):
    assert store.all_paths() == set()
