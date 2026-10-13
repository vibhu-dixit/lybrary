"""Tests for lybrary.query — score formula and search() integration."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lybrary.query import _distance_to_score, query_terms, search

_DIM = 384
_UNIT_VEC = [1.0 / (_DIM ** 0.5)] * _DIM


# ---------------------------------------------------------------------------
# _distance_to_score
# ---------------------------------------------------------------------------


def test_score_perfect_match():
    assert _distance_to_score(0.0) == pytest.approx(1.0)


def test_score_opposite_vectors():
    # L2 distance = 2 for unit vectors pointing in opposite directions
    assert _distance_to_score(2.0) == pytest.approx(0.0)


def test_score_orthogonal():
    # For unit-normalised vectors: sim = 1 - d²/2
    # d=1.0 → sim = 1 - 0.5 = 0.5 (not d=√2, which gives sim≈0)
    assert _distance_to_score(1.0) == pytest.approx(0.5, abs=1e-9)


def test_score_sqrt2_is_zero():
    # d=√2 → cosine similarity = 0 (orthogonal unit vectors)
    assert _distance_to_score(math.sqrt(2)) == pytest.approx(0.0, abs=1e-6)


def test_score_clipped_below_zero():
    assert _distance_to_score(3.0) == 0.0


def test_score_in_unit_range():
    for d in (0.0, 0.5, 1.0, 1.5, 2.0):
        s = _distance_to_score(d)
        assert 0.0 <= s <= 1.0


def test_score_monotone_decreasing():
    scores = [_distance_to_score(d) for d in (0.0, 0.5, 1.0, 1.5, 2.0)]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# search() with mocked store + embedder
# ---------------------------------------------------------------------------


def _fake_row(
    symbol: str = "foo",
    path: str = "src/foo.py",
    content: str = "def foo(): pass",
    distance: float = 0.1,
) -> dict:
    return {
        "id": f"{path}::{symbol}:1:0",
        "path": path,
        "symbol": symbol,
        "kind": "function_definition",
        "start_line": 1,
        "end_line": 3,
        "content": content,
        "context_header": f"// File: {path}",
        "_distance": distance,
    }


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / ".lybrary").mkdir()
    return tmp_path


def _run_search(repo, rows, query="test", limit=8, max_tokens=6000):
    with (
        patch("lybrary.query.IndexStore") as MockStore,
        patch("lybrary.query.Embedder") as MockEmbedder,
    ):
        store_inst = MagicMock()
        store_inst.search_vector.return_value = rows
        store_inst.search_symbol.return_value = []
        store_inst.search_text.return_value = []
        MockStore.return_value = store_inst

        emb_inst = MagicMock()
        emb_inst.embed_query.return_value = _UNIT_VEC
        MockEmbedder.return_value = emb_inst

        return search(repo, query, limit=limit, max_tokens=max_tokens)


def test_search_returns_results(repo):
    rows = [_fake_row("foo"), _fake_row("bar", path="src/bar.py")]
    results = _run_search(repo, rows)
    assert len(results) == 2


def test_search_empty_store(repo):
    assert _run_search(repo, []) == []


def test_search_result_score_perfect_match(repo):
    results = _run_search(repo, [_fake_row(distance=0.0)])
    assert results[0]["score"] == pytest.approx(1.0)


def test_search_result_score_orthogonal(repo):
    # d=1.0 → cosine similarity = 0.5
    results = _run_search(repo, [_fake_row(distance=1.0)])
    assert results[0]["score"] == pytest.approx(0.5, abs=1e-9)


def test_search_required_fields(repo):
    results = _run_search(repo, [_fake_row()])
    row = results[0]
    for field in ("id", "path", "symbol", "kind", "start_line", "end_line", "content", "score"):
        assert field in row


def test_search_limit_respected(repo):
    rows = [_fake_row(symbol=f"fn{i}", path=f"src/f{i}.py") for i in range(10)]
    results = _run_search(repo, rows, limit=3)
    assert len(results) <= 3


def test_search_token_budget(repo):
    """First result alone exceeds budget → subsequent results excluded."""
    big = _fake_row(symbol="big", content="x" * 4000, distance=0.1)
    small = _fake_row(symbol="small", content="tiny", distance=0.2, path="src/b.py")
    results = _run_search(repo, [big, small], max_tokens=200)
    # big chunk (~1000 tokens) fills the budget; small should be cut
    symbols = {r["symbol"] for r in results}
    assert "big" in symbols
    assert "small" not in symbols


def test_search_skips_oversized_result_when_smaller_one_fits(repo):
    rows = [
        _fake_row(symbol="first", content="small", distance=0.1),
        _fake_row(symbol="huge", content="x" * 4000, distance=0.2),
        _fake_row(symbol="later", content="small", distance=0.3),
    ]
    results = _run_search(repo, rows, max_tokens=200)
    assert [r["symbol"] for r in results] == ["first", "later"]


def test_search_missing_distance_defaults(repo):
    """Rows without _distance key must not raise."""
    row = _fake_row()
    del row["_distance"]
    results = _run_search(repo, [row])
    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(0.0)


def test_exact_identifier_prefers_text_lookup_without_embedding(repo):
    with (
        patch("lybrary.query.IndexStore") as MockStore,
        patch("lybrary.query.Embedder") as MockEmbedder,
    ):
        store = MockStore.return_value
        store.search_symbol.return_value = [_fake_row(symbol="formatIcsDate")]
        results = search(repo, "formatIcsDate", limit=3)
    assert [row["symbol"] for row in results] == ["formatIcsDate"]
    MockEmbedder.assert_not_called()


def test_backticked_identifier_in_question_uses_text_lookup(repo):
    with (
        patch("lybrary.query.IndexStore") as MockStore,
        patch("lybrary.query.Embedder") as MockEmbedder,
    ):
        store = MockStore.return_value
        store.search_symbol.return_value = [_fake_row(symbol="formatIcsDate")]
        results = search(repo, "Where is `formatIcsDate` defined?", limit=3)
    assert [row["symbol"] for row in results] == ["formatIcsDate"]
    MockEmbedder.assert_not_called()


def test_broad_question_with_identifier_still_uses_semantic_search(repo):
    with (
        patch("lybrary.query.IndexStore") as MockStore,
        patch("lybrary.query.Embedder"),
    ):
        store = MockStore.return_value
        store.search_symbol.return_value = [_fake_row(symbol="sendInterviewPanelEmail")]
        store.search_text.return_value = []
        store.search_vector.return_value = [_fake_row(symbol="buildIcs")]
        results = search(repo, "create calendar attachment and sendInterviewPanelEmail")
    assert [row["symbol"] for row in results] == ["buildIcs"]


def test_conceptual_query_blends_text_and_vector_results(repo):
    with (
        patch("lybrary.query.IndexStore") as MockStore,
        patch("lybrary.query.Embedder"),
    ):
        store = MockStore.return_value
        store.search_text.return_value = [
            _fake_row(symbol="emailAttachment", path="src/comms.ts")
        ]
        store.search_vector.return_value = [
            _fake_row(symbol="unrelated", path="src/other.ts", distance=0.01)
        ]
        results = search(repo, "interview calendar attachment", limit=2)
    assert [row["symbol"] for row in results] == ["emailAttachment", "unrelated"]


def test_strong_lexical_match_avoids_embedding_model(repo):
    with (
        patch("lybrary.query.IndexStore") as MockStore,
        patch("lybrary.query.Embedder") as MockEmbedder,
    ):
        store = MockStore.return_value
        store.search_symbol.return_value = []
        store.search_text.return_value = [
            _fake_row(
                symbol="save_and_promote", path="harness_evolve.py",
                content="candidate evaluation passed promotion active",
            )
        ]
        results = search(repo, "candidate evaluation promotion", limit=3)
    assert [row["symbol"] for row in results] == ["save_and_promote"]
    MockEmbedder.assert_not_called()


def test_long_flow_query_keeps_later_promotion_terms():
    terms = query_terms(
        "Trace how dashboard POST /api/run reaches final candidate evaluation "
        "and Atlas promotion. What prevents a failed candidate from becoming active?"
    )
    assert "promotion" in terms
    assert "active" in terms


def test_lexical_flow_keeps_third_distinct_symbol_in_same_file(repo):
    with (
        patch("lybrary.query.IndexStore") as MockStore,
        patch("lybrary.query.Embedder") as MockEmbedder,
    ):
        store = MockStore.return_value
        store.search_symbol.return_value = []
        store.search_text.return_value = [
            _fake_row("run_evolution", "harness.py", "candidate evaluation active"),
            _fake_row("get_dashboard_status", "harness.py", "dashboard active candidate"),
            _fake_row("save_and_promote", "harness.py", "evaluation passed promotion active"),
            _fake_row("do_POST", "dashboard.py", "POST /api/run"),
        ]
        results = search(repo, "dashboard POST candidate evaluation promotion active", limit=5)
    assert "save_and_promote" in [row["symbol"] for row in results]
    MockEmbedder.assert_not_called()


def test_single_stemmed_lexical_hit_does_not_embed(repo):
    with (
        patch("lybrary.query.IndexStore") as MockStore,
        patch("lybrary.query.Embedder") as MockEmbedder,
    ):
        store = MockStore.return_value
        store.search_symbol.return_value = []
        store.search_text.return_value = [
            _fake_row("save_and_promote", "harness.py", "if evaluation passed: active = True")
        ]
        results = search(repo, "promotion")
    assert [row["symbol"] for row in results] == ["save_and_promote"]
    MockEmbedder.assert_not_called()


def test_conceptual_results_limit_near_duplicates_from_one_file(repo):
    rows = [
        _fake_row(symbol=f"template{i}", path="src/templates.ts", distance=i / 100)
        for i in range(4)
    ] + [_fake_row(symbol="sendEmail", path="src/send.ts", distance=0.5)]
    results = _run_search(repo, rows, query="calendar attachment flow", limit=4)
    assert [row["symbol"] for row in results] == [
        "template0", "template1", "sendEmail"
    ]
