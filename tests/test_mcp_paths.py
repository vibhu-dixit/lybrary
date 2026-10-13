import json

import pytest
from pydantic import ValidationError

from lybrary.mcp.server import QueryInput, UpdateInput, _resolve_root, memory_query, memory_update


def test_query_limit_prevents_large_result_sets():
    with pytest.raises(ValidationError):
        QueryInput(query="interview invitation", limit=20)


@pytest.mark.asyncio
async def test_targeted_update_rejects_path_outside_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def keep(): pass")
    result = json.loads(
        await memory_update(UpdateInput(repo_root=str(repo), paths=["../outside.py"]))
    )
    assert "error" in result
    assert outside.read_text() == "def keep(): pass"
    assert not (repo / ".lybrary").exists()


def test_parent_folder_points_to_indexed_child(tmp_path):
    child = tmp_path / "core" / ".lybrary"
    child.mkdir(parents=True)
    (child / "index.db").touch()
    with pytest.raises(ValueError, match=r"core.*repo_root"):
        _resolve_root(str(tmp_path))


@pytest.mark.asyncio
async def test_memory_query_returns_compact_locations_by_default(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    row = {
        "path": "src/util.ts", "symbol": "buildIcs", "kind": "function",
        "start_line": 24, "end_line": 80, "score": 0.8,
        "content": (
            "export function buildIcs() {\nconst x = 1;\n"
            "const attachment = 'interview.ics';\n" + "x" * 1000
        ),
        "context_header": "util.ts",
    }
    monkeypatch.setattr("lybrary.query.search", lambda *args, **kwargs: [row])
    result = json.loads(
        await memory_query(QueryInput(repo_root=str(repo), query="calendar invite"))
    )
    chunk = result["chunks"][0]
    assert chunk["path"] == "src/util.ts"
    assert chunk["start_line"] == 24
    assert "content" not in chunk
    assert "26: const attachment = 'interview.ics';" in chunk["excerpt"]
    assert len(chunk["excerpt"]) < 750


@pytest.mark.asyncio
async def test_memory_query_can_return_full_content(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    row = {
        "path": "src/util.ts", "symbol": "buildIcs", "kind": "function",
        "start_line": 24, "end_line": 26, "score": 0.8,
        "content": "function buildIcs() {}", "context_header": "util.ts",
    }
    monkeypatch.setattr("lybrary.query.search", lambda *args, **kwargs: [row])
    result = json.loads(
        await memory_query(
            QueryInput(repo_root=str(repo), query="buildIcs", include_content=True)
        )
    )
    assert result["chunks"][0]["content"] == "function buildIcs() {}"


@pytest.mark.asyncio
async def test_excerpt_prefers_distinctive_matches_over_generic_status_log(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    rows = [
        {
            "path": "comms.ts", "symbol": "sendInvitation", "kind": "function",
            "start_line": 100, "end_line": 110, "score": 0.8,
            "content": (
                "function sendInvitation() {\n// calendar attachment\nconst ics = buildIcs();\n"
                "const bytes = toBase64(ics);\nconst attachments = [bytes];\n"
                "sendEmail(attachments);\nconsole.log('interview email sent');\n}"
            ),
        },
        {
            "path": "email.ts", "symbol": "sendEmail", "kind": "function",
            "start_line": 1, "end_line": 2, "score": 0.6,
            "content": "function sendEmail() { console.log('interview email sent'); }",
        },
    ]
    monkeypatch.setattr("lybrary.query.search", lambda *args, **kwargs: rows)
    result = json.loads(await memory_query(QueryInput(
        repo_root=str(repo), query="interview calendar attachment sent email"
    )))
    excerpt = result["chunks"][0]["excerpt"]
    assert "101: // calendar attachment" in excerpt
    assert "interview email sent" not in excerpt


@pytest.mark.asyncio
async def test_compact_response_shows_attachment_and_send_steps(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    lines = ["function invite() {"] + ["const unrelated = 0;"] * 12 + [
        "attachments = [{ filename: 'interview.ics', content: toBase64(ics) }];",
        "const unused = 1;",
        "const unused2 = 2;",
        "const unused3 = 3;",
        "const unused4 = 4;",
        "await sendEmailWithResend({ to: candidate, attachments });",
        "console.log('interview email sent');",
        "}",
    ]
    lines[3] = "const calendar = buildCalendarLink();"
    row = {
        "path": "comms.ts", "symbol": "invite", "kind": "function",
        "start_line": 100, "end_line": 120, "score": 0.9,
        "content": "\n".join(lines), "context_header": "comms.ts",
    }
    monkeypatch.setattr("lybrary.query.search", lambda *args, **kwargs: [row])
    result = json.loads(await memory_query(QueryInput(
        repo_root=str(repo), query="interview calendar attachment built and sent email"
    )))
    excerpt = result["chunks"][0]["excerpt"]
    assert "103: const calendar =" in excerpt
    assert "113: attachments =" in excerpt
    assert "118: await sendEmailWithResend" in excerpt
    assert "console.log" not in excerpt
    assert len(excerpt) < 650


@pytest.mark.asyncio
async def test_compact_response_keeps_only_matching_log_line(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    row = {
        "path": "audit.ts", "symbol": "logAudit", "kind": "function",
        "start_line": 7, "end_line": 7, "score": 0.8,
        "content": "console.log('audit event');", "context_header": "audit.ts",
    }
    monkeypatch.setattr("lybrary.query.search", lambda *args, **kwargs: [row])
    result = json.loads(await memory_query(QueryInput(
        repo_root=str(repo), query="audit event"
    )))
    assert result["chunks"][0]["excerpt"] == "7: console.log('audit event');"
