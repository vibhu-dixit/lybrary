import json

import pytest

from lybrary.mcp.server import UpdateInput, memory_update


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
