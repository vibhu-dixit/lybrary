"""Exercise the published stdio MCP entry point as a client."""

import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.skipif(
    sys.platform == "win32", reason="MCP SDK stdio subprocess teardown hangs on Windows"
)
@pytest.mark.asyncio
async def test_stdio_server_exposes_memory_tools(tmp_path: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    server = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from lybrary.mcp.server import run; run()"],
        env=env,
        cwd=str(tmp_path),
    )
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "memory_query", "memory_status", "memory_update"
            }
            status = await session.call_tool(
                "memory_status", {"params": {"repo_root": str(tmp_path)}}
            )
            assert '"chunks": 0' in status.content[0].text
