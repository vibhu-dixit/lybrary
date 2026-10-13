"""A saved Codex trace must produce comparable token and tool-call counts."""

import json

from benchmark import summarize_trace


def test_summarize_trace_counts_usage_and_memory_calls(tmp_path):
    trace = tmp_path / "run.jsonl"
    events = [
        {
            "type": "item.completed",
            "item": {"type": "mcp_tool_call", "name": "lybrary.memory_query"},
        },
        {"type": "item.completed", "item": {"type": "command_execution", "command": "rg foo"}},
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 1200,
                "cached_input_tokens": 900,
                "output_tokens": 300,
            },
        },
    ]
    trace.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    assert summarize_trace(trace) == {
        "input_tokens": 1200,
        "cached_input_tokens": 900,
        "uncached_input_tokens": 300,
        "output_tokens": 300,
        "memory_queries": 1,
        "shell_commands": 1,
    }
