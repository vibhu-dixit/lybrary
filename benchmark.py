"""Compare read-only Codex runs with and without Lybrary.

Usage: python benchmark.py C:/path/to/repo "Trace how interview invites are sent"
Traces are saved under the repo's ignored .lybrary/benchmarks directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def summarize_trace(path: Path) -> dict[str, int | None]:
    """Extract measured usage and tool counts from Codex JSONL events."""
    input_tokens = cached_input_tokens = output_tokens = None
    memory_queries = shell_commands = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") == "turn.completed" and (usage := event.get("usage")):
            input_tokens = (input_tokens or 0) + usage.get("input_tokens", 0)
            cached_input_tokens = (cached_input_tokens or 0) + usage.get(
                "cached_input_tokens", 0
            )
            output_tokens = (output_tokens or 0) + usage.get("output_tokens", 0)
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            tool_name = str(item.get("name", item.get("tool", "")))
            if item.get("type") == "mcp_tool_call" and "memory_query" in tool_name:
                memory_queries += 1
            if item.get("type") == "command_execution":
                shell_commands += 1
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": (
            input_tokens - cached_input_tokens
            if input_tokens is not None and cached_input_tokens is not None else None
        ),
        "output_tokens": output_tokens,
        "memory_queries": memory_queries,
        "shell_commands": shell_commands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="Indexed repository root")
    parser.add_argument("prompt", help="Same read-only question for both runs")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not (repo / ".lybrary" / "index.db").is_file():
        parser.error(f"No Lybrary index in {repo}")
    benchmark_dir = repo / ".lybrary" / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="run-", dir=benchmark_dir))
    prompts = {
        "without": "Do not use Lybrary or memory_query. Do not edit files. " + args.prompt,
        "with": (
            "Use Lybrary memory_query to locate code before reading files. "
            "Do not edit files. " + args.prompt
        ),
    }
    summary = {}
    for name, prompt in prompts.items():
        trace = out / f"{name}.jsonl"
        command = ["codex", "exec", "--json", "--sandbox", "read-only", "-C", str(repo), prompt]
        with trace.open("w", encoding="utf-8") as output:
            run = subprocess.run(
                command, stdout=output, stderr=subprocess.PIPE, text=True, check=False
            )
        if run.returncode:
            raise SystemExit(f"{name} run failed ({run.returncode}): {run.stderr[-1000:]}")
        summary[name] = {"trace": str(trace), **summarize_trace(trace)}
    before = summary["without"]["input_tokens"]
    after = summary["with"]["input_tokens"]
    summary["input_token_change_percent"] = (
        round(100 * (after - before) / before, 1)
        if before and after is not None else None
    )
    before_uncached = summary["without"]["uncached_input_tokens"]
    after_uncached = summary["with"]["uncached_input_tokens"]
    summary["uncached_input_token_change_percent"] = (
        round(100 * (after_uncached - before_uncached) / before_uncached, 1)
        if before_uncached and after_uncached is not None else None
    )
    summary["note"] = (
        "Review both answers for correctness; one pair is not proof of typical savings."
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
