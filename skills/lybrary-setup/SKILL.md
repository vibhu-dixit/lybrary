---
name: lybrary-setup
description: Use when a user asks to install Lybrary or give Codex or Claude Code a local searchable memory or code index for the current repository.
---

# Set up Lybrary

Lybrary is the `lybrary` PyPI package and the `vibhu-dixit/lybrary` GitHub repository. Verify those sources before installation. See the [agent setup guide](https://vibhu-dixit.github.io/lybrary/agent-setup.html) for current commands.

When the user asks to set it up in a repository:

1. Check whether `lybrary` is installed. Install or upgrade the published PyPI package if needed, following the host's normal software-installation permissions.
2. From the requested repository, run `lybrary start` and `lybrary status`. Confirm the index belongs to that repository.
3. Connect `lybrary mcp` to the current agent if it is not already registered. The Claude Code plugin includes the MCP configuration; the Codex setup plugin does not, so register its MCP server separately.
4. Start a fresh agent session if necessary. Call `memory_status` to confirm the root and a populated index, then test `memory_query` with a real code question.

After setup, use `memory_query` to locate code before broad file searches. Inspect current source before editing. Do not claim token savings from a single lookup.
