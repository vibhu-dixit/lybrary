"""AST-aware code chunking using tree-sitter (cAST-style)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tree_sitter import Language, Parser, Node

# Language → tree-sitter language module mapping
_LANGUAGE_MODULES: dict[str, Any] = {}


def _load_language(lang: str) -> Language | None:
    global _LANGUAGE_MODULES
    if lang in _LANGUAGE_MODULES:
        return _LANGUAGE_MODULES[lang]

    try:
        if lang == "python":
            import tree_sitter_python as mod
        elif lang == "javascript":
            import tree_sitter_javascript as mod
        elif lang in ("typescript", "tsx"):
            import tree_sitter_typescript as mod
            # The typescript package exposes two grammars; load both at once.
            _LANGUAGE_MODULES["typescript"] = Language(mod.language_typescript())
            _LANGUAGE_MODULES["tsx"] = Language(mod.language_tsx())
            return _LANGUAGE_MODULES[lang]  # Fix #1: return the grammar that was actually requested
        elif lang == "go":
            import tree_sitter_go as mod
        elif lang == "rust":
            import tree_sitter_rust as mod
        elif lang == "java":
            import tree_sitter_java as mod
        elif lang == "c":
            import tree_sitter_c as mod
        elif lang == "cpp":
            import tree_sitter_cpp as mod
        else:
            return None

        lang_obj = Language(mod.language())
        _LANGUAGE_MODULES[lang] = lang_obj
        return lang_obj
    except Exception:
        return None


# Node types that represent meaningful semantic units per language
DEFINITION_TYPES: dict[str, set[str]] = {
    "python": {
        "function_definition",
        "class_definition",
        "decorated_definition",
    },
    "javascript": {
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "method_definition",
        "export_statement",
        "lexical_declaration",  # const/let often hold arrow functions
    },
    "typescript": {
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "method_definition",
        "export_statement",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "lexical_declaration",
    },
    "tsx": {
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "method_definition",
        "export_statement",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "lexical_declaration",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
    },
    "rust": {
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "mod_item",
    },
    "java": {
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "constructor_declaration",
        "enum_declaration",
    },
    "c": {
        "function_definition",
        "struct_specifier",
        "type_definition",
    },
    "cpp": {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
    },
}

# Node types that are "containers" — we emit them as a chunk AND recurse
# into their children to emit nested definitions as separate chunks too.
_CONTAINER_TYPES: dict[str, set[str]] = {
    "python": {"class_definition", "decorated_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration"},
    "tsx": {"class_declaration"},
    "go": set(),
    "rust": {"impl_item", "mod_item"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration"},
    "c": set(),
    "cpp": {"class_specifier", "struct_specifier", "namespace_definition"},
}

EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",  # Fix #1: tsx kept as a distinct language key
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
}


@dataclass
class CodeChunk:
    """A single semantically coherent code chunk."""

    id: str
    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    content: str
    language: str
    scope_chain: list[str] = field(default_factory=list)
    signature: str = ""
    context_header: str = ""
    dependencies: list[str] = field(default_factory=list)
    byte_start: int = 0
    byte_end: int = 0

    def embedding_text(self) -> str:
        """Text used for embedding (header + content)."""
        header = self.context_header or f"// {self.path} :: {self.symbol}"
        return f"{header}\n{self.content}"


def detect_language(path: Path) -> str | None:
    return EXT_TO_LANG.get(path.suffix.lower())


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _approx_tokens(text: str) -> int:
    # rough heuristic: ~4 chars per token for code
    return max(1, len(text) // 4)


def _sanitize_name(name: str) -> str:
    """Fix #13: strip characters that corrupt chunk IDs or SQL filters."""
    # Remove characters that break the "path::name:line" ID format
    # and the LanceDB filter string interpolation.
    return name.replace('"', "'").replace("\n", " ").replace("::", "__")[:80]


def _extract_name(node: Node, source: bytes, lang: str) -> str:
    """Best-effort symbol name extraction."""
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier", "property_identifier"):
            return _sanitize_name(_node_text(child, source))
        # Python decorated_definition wraps the real def
        if child.type in ("function_definition", "class_definition"):
            return _extract_name(child, source, lang)
    # Fix #13: fallback uses sanitized first line so IDs stay safe
    text = _node_text(node, source)
    first = text.strip().split("\n", 1)[0]
    return _sanitize_name(first)


def _collect_definitions(
    node: Node,
    source: bytes,
    lang: str,
    results: list[tuple[Node, str, str]],
    depth: int = 0,
) -> None:
    """Recursively collect definition nodes.

    Fix #3: Nested definitions inside container nodes (classes, impls, …) are
    collected as children, but the container itself is also kept as a whole-body
    chunk.  Non-container definitions (plain functions) are NOT recursed into, so
    their inner helpers don't produce duplicate top-level entries.
    """
    def_types = DEFINITION_TYPES.get(lang, set())
    container_types = _CONTAINER_TYPES.get(lang, set())

    if node.type in def_types:
        kind = node.type
        name = _extract_name(node, source, lang)
        results.append((node, name, kind))

        if node.type in container_types:
            # Recurse to also emit nested methods/functions as own chunks
            for child in node.children:
                _collect_definitions(child, source, lang, results, depth + 1)
        # For plain functions/methods, do NOT recurse — inner helpers are part
        # of the parent chunk, not separate top-level chunks.
        return

    for child in node.children:
        _collect_definitions(child, source, lang, results, depth + 1)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Fix #4: Merge overlapping/adjacent byte intervals."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals)
    merged: list[tuple[int, int]] = [sorted_ivs[0]]
    for start, end in sorted_ivs[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def chunk_source(
    source: str,
    path: str,
    language: str,
    soft_max_tokens: int = 1200,
    min_chunk_tokens: int = 50,
) -> list[CodeChunk]:
    """
    Chunk source code into structure-aware units.

    Strategy (cAST-inspired):
    1. Parse with tree-sitter
    2. Extract definition nodes (containers emit self + children; plain defs emit self only)
    3. Emit complete definitions as chunks
    4. Group leftover top-level statements into one or more chunks
    """
    # Fix #1: use the language as-is; _load_language handles "tsx" correctly now.
    lang_obj = _load_language(language)

    if lang_obj is None:
        return _fallback_line_chunks(source, path, language, soft_max_tokens)

    parser = Parser(lang_obj)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    definitions: list[tuple[Node, str, str]] = []
    _collect_definitions(root, source_bytes, language, definitions)

    chunks: list[CodeChunk] = []
    # Fix #4: collect byte ranges and merge overlaps before computing leftovers
    raw_ranges: list[tuple[int, int]] = []

    seen_ids: set[str] = set()

    for node, name, kind in definitions:
        text = _node_text(node, source_bytes)

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        scope = [path]
        if kind.startswith("class") or "class" in kind:
            scope.append(f"class {name}")
        header = f"// File: {path} → {kind} {name}"

        # Fix #2: include byte_start in the ID to guarantee uniqueness even
        # when two symbols share the same name and line (e.g., overloads).
        chunk_id = f"{path}::{name}:{start_line}:{node.start_byte}"
        # Deduplicate: if somehow an identical ID was already added, skip.
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        chunks.append(
            CodeChunk(
                id=chunk_id,
                path=path,
                symbol=name,
                kind=kind,
                start_line=start_line,
                end_line=end_line,
                content=text,
                language=language,
                scope_chain=scope,
                signature=text.strip().split("\n", 1)[0][:120],
                context_header=header,
                byte_start=node.start_byte,
                byte_end=node.end_byte,
            )
        )
        raw_ranges.append((node.start_byte, node.end_byte))

    # Fix #4: merge overlapping intervals before computing leftover bytes
    covered_ranges = _merge_intervals(raw_ranges)

    # Collect uncovered top-level content (imports, constants, etc.)
    leftovers: list[str] = []
    cursor = 0
    for start, end in covered_ranges:
        if cursor < start:
            piece = source_bytes[cursor:start].decode("utf-8", errors="replace").strip()
            if piece:
                leftovers.append(piece)
        cursor = max(cursor, end)
    if cursor < len(source_bytes):
        piece = source_bytes[cursor:].decode("utf-8", errors="replace").strip()
        if piece:
            leftovers.append(piece)

    if leftovers:
        combined = "\n\n".join(leftovers)
        if _approx_tokens(combined) >= min_chunk_tokens:
            chunks.append(
                CodeChunk(
                    id=f"{path}::__module__:1:0",
                    path=path,
                    symbol="__module__",
                    kind="module_level",
                    start_line=1,
                    end_line=source.count("\n") + 1,
                    content=combined,
                    language=language,
                    scope_chain=[path],
                    context_header=f"// File: {path} → module-level (imports, constants, etc.)",
                )
            )

    # Stable order by start line
    chunks.sort(key=lambda c: c.start_line)
    return chunks


def chunk_file(
    path: Path,
    root: Path,
    soft_max_tokens: int = 1200,
    min_chunk_tokens: int = 50,
) -> list[CodeChunk]:
    """Chunk a single file on disk."""
    lang = detect_language(path)
    if not lang:
        return []

    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    rel = str(path.resolve().relative_to(root.resolve()))
    # Fix #1: pass lang directly — no longer strips "tsx" to "typescript".
    return chunk_source(
        source,
        rel,
        lang,
        soft_max_tokens=soft_max_tokens,
        min_chunk_tokens=min_chunk_tokens,
    )


def _fallback_line_chunks(
    source: str,
    path: str,
    language: str,
    soft_max_tokens: int,
) -> list[CodeChunk]:
    """Simple line-based fallback when tree-sitter is unavailable."""
    lines = source.splitlines(keepends=True)
    chunks: list[CodeChunk] = []
    current: list[str] = []
    start_line = 1
    approx = 0

    for i, line in enumerate(lines, 1):
        current.append(line)
        approx += _approx_tokens(line)
        if approx >= soft_max_tokens:
            text = "".join(current)
            chunks.append(
                CodeChunk(
                    id=f"{path}::lines:{start_line}:{start_line}",
                    path=path,
                    symbol=f"lines_{start_line}_{i}",
                    kind="block",
                    start_line=start_line,
                    end_line=i,
                    content=text,
                    language=language,
                    scope_chain=[path],
                    context_header=f"// File: {path} (lines {start_line}-{i})",
                )
            )
            current = []
            approx = 0
            start_line = i + 1

    if current:
        text = "".join(current)
        end = start_line + len(current) - 1
        chunks.append(
            CodeChunk(
                id=f"{path}::lines:{start_line}:{start_line}",
                path=path,
                symbol=f"lines_{start_line}_{end}",
                kind="block",
                start_line=start_line,
                end_line=end,
                content=text,
                language=language,
                scope_chain=[path],
                context_header=f"// File: {path} (lines {start_line}-{end})",
            )
        )
    return chunks
