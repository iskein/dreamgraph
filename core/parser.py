"""AST parser that extracts Method records from a Python file."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from core.tokenizer import apply_df_filter, tokenize_method


@dataclass
class Method:
    fqn: str               # "Scorer.score_keyword"
    name: str              # "score_keyword"
    file: str              # "data/scorer.py"
    lines: tuple[int, int] # (start, end)
    docstring: str
    source: str            # full source of the method
    calls: list[str]       # FQNs of methods called within
    tokens: list[str]      # identifiers + params + locals + called names


@dataclass
class _Parsed:
    node: ast.FunctionDef | ast.AsyncFunctionDef
    class_name: str | None
    method: Method = field(init=False)


def parse_file(path: str) -> list[Method]:
    """Walk the AST of `path` and return one Method per function/method."""
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    file_label = path
    parsed: list[_Parsed] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parsed.append(_Parsed(node=node, class_name=None))
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    parsed.append(_Parsed(node=child, class_name=node.name))

    methods: list[Method] = []
    raw_token_lists: list[list[str]] = []
    for item in parsed:
        fqn = f"{item.class_name}.{item.node.name}" if item.class_name else item.node.name
        start = item.node.lineno
        end = item.node.end_lineno or item.node.lineno
        methods.append(Method(
            fqn=fqn,
            name=item.node.name,
            file=file_label,
            lines=(start, end),
            docstring=ast.get_docstring(item.node) or "",
            source=ast.get_source_segment(source, item.node) or "",
            calls=_collect_calls(item.node, item.class_name),
            tokens=[],
        ))
        raw_token_lists.append(tokenize_method(item.node))

    filtered = apply_df_filter(raw_token_lists)
    for method, tokens in zip(methods, filtered):
        method.tokens = tokens
    return methods


def _collect_calls(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    class_name: str | None,
) -> list[str]:
    calls: list[str] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        fqn = _call_fqn(node.func, class_name)
        if fqn and fqn not in calls:
            calls.append(fqn)
    return calls


def _call_fqn(func: ast.AST, class_name: str | None) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            if func.value.id in ("self", "cls") and class_name:
                return f"{class_name}.{func.attr}"
            return f"{func.value.id}.{func.attr}"
        return func.attr
    return None
