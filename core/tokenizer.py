"""Tokenize method ASTs: split identifiers, drop keywords and frequent tokens."""

from __future__ import annotations

import ast
import keyword
import re
from collections import Counter

_SPLIT_BEFORE_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_SPLIT_ABBREV = re.compile(r"([A-Z]+)([A-Z][a-z])")
_WORD = re.compile(r"[a-z0-9]+")
_PY_KEYWORDS = set(keyword.kwlist)


def tokenize_method(method_ast: ast.AST) -> list[str]:
    """Return filtered tokens for a FunctionDef AST node.

    Includes method-name pieces, parameters, locals, called names, and
    string literals longer than 3 characters. Splits on `_` and camelCase.
    Drops Python keywords and single-character tokens. File-level DF
    stopwords (>30% of methods) are applied by `apply_df_filter`.
    """
    raw: list[str] = []
    _collect(method_ast, raw)
    tokens: list[str] = []
    seen: set[str] = set()
    for piece in raw:
        for tok in _split_ident(piece):
            if tok in _PY_KEYWORDS or len(tok) <= 1 or tok in seen:
                continue
            seen.add(tok)
            tokens.append(tok)
    return tokens


def apply_df_filter(method_tokens: list[list[str]]) -> list[list[str]]:
    """Drop tokens that appear in more than 30% of methods (and in ≥2 methods).

    A unique token is never treated as a stopword: with only 3 methods a
    single appearance is already 33%, which would otherwise wipe the set.
    """
    n = len(method_tokens)
    if n == 0:
        return method_tokens
    df: Counter[str] = Counter()
    for tokens in method_tokens:
        df.update(set(tokens))
    stop = {
        tok for tok, count in df.items()
        if count >= 2 and count / n > 0.30
    }
    return [[t for t in tokens if t not in stop] for tokens in method_tokens]


def _collect(node: ast.AST, raw: list[str]) -> None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raw.append(node.name)
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            raw.append(arg.arg)
        if node.args.vararg:
            raw.append(node.args.vararg.arg)
        if node.args.kwarg:
            raw.append(node.args.kwarg.arg)
        for child in node.body:
            _collect(child, raw)
        return

    if isinstance(node, ast.Call):
        name = _called_name(node.func)
        if name:
            raw.append(name)
        for child in ast.iter_child_nodes(node):
            _collect(child, raw)
        return

    if isinstance(node, ast.Name):
        raw.append(node.id)
        return

    if isinstance(node, ast.Attribute):
        raw.append(node.attr)
        _collect(node.value, raw)
        return

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if len(node.value) > 3:
            raw.extend(_WORD.findall(node.value.lower()))
        return

    for child in ast.iter_child_nodes(node):
        _collect(child, raw)


def _called_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _split_ident(name: str) -> list[str]:
    name = _SPLIT_BEFORE_CAMEL.sub(r"\1_\2", name)
    name = _SPLIT_ABBREV.sub(r"\1_\2", name)
    return [p for p in name.lower().split("_") if p]
