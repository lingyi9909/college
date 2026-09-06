"""Deterministic HTML/XML-ish source formatting normalization to Markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypedDict

from bs4 import BeautifulSoup, NavigableString, Tag
from bs4.element import PageElement

from college_builder.normalize.math import MathStatus, convert_mathml


class UnresolvedMathEvidence(TypedDict):
    code: str
    reason: str
    source_mathml: str


@dataclass(frozen=True)
class HtmlNormalizationResult:
    """Normalized text plus structural evidence extracted without semantic inference."""

    text: str
    images: tuple[str, ...]
    transformations: tuple[str, ...]
    unresolved_math: tuple[UnresolvedMathEvidence, ...]


class _Renderer:
    def __init__(self) -> None:
        self.images: list[str] = []
        self.unresolved_math: list[UnresolvedMathEvidence] = []
        self.math_converted = False
        self._tokens: dict[str, str] = {}

    def render(self, node: PageElement) -> str:
        if isinstance(node, NavigableString):
            return re.sub(r"\s+", " ", str(node))
        if not isinstance(node, Tag):
            return ""

        name = node.name.lower() if node.name else ""
        if name in {"p", "para", "div", "section", "article", "problem", "solution"}:
            return f"{self._render_children(node)}\n\n"
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(name[1])
            return f"{'#' * level} {self._render_children(node).strip()}\n\n"
        if name == "br":
            return "\n"
        if name == "hr":
            return "\n---\n"
        if name in {"strong", "b"}:
            return f"**{self._render_children(node)}**"
        if name in {"em", "i"}:
            return f"*{self._render_children(node)}*"
        if name == "code" and node.parent is not None and node.parent.name == "pre":
            return node.get_text()
        if name == "code":
            code = _normalize_newlines(node.get_text())
            fence = "``" if "`" in code else "`"
            return f"{fence}{code}{fence}"
        if name == "pre":
            source = _normalize_newlines(node.get_text())
            source = source.strip("\n")
            return f"{self._token(f'```\n{source}\n```')}\n\n"
        if name in {"ul", "ol"}:
            return self._render_list(node, ordered=name == "ol")
        if name == "blockquote":
            body = _cleanup_markdown(self._render_children(node))
            quoted = "\n".join(f"> {line}" if line else ">" for line in body.splitlines())
            return f"{quoted}\n\n"
        if name == "a":
            label = self._render_children(node).strip()
            href = _attribute(node, "href")
            return f"[{label}]({href})" if href else label
        if name == "img":
            src = _attribute(node, "src")
            alt = _attribute(node, "alt")
            if not src:
                return alt
            self.images.append(src)
            return f"![{alt}]({src})"
        if name == "math":
            return self._render_math(node)

        return self._render_children(node)

    def restore_tokens(self, text: str) -> str:
        restored = text
        for token, value in self._tokens.items():
            restored = restored.replace(token, value)
        return restored

    def _render_children(self, node: Tag) -> str:
        return "".join(self.render(child) for child in node.children)

    def _render_list(self, node: Tag, *, ordered: bool) -> str:
        lines: list[str] = []
        item_index = 0
        for child in node.children:
            if not isinstance(child, Tag) or child.name != "li":
                continue
            item_index += 1
            body = _cleanup_markdown(self._render_children(child)).replace("\n", " ")
            prefix = f"{item_index}. " if ordered else "- "
            lines.append(f"{prefix}{body.strip()}")
        return f"{'\n'.join(lines)}\n\n" if lines else ""

    def _render_math(self, node: Tag) -> str:
        source_mathml = str(node)
        converted = convert_mathml(source_mathml)
        if converted.status is MathStatus.MATH_UNRESOLVED:
            self.unresolved_math.append(
                {
                    "code": "MATH_UNRESOLVED",
                    "reason": converted.reason or "MATHML_CONVERSION_FAILED",
                    "source_mathml": converted.source_mathml,
                }
            )
            return self._token(converted.source_mathml)

        self.math_converted = True
        latex = converted.latex or ""
        display = _attribute(node, "display").lower() == "block"
        rendered = f"$${latex}$$" if display else f"${latex}$"
        return self._token(rendered)

    def _token(self, value: str) -> str:
        token = f"\ue000COLLEGE_NORMALIZE_{len(self._tokens)}\ue001"
        self._tokens[token] = value
        return token


def normalize_html(text: str) -> HtmlNormalizationResult:
    """Normalize deterministic formatting while leaving source semantics untouched."""
    normalized_newlines = _normalize_newlines(text)
    soup = BeautifulSoup(normalized_newlines, "html.parser")
    has_markup = soup.find(True) is not None

    transformations: list[str] = []
    if normalized_newlines != text:
        transformations.append("NEWLINES_NORMALIZED")

    if not has_markup:
        return HtmlNormalizationResult(
            text=_normalize_plain_text(normalized_newlines),
            images=(),
            transformations=tuple(transformations),
            unresolved_math=(),
        )

    renderer = _Renderer()
    rendered = "".join(renderer.render(child) for child in soup.contents)
    cleaned = _cleanup_markdown(rendered)
    cleaned = renderer.restore_tokens(cleaned)
    transformations.append("HTML_TO_MARKDOWN")
    if renderer.math_converted:
        transformations.append("MATHML_TO_LATEX")

    return HtmlNormalizationResult(
        text=cleaned,
        images=_stable_unique(renderer.images),
        transformations=tuple(transformations),
        unresolved_math=tuple(renderer.unresolved_math),
    )


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_plain_text(text: str) -> str:
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _cleanup_markdown(text: str) -> str:
    text = _normalize_newlines(text)
    output: list[str] = []
    previous_blank = False
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            if output and not previous_blank:
                output.append("")
            previous_blank = True
            continue
        output.append(line)
        previous_blank = False
    while output and output[-1] == "":
        output.pop()
    return "\n".join(output)


def _attribute(tag: Tag, name: str) -> str:
    value = tag.attrs.get(name)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return "" if value is None else str(value)


def _stable_unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)
