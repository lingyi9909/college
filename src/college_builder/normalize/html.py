"""Deterministic HTML/CNXML formatting normalization without semantic rewriting."""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass
from typing import TypedDict

from bs4 import BeautifulSoup
from bs4.element import NavigableString, PageElement, Tag

from college_builder.normalize.math import MathStatus, convert_mathml


class UnresolvedMathEvidence(TypedDict):
    code: str
    reason: str
    source_mathml: str


class UnresolvedStructureEvidence(TypedDict):
    code: str
    reason: str
    source_markup: str


@dataclass(frozen=True)
class HtmlNormalizationResult:
    """Normalized text plus structural evidence extracted without semantic inference."""

    text: str
    images: tuple[str, ...]
    transformations: tuple[str, ...]
    unresolved_math: tuple[UnresolvedMathEvidence, ...]
    unresolved_structure: tuple[UnresolvedStructureEvidence, ...] = ()


_BLOCK_TAGS = frozenset(
    {
        "p",
        "para",
        "div",
        "section",
        "article",
        "problem",
        "solution",
        "content",
        "document",
        "exercise",
        "body",
        "html",
    }
)
_LIST_TAGS = frozenset({"ul", "ol"})
_MATH_START_RE = re.compile(
    r"<(?P<qname>(?:[A-Za-z_][\w.-]*:)?math)\b[^>]*>",
    flags=re.IGNORECASE,
)
_BLOCK_CLOSE_RE = re.compile(
    r"</(?:[A-Za-z_][\w.-]*:)?(?:p|para|div|section|article|problem|solution|li)\s*>",
    flags=re.IGNORECASE,
)
_NAMESPACE_RE = re.compile(
    r"\bxmlns:(?P<prefix>[A-Za-z_][\w.-]*)\s*=\s*['\"](?P<uri>[^'\"]+)['\"]"
)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]\n]*\]\((?P<target>[^)\n]+)\)")


class _SourceProtector:
    """Protect source-native literal syntax before any HTML repair/whitespace handling."""

    def __init__(self, original: str) -> None:
        self._original = original
        self._tokens: dict[str, str] = {}
        self.images: list[str] = []
        self.unresolved_math: list[UnresolvedMathEvidence] = []
        self.math_converted = False

    def protect(self, text: str) -> str:
        protected = self._protect_fenced_code(text)
        protected = self._protect_inline_code(protected)
        protected = self._protect_mathml(protected)
        protected = self._protect_latex(protected)
        protected = self._protect_markdown_images(protected)
        return protected

    def token(self, value: str) -> str:
        index = len(self._tokens)
        token = f"COLLEGEZNORMZTOKENZ{index:08d}Z"
        while token in self._original or token in self._tokens:
            index += 1
            token = f"COLLEGEZNORMZTOKENZ{index:08d}Z"
        self._tokens[token] = value
        return token

    def restore(self, text: str) -> str:
        restored = text
        for _ in range(3):
            changed = False
            for token, value in self._tokens.items():
                if token in restored:
                    restored = restored.replace(token, value)
                    changed = True
            if not changed:
                break
        return restored

    def _protect_fenced_code(self, text: str) -> str:
        lines = text.splitlines(keepends=True)
        output: list[str] = []
        index = 0
        while index < len(lines):
            line_without_newline = lines[index].rstrip("\n")
            opening = re.match(r"^[ \t]*(`{3,}|~{3,})", line_without_newline)
            if opening is None:
                output.append(lines[index])
                index += 1
                continue

            fence = opening.group(1)
            fence_character = re.escape(fence[0])
            closing = re.compile(
                rf"^[ \t]*{fence_character}{{{len(fence)},}}[ \t]*$"
            )
            end = index + 1
            while end < len(lines):
                candidate = lines[end].rstrip("\n")
                if closing.match(candidate):
                    end += 1
                    break
                end += 1
            fragment = "".join(lines[index:end])
            output.append(self.token(fragment))
            index = end
        return "".join(output)

    def _protect_inline_code(self, text: str) -> str:
        pattern = re.compile(r"(?<!`)(`+)([^`\n]*?)\1(?!`)")
        return pattern.sub(lambda match: self.token(match.group(0)), text)

    def _protect_latex(self, text: str) -> str:
        patterns = (
            re.compile(r"(?s)(?<!\\)\$\$(.+?)(?<!\\)\$\$"),
            re.compile(r"(?s)\\\[(.+?)\\\]"),
            re.compile(r"(?s)\\\((.+?)\\\)"),
            re.compile(r"(?s)(?<![$\\])\$(?!\$)(.+?)(?<!\\)\$(?!\$)"),
        )
        protected = text
        for pattern in patterns:
            protected = pattern.sub(lambda match: self.token(match.group(0)), protected)
        return protected

    def _protect_markdown_images(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            target = match.group("target").strip()
            url = _markdown_target_url(target)
            if url:
                self.images.append(url)
            return self.token(match.group(0))

        return _MARKDOWN_IMAGE_RE.sub(replace, text)

    def _protect_mathml(self, text: str) -> str:
        namespace_context = _namespace_context(text)
        output: list[str] = []
        position = 0
        while True:
            match = _MATH_START_RE.search(text, position)
            if match is None:
                output.append(text[position:])
                break

            output.append(text[position : match.start()])
            qname = match.group("qname")
            close_re = re.compile(rf"</{re.escape(qname)}\s*>", flags=re.IGNORECASE)
            close = close_re.search(text, match.end())
            if close is not None:
                end = close.end()
            else:
                block_close = _BLOCK_CLOSE_RE.search(text, match.end())
                end = block_close.start() if block_close is not None else len(text)

            source_mathml = text[match.start() : end]
            converted = convert_mathml(
                source_mathml,
                namespace_context=namespace_context,
            )
            if converted.status is MathStatus.MATH_UNRESOLVED:
                self.unresolved_math.append(
                    {
                        "code": "MATH_UNRESOLVED",
                        "reason": converted.reason or "MATHML_CONVERSION_FAILED",
                        "source_mathml": source_mathml,
                    }
                )
                replacement = source_mathml
            else:
                self.math_converted = True
                latex = converted.latex or ""
                replacement = f"$${latex}$$" if _is_display_math(source_mathml) else f"${latex}$"
            output.append(self.token(replacement))
            position = end
        return "".join(output)


class _Renderer:
    def __init__(self, protector: _SourceProtector) -> None:
        self._protector = protector
        self.images = protector.images
        self.unresolved_math = protector.unresolved_math
        self.unresolved_structure: list[UnresolvedStructureEvidence] = []
        self.math_converted = protector.math_converted

    def render(self, node: PageElement) -> str:
        if isinstance(node, NavigableString):
            value = str(node)
            if not value.strip():
                return "" if "\n" in value else value
            return re.sub(r"\s+", " ", value)
        if not isinstance(node, Tag):
            return ""

        name = _local_name(node.name)
        if name in _BLOCK_TAGS:
            body = _cleanup_markdown(self._render_children(node)).strip()
            return f"{body}\n\n" if body else ""
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(name[1])
            return f"{'#' * level} {self._render_children(node).strip()}\n\n"
        if name == "title":
            body = self._render_children(node).strip()
            return f"{body}\n\n" if body else ""
        if name == "br":
            return "\n"
        if name == "hr":
            return "\n---\n"
        if name in {"strong", "b"}:
            return f"**{self._render_children(node)}**"
        if name in {"em", "i"}:
            return f"*{self._render_children(node)}*"
        if name == "emphasis":
            return self._render_emphasis(node)
        if name == "sup":
            return f"^{{{self._render_children(node).strip()}}}"
        if name == "sub":
            return f"_{{{self._render_children(node).strip()}}}"
        if name == "span":
            if node.attrs:
                return self._preserve_structure(node, "UNSUPPORTED_SPAN_ATTRIBUTES")
            return self._render_children(node)
        if name == "code" and node.parent is not None and _local_name(node.parent.name) == "pre":
            return node.get_text()
        if name == "code":
            code = _normalize_newlines(node.get_text())
            fence = "``" if "`" in code else "`"
            return f"{fence}{code}{fence}"
        if name == "pre":
            source = _normalize_newlines(node.get_text()).strip("\n")
            return f"\n{self._protector.token(f'```\n{source}\n```')}\n\n"
        if name in _LIST_TAGS:
            return self._render_list(node, ordered=name == "ol")
        if name == "blockquote":
            return self._render_blockquote(node)
        if name in {"a", "link"}:
            return self._render_link(node)
        if name in {"img", "image"}:
            return self._render_image(node)
        if name == "table":
            return self._render_table(node)
        if name == "math":
            return self._preserve_structure(node, "UNPROTECTED_MATHML")

        return self._preserve_structure(node, f"UNSUPPORTED_TAG:{name}")

    def _render_children(self, node: Tag) -> str:
        return "".join(self.render(child) for child in node.children)

    def _render_emphasis(self, node: Tag) -> str:
        effect = _attribute(node, "effect").lower()
        body = self._render_children(node)
        if effect in {"italics", "italic"}:
            return f"*{body}*"
        if effect in {"bold", "strong"}:
            return f"**{body}**"
        if not effect:
            return body
        return self._preserve_structure(node, f"UNSUPPORTED_EMPHASIS_EFFECT:{effect}")

    def _render_link(self, node: Tag) -> str:
        label = self._render_children(node).strip()
        target = _attribute(node, "href") or _attribute(node, "url")
        return f"[{label}]({target})" if target else label

    def _render_image(self, node: Tag) -> str:
        src = _attribute(node, "src")
        alt = _attribute(node, "alt")
        if not src:
            return alt
        self.images.append(src)
        return f"![{alt}]({src})"

    def _render_list(self, node: Tag, *, ordered: bool) -> str:
        settings = _ordered_list_settings(node) if ordered else (1, "1")
        if settings is None:
            return self._preserve_structure(node, "UNSUPPORTED_ORDERED_LIST_ATTRIBUTES")
        current, marker_type = settings
        lines: list[str] = []

        for child in node.children:
            if not isinstance(child, Tag) or _local_name(child.name) != "li":
                continue
            if ordered:
                value = _optional_int_attribute(child, "value")
                if value is False:
                    return self._preserve_structure(node, "UNSUPPORTED_LIST_ITEM_VALUE")
                if isinstance(value, int):
                    current = value
                marker = _ordered_marker(current, marker_type)
                if marker is None:
                    return self._preserve_structure(node, "UNSUPPORTED_ORDERED_LIST_VALUE")
                current += 1
            else:
                marker = "-"

            direct_parts: list[str] = []
            nested: list[Tag] = []
            for item_child in child.children:
                if isinstance(item_child, Tag) and _local_name(item_child.name) in _LIST_TAGS:
                    nested.append(item_child)
                else:
                    direct_parts.append(self.render(item_child))

            body = self._protector.restore(_cleanup_markdown("".join(direct_parts))).strip("\n")
            body_lines = body.splitlines() if body else [""]
            lines.append(f"{marker} {body_lines[0].strip()}".rstrip())
            for continuation in body_lines[1:]:
                lines.append(f"  {continuation}" if continuation else "")

            for nested_list in nested:
                nested_text = self._protector.restore(
                    self._render_list(
                        nested_list,
                        ordered=_local_name(nested_list.name) == "ol",
                    )
                ).strip("\n")
                lines.extend(f"  {line}" if line else "" for line in nested_text.splitlines())

        rendered = "\n".join(lines)
        return f"{self._protector.token(rendered)}\n\n" if rendered else ""

    def _render_blockquote(self, node: Tag) -> str:
        body = self._protector.restore(_cleanup_markdown(self._render_children(node))).strip("\n")
        quoted = "\n".join(f"> {line}" if line else ">" for line in body.splitlines())
        return f"{self._protector.token(quoted)}\n\n" if quoted else ""

    def _render_table(self, node: Tag) -> str:
        if _table_has_span(node):
            return self._preserve_structure(node, "UNSUPPORTED_TABLE_SPAN")

        rows = _table_rows(node)
        if not rows:
            return self._preserve_structure(node, "UNSUPPORTED_TABLE_SHAPE")
        rendered_rows: list[list[str]] = []
        width: int | None = None
        for row in rows:
            cells = [
                child
                for child in row.children
                if isinstance(child, Tag) and _local_name(child.name) in {"td", "th"}
            ]
            if not cells:
                return self._preserve_structure(node, "UNSUPPORTED_TABLE_SHAPE")
            if width is None:
                width = len(cells)
            elif width != len(cells):
                return self._preserve_structure(node, "UNSUPPORTED_TABLE_SHAPE")

            rendered_cells: list[str] = []
            for cell in cells:
                value = self._protector.restore(
                    _cleanup_markdown(self._render_children(cell))
                ).strip()
                if "\n" in value:
                    return self._preserve_structure(node, "UNSUPPORTED_TABLE_CELL_BLOCK")
                rendered_cells.append(value)
            rendered_rows.append(rendered_cells)

        rendered = "\n".join(f"| {' | '.join(row)} |" for row in rendered_rows)
        return f"{self._protector.token(rendered)}\n\n"

    def _preserve_structure(self, node: Tag, reason: str) -> str:
        source_markup = self._protector.restore(str(node))
        self.unresolved_structure.append(
            {
                "code": "STRUCTURE_UNRESOLVED",
                "reason": reason,
                "source_markup": source_markup,
            }
        )
        return self._protector.token(source_markup)


def normalize_html(text: str) -> HtmlNormalizationResult:
    """Normalize deterministic formatting while preserving source semantics and evidence."""
    normalized_newlines = _normalize_newlines(text)
    protector = _SourceProtector(normalized_newlines)
    protected = protector.protect(normalized_newlines)
    soup = BeautifulSoup(protected, "html.parser")
    has_markup = soup.find(True) is not None

    transformations: list[str] = []
    if normalized_newlines != text:
        transformations.append("NEWLINES_NORMALIZED")

    if not has_markup:
        decoded = html_lib.unescape(protected)
        if decoded != protected:
            transformations.append("ENTITIES_DECODED")
        if protector.math_converted:
            transformations.append("MATHML_TO_LATEX")
        return HtmlNormalizationResult(
            text=protector.restore(_normalize_plain_text(decoded)),
            images=_stable_unique(protector.images),
            transformations=tuple(transformations),
            unresolved_math=tuple(protector.unresolved_math),
        )

    renderer = _Renderer(protector)
    rendered = "".join(renderer.render(child) for child in soup.contents)
    cleaned = _cleanup_markdown(rendered)
    cleaned = protector.restore(cleaned)
    transformations.append("HTML_TO_MARKDOWN")
    if renderer.math_converted:
        transformations.append("MATHML_TO_LATEX")
    if renderer.unresolved_structure:
        transformations.append("STRUCTURE_PRESERVED")

    return HtmlNormalizationResult(
        text=cleaned,
        images=_stable_unique(renderer.images),
        transformations=tuple(transformations),
        unresolved_math=tuple(renderer.unresolved_math),
        unresolved_structure=tuple(renderer.unresolved_structure),
    )


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_plain_text(text: str) -> str:
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _cleanup_markdown(text: str) -> str:
    text = _normalize_newlines(text)
    output: list[str] = []
    previous_blank = False
    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        if not line.strip():
            if output and not previous_blank:
                output.append("")
            previous_blank = True
            continue
        output.append(line)
        previous_blank = False
    while output and not output[-1].strip():
        output.pop()
    return "\n".join(output)


def _attribute(tag: Tag, name: str) -> str:
    value = tag.attrs.get(name)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return "" if value is None else str(value)


def _local_name(name: str) -> str:
    return name.rsplit(":", maxsplit=1)[-1].lower()


def _namespace_context(text: str) -> dict[str, str]:
    return {match.group("prefix"): match.group("uri") for match in _NAMESPACE_RE.finditer(text)}


def _is_display_math(source_mathml: str) -> bool:
    start_tag = source_mathml.split(">", maxsplit=1)[0]
    match = re.search(r"\bdisplay\s*=\s*['\"](?P<value>[^'\"]+)['\"]", start_tag)
    return match is not None and match.group("value").lower() == "block"


def _markdown_target_url(target: str) -> str:
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split(maxsplit=1)[0] if target else ""


def _ordered_list_settings(node: Tag) -> tuple[int, str] | None:
    marker_type = _attribute(node, "type") or "1"
    if marker_type not in {"1", "A", "a", "I", "i"}:
        return None
    raw_start = _attribute(node, "start")
    if not raw_start:
        return 1, marker_type
    try:
        return int(raw_start), marker_type
    except ValueError:
        return None


def _optional_int_attribute(node: Tag, name: str) -> int | bool | None:
    raw = _attribute(node, name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return False


def _ordered_marker(value: int, marker_type: str) -> str | None:
    if marker_type == "1":
        return f"{value}."
    if value <= 0:
        return None
    if marker_type in {"A", "a"}:
        alpha = _alpha_number(value)
        if alpha is None:
            return None
        return f"{alpha.upper() if marker_type == 'A' else alpha}."
    roman = _roman_number(value)
    if roman is None:
        return None
    return f"{roman if marker_type == 'I' else roman.lower()}."


def _alpha_number(value: int) -> str | None:
    if value <= 0:
        return None
    characters: list[str] = []
    current = value
    while current:
        current, remainder = divmod(current - 1, 26)
        characters.append(chr(ord("a") + remainder))
    return "".join(reversed(characters))


def _roman_number(value: int) -> str | None:
    if value <= 0 or value > 3999:
        return None
    mapping = (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    )
    parts: list[str] = []
    remaining = value
    for number, numeral in mapping:
        while remaining >= number:
            parts.append(numeral)
            remaining -= number
    return "".join(parts)


def _table_rows(node: Tag) -> list[Tag]:
    rows: list[Tag] = []
    for child in node.children:
        if not isinstance(child, Tag):
            continue
        name = _local_name(child.name)
        if name == "tr":
            rows.append(child)
        elif name in {"thead", "tbody", "tfoot"}:
            rows.extend(
                row
                for row in child.children
                if isinstance(row, Tag) and _local_name(row.name) == "tr"
            )
        else:
            return []
    return rows


def _table_has_span(node: Tag) -> bool:
    return any(
        isinstance(element, Tag)
        and _local_name(element.name) in {"td", "th"}
        and ("rowspan" in element.attrs or "colspan" in element.attrs)
        for element in node.descendants
    )


def _stable_unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)
