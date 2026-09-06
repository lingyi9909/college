"""Deterministic MathML-to-LaTeX conversion with fail-closed unresolved evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from xml.etree import ElementTree as ET


class MathStatus(StrEnum):
    """Outcome of deterministic MathML normalization."""

    CONVERTED = "CONVERTED"
    MATH_UNRESOLVED = "MATH_UNRESOLVED"


@dataclass(frozen=True)
class MathConversion:
    """Deterministic conversion result retaining the source MathML evidence."""

    status: MathStatus
    latex: str | None
    source_mathml: str
    reason: str | None


class _UnsupportedMathML(ValueError):
    pass


_ALLOWED_ATTRIBUTES: dict[str, frozenset[str]] = {
    "math": frozenset({"display"}),
    "mfenced": frozenset({"open", "close", "separators"}),
}
_SUPPORTED_TAGS = frozenset(
    {
        "math",
        "mrow",
        "mstyle",
        "semantics",
        "annotation",
        "annotation-xml",
        "mi",
        "mn",
        "mo",
        "mtext",
        "msup",
        "msub",
        "msubsup",
        "mfrac",
        "msqrt",
        "mroot",
        "mfenced",
    }
)
_LATEX_SPECIAL_ATOMS = frozenset("\\{}$&#%_^~")


def convert_mathml(
    source_mathml: str,
    *,
    namespace_context: Mapping[str, str] | None = None,
) -> MathConversion:
    """Convert proven-safe MathML to LaTeX or fail closed without guessing."""
    parse_source = _inject_namespace_context(source_mathml, namespace_context or {})
    try:
        root = ET.fromstring(parse_source)
    except ET.ParseError:
        return MathConversion(
            status=MathStatus.MATH_UNRESOLVED,
            latex=None,
            source_mathml=source_mathml,
            reason="MATHML_PARSE_ERROR",
        )

    if _local_name(root.tag) != "math":
        return MathConversion(
            status=MathStatus.MATH_UNRESOLVED,
            latex=None,
            source_mathml=source_mathml,
            reason="MATHML_ROOT_NOT_MATH",
        )

    try:
        latex = _render(root)
    except _UnsupportedMathML as exc:
        return MathConversion(
            status=MathStatus.MATH_UNRESOLVED,
            latex=None,
            source_mathml=source_mathml,
            reason=str(exc),
        )

    return MathConversion(
        status=MathStatus.CONVERTED,
        latex=latex,
        source_mathml=source_mathml,
        reason=None,
    )


def _render(element: ET.Element) -> str:
    tag = _local_name(element.tag)
    if tag not in _SUPPORTED_TAGS:
        raise _UnsupportedMathML(f"UNSUPPORTED_MATHML_TAG:{tag}")
    _validate_attributes(element, tag)
    children = list(element)

    if tag in {"mi", "mn", "mo"}:
        if children:
            raise _UnsupportedMathML(f"UNSUPPORTED_MATHML_STRUCTURE:{tag}")
        value = (element.text or "").strip()
        if any(character in _LATEX_SPECIAL_ATOMS for character in value):
            raise _UnsupportedMathML(f"UNSUPPORTED_MATHML_TEXT:{tag}")
        return value

    if tag == "mtext":
        if children:
            raise _UnsupportedMathML("UNSUPPORTED_MATHML_STRUCTURE:mtext")
        return f"\\text{{{_escape_latex_text(element.text or '')}}}"

    if _has_semantic_mixed_text(element):
        raise _UnsupportedMathML(f"UNSUPPORTED_MATHML_MIXED_TEXT:{tag}")

    if tag in {"math", "mrow", "mstyle"}:
        return "".join(_render(child) for child in children)

    if tag == "semantics":
        expression_children = [
            child
            for child in children
            if _local_name(child.tag) not in {"annotation", "annotation-xml"}
        ]
        if len(expression_children) != 1:
            raise _UnsupportedMathML("UNSUPPORTED_MATHML_STRUCTURE:semantics")
        return _render(expression_children[0])

    if tag in {"annotation", "annotation-xml"}:
        raise _UnsupportedMathML(f"UNSUPPORTED_MATHML_TAG:{tag}")

    if tag == "msup":
        _require_children(tag, children, 2)
        return f"{_render(children[0])}^{{{_render(children[1])}}}"

    if tag == "msub":
        _require_children(tag, children, 2)
        return f"{_render(children[0])}_{{{_render(children[1])}}}"

    if tag == "msubsup":
        _require_children(tag, children, 3)
        return (
            f"{_render(children[0])}_{{{_render(children[1])}}}"
            f"^{{{_render(children[2])}}}"
        )

    if tag == "mfrac":
        _require_children(tag, children, 2)
        return f"\\frac{{{_render(children[0])}}}{{{_render(children[1])}}}"

    if tag == "msqrt":
        if not children:
            raise _UnsupportedMathML("UNSUPPORTED_MATHML_STRUCTURE:msqrt")
        return f"\\sqrt{{{''.join(_render(child) for child in children)}}}"

    if tag == "mroot":
        _require_children(tag, children, 2)
        return f"\\sqrt[{_render(children[1])}]{{{_render(children[0])}}}"

    if tag == "mfenced":
        return _render_fenced(element, children)

    raise _UnsupportedMathML(f"UNSUPPORTED_MATHML_TAG:{tag}")


def _validate_attributes(element: ET.Element, tag: str) -> None:
    allowed = _ALLOWED_ATTRIBUTES.get(tag, frozenset())
    for raw_name in element.attrib:
        name = _local_name(raw_name)
        if name not in allowed:
            raise _UnsupportedMathML(f"UNSUPPORTED_MATHML_ATTRIBUTE:{tag}:{name}")


def _render_fenced(element: ET.Element, children: list[ET.Element]) -> str:
    opening = element.attrib.get("open", "(")
    closing = element.attrib.get("close", ")")
    separators = element.attrib["separators"] if "separators" in element.attrib else ","
    rendered: list[str] = []
    for index, child in enumerate(children):
        if index and separators:
            separator_index = min(index - 1, len(separators) - 1)
            rendered.append(separators[separator_index])
        rendered.append(_render(child))
    return f"{opening}{''.join(rendered)}{closing}"


def _require_children(tag: str, children: list[ET.Element], expected: int) -> None:
    if len(children) != expected:
        raise _UnsupportedMathML(f"UNSUPPORTED_MATHML_STRUCTURE:{tag}")


def _has_semantic_mixed_text(element: ET.Element) -> bool:
    if (element.text or "").strip():
        return True
    return any((child.tail or "").strip() for child in element)


def _escape_latex_text(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "%": r"\%",
        "_": r"\_",
        "^": r"\^{}",
        "~": r"\~{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _inject_namespace_context(source: str, context: Mapping[str, str]) -> str:
    if not context:
        return source
    prefixes = set(re.findall(r"</?([A-Za-z_][\w.-]*):[A-Za-z_]", source))
    declarations: list[str] = []
    for prefix in sorted(prefixes):
        if re.search(rf"\bxmlns:{re.escape(prefix)}\s*=", source):
            continue
        uri = context.get(prefix)
        if uri is None:
            continue
        escaped_uri = uri.replace("&", "&amp;").replace('"', "&quot;")
        declarations.append(f' xmlns:{prefix}="{escaped_uri}"')
    if not declarations:
        return source
    first_close = source.find(">")
    if first_close < 0:
        return source
    return f"{source[:first_close]}{''.join(declarations)}{source[first_close:]}"


def _local_name(tag: str) -> str:
    local = tag.rsplit("}", maxsplit=1)[-1]
    return local.rsplit(":", maxsplit=1)[-1]
