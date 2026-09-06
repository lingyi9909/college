"""Deterministic MathML-to-LaTeX conversion with fail-closed unresolved evidence."""

from __future__ import annotations

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


def convert_mathml(source_mathml: str) -> MathConversion:
    """Convert supported MathML to LaTeX or fail closed without guessing."""
    try:
        root = ET.fromstring(source_mathml)
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
    children = list(element)

    if tag in {"mi", "mn", "mo", "mtext"}:
        if children:
            raise _UnsupportedMathML(f"UNSUPPORTED_MATHML_STRUCTURE:{tag}")
        return (element.text or "").strip()

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


def _render_fenced(element: ET.Element, children: list[ET.Element]) -> str:
    opening = element.attrib.get("open", "(")
    closing = element.attrib.get("close", ")")
    separators = element.attrib.get("separators", ",") or ","
    rendered: list[str] = []
    for index, child in enumerate(children):
        if index:
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


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]
