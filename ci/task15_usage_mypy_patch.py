from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one patch context, got {count}")
    return text.replace(old, new, 1)


provider = Path("src/college_builder/providers/openai_compatible.py")
text = provider.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''        valid = (\n            isinstance(total_tokens, int)\n            and not isinstance(total_tokens, bool)\n            and total_tokens >= 0\n        )\n        with self._usage_lock:\n            if valid:\n                self._total_tokens += total_tokens\n            else:\n                self._usage_complete = False\n''',
    '''        with self._usage_lock:\n            if (\n                isinstance(total_tokens, int)\n                and not isinstance(total_tokens, bool)\n                and total_tokens >= 0\n            ):\n                self._total_tokens += total_tokens\n            else:\n                self._usage_complete = False\n''',
    "provider token narrowing",
)
provider.write_text(text, encoding="utf-8", newline="\n")

runner = Path("src/college_builder/pipeline/runner.py")
text = runner.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''def _optional_usage_delta(before: int | float | None, after: int | float | None):\n    if before is None or after is None:\n        return None\n    return max(0, after - before)\n\n\ndef _optional_usage_sum(left: int | float | None, right: int | float | None):\n    if left is None or right is None:\n        return None\n    return left + right\n''',
    '''def _optional_int_delta(before: int | None, after: int | None) -> int | None:\n    if before is None or after is None:\n        return None\n    return max(0, after - before)\n\n\ndef _optional_float_delta(before: float | None, after: float | None) -> float | None:\n    if before is None or after is None:\n        return None\n    return max(0.0, after - before)\n\n\ndef _optional_int_sum(left: int | None, right: int | None) -> int | None:\n    if left is None or right is None:\n        return None\n    return left + right\n\n\ndef _optional_float_sum(left: float | None, right: float | None) -> float | None:\n    if left is None or right is None:\n        return None\n    return left + right\n''',
    "typed usage helpers",
)
text = replace_once(
    text,
    '''        tokens=_optional_usage_delta(before.tokens, after.tokens),\n        estimated_cost_usd=_optional_usage_delta(\n            before.estimated_cost_usd, after.estimated_cost_usd\n        ),\n''',
    '''        tokens=_optional_int_delta(before.tokens, after.tokens),\n        estimated_cost_usd=_optional_float_delta(\n            before.estimated_cost_usd, after.estimated_cost_usd\n        ),\n''',
    "typed usage delta calls",
)
text = replace_once(
    text,
    '''        tokens=_optional_usage_sum(left.tokens, right.tokens),\n        estimated_cost_usd=_optional_usage_sum(\n            left.estimated_cost_usd, right.estimated_cost_usd\n        ),\n''',
    '''        tokens=_optional_int_sum(left.tokens, right.tokens),\n        estimated_cost_usd=_optional_float_sum(\n            left.estimated_cost_usd, right.estimated_cost_usd\n        ),\n''',
    "typed usage sum calls",
)
runner.write_text(text, encoding="utf-8", newline="\n")
