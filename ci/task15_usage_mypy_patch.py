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
    '''def _optional_usage_delta(before: int | float | None, after: int | float | None):\n''',
    '''def _optional_usage_delta(\n    before: int | float | None, after: int | float | None\n) -> int | float | None:\n''',
    "usage delta return type",
)
text = replace_once(
    text,
    '''def _optional_usage_sum(left: int | float | None, right: int | float | None):\n''',
    '''def _optional_usage_sum(\n    left: int | float | None, right: int | float | None\n) -> int | float | None:\n''',
    "usage sum return type",
)
runner.write_text(text, encoding="utf-8", newline="\n")
