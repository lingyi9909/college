from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

REVISION = "1eafca778505d699033c92554b5984621f11380d"
URL = (
    "https://huggingface.co/datasets/pranav-gupta/openstax-qa/resolve/"
    + REVISION
    + "/openstax_processed_data.json?download=true"
)
EXPECTED_SHA = "4df9ff9fcefc6c41d527618f80935f5ce4198011b438f70e811dff26051a9fd2"
PATH = Path("/tmp/openstax.json")


def rows(node: object):
    if isinstance(node, dict):
        if {"id", "problem", "solution"}.issubset(node):
            yield node
            return
        for value in node.values():
            yield from rows(value)
    elif isinstance(node, list):
        for value in node:
            yield from rows(value)


def main() -> None:
    urllib.request.urlretrieve(URL, PATH)
    data = PATH.read_bytes()
    assert hashlib.sha256(data).hexdigest() == EXPECTED_SHA
    parsed = json.loads(data)
    by_id: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_book_id: dict[tuple[str, str], int] = defaultdict(int)
    by_book_chapter_id: dict[tuple[str, str, str], int] = defaultdict(int)
    count = 0
    for row in rows(parsed):
        count += 1
        source_id = str(row["id"])
        book = str(row.get("book", ""))
        chapter = str(row.get("chapter_number", ""))
        by_id[source_id].append(row)
        by_book_id[(book, source_id)] += 1
        by_book_chapter_id[(book, chapter, source_id)] += 1
    duplicates = {key: value for key, value in by_id.items() if len(value) > 1}
    duplicate_book_id = {key: value for key, value in by_book_id.items() if value > 1}
    duplicate_book_chapter_id = {
        key: value for key, value in by_book_chapter_id.items() if value > 1
    }
    print("ROW_COUNT", count)
    print("UNIQUE_ID_COUNT", len(by_id))
    print("DUPLICATE_ID_COUNT", len(duplicates))
    print("UNIQUE_BOOK_ID_COUNT", len(by_book_id))
    print("DUPLICATE_BOOK_ID_COUNT", len(duplicate_book_id))
    print("UNIQUE_BOOK_CHAPTER_ID_COUNT", len(by_book_chapter_id))
    print("DUPLICATE_BOOK_CHAPTER_ID_COUNT", len(duplicate_book_chapter_id))
    for key in sorted(duplicates)[:10]:
        print("DUPLICATE", key, len(duplicates[key]))
        for row in duplicates[key]:
            payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            print(json.dumps({
                "book": row.get("book"),
                "chapter_number": row.get("chapter_number"),
                "language": row.get("language"),
                "row_sha256": hashlib.sha256(payload).hexdigest(),
                "problem_sha256": hashlib.sha256(str(row.get("problem", "")).encode()).hexdigest(),
                "solution_sha256": hashlib.sha256(str(row.get("solution", "")).encode()).hexdigest(),
            }, sort_keys=True))


if __name__ == "__main__":
    main()
