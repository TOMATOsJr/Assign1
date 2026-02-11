"""Count unique characters in a text or JSONL file.

Usage examples:
  python count_charset.py --input cc100_en_cleaned_final.jsonl --jsonl-field text
  python count_charset.py --input cc100_en_tokens_whitespace
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield line.rstrip("\n")


def _iter_jsonl_field(path: Path, field: str) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            value = obj.get(field)
            if isinstance(value, str):
                yield value


def _count_chars(texts: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(text)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Count unique characters")
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument(
        "--jsonl-field",
        default=None,
        help="If input is JSONL, field name containing text",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output file to write all unique chars",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Show top N characters by frequency",
    )

    args = parser.parse_args()
    path = Path(args.input)

    if args.jsonl_field:
        texts = _iter_jsonl_field(path, args.jsonl_field)
    else:
        texts = _iter_lines(path)

    counts = _count_chars(texts)
    unique = len(counts)
    total = sum(counts.values())

    print(f"Total characters: {total}")
    print(f"Unique characters: {unique}")
    if args.out:
        out_path = Path(args.out)
        with out_path.open("w", encoding="utf-8") as handle:
            for ch in sorted(counts.keys()):
                handle.write(ch)
                handle.write("\n")
        print(f"Wrote unique characters to {out_path}")
    print("\nTop characters:")
    for ch, freq in counts.most_common(args.top):
        display = ch
        if ch == " ":
            display = "<SPACE>"
        elif ch == "\t":
            display = "<TAB>"
        elif ch == "\n":
            display = "<NEWLINE>"
        print(f"{display}\t{freq}")


if __name__ == "__main__":
    main()
