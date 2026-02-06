"""Tokenizer pipeline script.

Runs a tokenizer over input text or a JSONL corpus and writes tokenized output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from tokenizers import BaseTokenizer, get_tokenizer


def _iter_lines_from_file(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield line.rstrip("\n")


def _iter_texts_jsonl(path: Path, field: str) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if field in obj and isinstance(obj[field], str):
                yield obj[field]


def _tokenize_lines(tokenizer: BaseTokenizer, lines: Iterable[str]) -> Iterable[List[str]]:
    for line in lines:
        if line.strip() == "":
            yield []
            continue
        yield tokenizer.tokenize(line)


def run_pipeline(
    tokenizer_kind: str,
    input_path: Path,
    output_path: Path,
    jsonl_field: str | None = None,
) -> None:
    tokenizer = get_tokenizer(tokenizer_kind)

    if jsonl_field:
        lines = _iter_texts_jsonl(input_path, jsonl_field)
    else:
        lines = _iter_lines_from_file(input_path)

    tokenized = _tokenize_lines(tokenizer, lines)

    with output_path.open("w", encoding="utf-8") as handle:
        for tokens in tokenized:
            handle.write(" ".join(tokens))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tokenizer pipeline")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer kind: whitespace|regex|bpe")
    parser.add_argument("--input", required=True, help="Input file path (txt or jsonl)")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument(
        "--jsonl-field",
        default=None,
        help="If input is JSONL, name of the field to tokenize",
    )

    args = parser.parse_args()

    run_pipeline(
        tokenizer_kind=args.tokenizer,
        input_path=Path(args.input),
        output_path=Path(args.output),
        jsonl_field=args.jsonl_field,
    )


if __name__ == "__main__":
    main()
