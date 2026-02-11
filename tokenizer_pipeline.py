"""Tokenizer pipeline script.

Runs a tokenizer over input text or a JSONL corpus and writes tokenized output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List

from tokenizers import BaseTokenizer, BPETokenizer, get_tokenizer

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def _count_lines(path: Path, jsonl_field: str | None = None) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if jsonl_field:
                line = line.strip()
                if not line:
                    continue
            count += 1
    return count


def _iter_with_progress(
    lines: Iterable[str],
    label: str,
    progress_every: int,
    total: int | None = None,
) -> Iterator[str]:
    if tqdm is None:
        count = 0
        for line in lines:
            count += 1
            if progress_every and count % progress_every == 0:
                print(f"{label}: {count} lines...")
            yield line
        if progress_every:
            print(f"{label}: {count} lines total.")
        return

    for line in tqdm(lines, desc=label, unit="lines", total=total):
        yield line


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
    val_input_path: Path | None = None,
    val_output_path: Path | None = None,
    val_jsonl_field: str | None = None,
    train_input_path: Path | None = None,
    train_jsonl_field: str | None = None,
    vocab_size: int | None = None,
    progress_every: int = 100000,
) -> None:
    if tokenizer_kind.lower() == "bpe":
        if vocab_size is None:
            raise ValueError("--vocab-size is required for bpe")
        if train_input_path is None:
            raise ValueError("--train-input is required for bpe")
        train_field = train_jsonl_field if train_jsonl_field is not None else jsonl_field
        tokenizer = BPETokenizer(vocab_size=vocab_size)

        train_total = _count_lines(train_input_path, train_field)
        if train_field:
            train_lines = _iter_texts_jsonl(train_input_path, train_field)
        else:
            train_lines = _iter_lines_from_file(train_input_path)
        train_lines = _iter_with_progress(train_lines, "Training", progress_every, total=train_total)
        tokenizer.train(train_lines)
        print("\nTraining complete. Starting tokenization...\n")
    else:
        tokenizer = get_tokenizer(tokenizer_kind)

    def _tokenize_to_file(label: str, in_path: Path, out_path: Path, field: str | None) -> None:
        total = _count_lines(in_path, field)
        if field:
            lines = _iter_texts_jsonl(in_path, field)
        else:
            lines = _iter_lines_from_file(in_path)

        lines_list = list(_iter_with_progress(lines, f"Reading {label}", progress_every, total=total))

        if tqdm is not None:
            tokenized = [tokenizer.tokenize(line) if line.strip() else []
                         for line in tqdm(lines_list, desc=f"Tokenizing {label}", unit="lines", total=len(lines_list))]
        else:
            tokenized = []
            for i, line in enumerate(lines_list):
                if progress_every and (i + 1) % progress_every == 0:
                    print(f"Tokenizing {label}: {i + 1} lines...")
                tokenized.append(tokenizer.tokenize(line) if line.strip() else [])
            if progress_every:
                print(f"Tokenizing {label}: {len(lines_list)} lines total.")

        print(f"\nWriting {label} output...")
        with out_path.open("w", encoding="utf-8") as handle:
            for tokens in tokenized:
                handle.write(" ".join(tokens))
                handle.write("\n")
        print(f"Done! Output written to {out_path}")

    _tokenize_to_file("input", input_path, output_path, jsonl_field)

    if val_input_path and val_output_path:
        val_field = val_jsonl_field if val_jsonl_field is not None else jsonl_field
        _tokenize_to_file("val", val_input_path, val_output_path, val_field)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tokenizer pipeline")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer kind: whitespace|regex|bpe")
    parser.add_argument("--input", required=True, help="Input file path (txt or jsonl)")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--val-input", default=None, help="Validation file path")
    parser.add_argument("--val-output", default=None, help="Validation output file path")
    parser.add_argument("--train-input", default=None, help="Training file path for bpe")
    parser.add_argument("--vocab-size", type=int, default=None, help="Vocab size for bpe")
    parser.add_argument(
        "--jsonl-field",
        default=None,
        help="If input is JSONL, name of the field to tokenize",
    )
    parser.add_argument(
        "--json-field",
        default=None,
        help="Alias for --jsonl-field",
    )
    parser.add_argument(
        "--train-jsonl-field",
        default=None,
        help="If train input is JSONL, name of the field to train on",
    )
    parser.add_argument(
        "--val-jsonl-field",
        default=None,
        help="If val input is JSONL, name of the field to tokenize",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N lines (0 to disable)",
    )

    args = parser.parse_args()

    jsonl_field = args.jsonl_field if args.jsonl_field is not None else args.json_field

    run_pipeline(
        tokenizer_kind=args.tokenizer,
        input_path=Path(args.input),
        output_path=Path(args.output),
        jsonl_field=jsonl_field,
        val_input_path=Path(args.val_input) if args.val_input else None,
        val_output_path=Path(args.val_output) if args.val_output else None,
        val_jsonl_field=args.val_jsonl_field,
        train_input_path=Path(args.train_input) if args.train_input else None,
        train_jsonl_field=args.train_jsonl_field,
        vocab_size=args.vocab_size,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
