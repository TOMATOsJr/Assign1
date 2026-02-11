"""Train a tokenizer + n-gram LM and run interactive inference.

Example:
  python interactive_autocomplete.py \
    --train cc100_en_final_train.jsonl --train-format jsonl --jsonl-field text \
    --tokenizer bpe --bpe-vocab-size 4000 --smoothing kneser_ney --n 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, List

from language_models import KneserNeySmoothing, MLESmoothing, NGramLanguageModel, WittenBellSmoothing
from tokenizers import BPETokenizer, get_tokenizer

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - required dependency
    raise RuntimeError("tqdm is required for progress bars. Please install tqdm.") from exc


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


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield line.rstrip("\n")


def _iter_jsonl(path: Path, field: str) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(field, "")
            if not text:
                continue
            yield text


def _iter_tokenized(path: Path) -> Iterator[List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield line.split()


def _format_generated(tokens: List[str], eos_token: str) -> List[str]:
    cleaned: List[str] = []
    for token in tokens:
        if token == eos_token:
            break
        if token in ("—", "–"):
            token = "-"
        cleaned.append(token)
    return cleaned


def get_smoothing(name: str):
    if name == "mle":
        return MLESmoothing()
    if name == "kneser_ney":
        return KneserNeySmoothing()
    if name == "witten_bell":
        return WittenBellSmoothing()
    raise ValueError(f"Unsupported smoothing: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tokenizer + n-gram LM and run interactive inference.",
    )
    parser.add_argument("--train", required=True, help="Training data path.")
    parser.add_argument("--train-format", choices=["text", "jsonl", "tokenized"], default="jsonl")
    parser.add_argument("--jsonl-field", default="text", help="JSONL field for text.")
    parser.add_argument("--tokenizer", choices=["whitespace", "regex", "bpe"], default="whitespace")
    parser.add_argument("--bpe-vocab-size", type=int, default=None, help="Vocab size for BPE.")
    parser.add_argument("--n", type=int, default=4, help="n-gram order.")
    parser.add_argument(
        "--smoothing",
        choices=["mle", "kneser_ney", "witten_bell"],
        default="mle",
    )
    parser.add_argument(
        "--add-bos-eos",
        action="store_true",
        default=True,
        help="Add BOS/EOS tokens (enabled by default).",
    )
    parser.add_argument(
        "--no-add-bos-eos",
        action="store_false",
        dest="add_bos_eos",
        help="Disable adding BOS/EOS tokens.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k suggestions to show.")
    parser.add_argument(
        "--generate",
        action="store_true",
        default=True,
        help="Generate a continuation (enabled by default).",
    )
    parser.add_argument(
        "--no-generate",
        action="store_false",
        dest="generate",
        help="Disable generation and show top-k suggestions.",
    )
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate.")
    return parser.parse_args()


def _load_texts(args: argparse.Namespace) -> List[str]:
    path = Path(args.train)
    if args.train_format == "tokenized":
        raise ValueError("tokenized format is not supported for tokenizer training")

    if args.train_format == "jsonl":
        total = _count_lines(path, jsonl_field=args.jsonl_field)
        iterator = _iter_jsonl(path, args.jsonl_field)
    else:
        total = _count_lines(path)
        iterator = _iter_lines(path)

    return list(tqdm(iterator, desc="Loading text", unit="lines", total=total))


def _load_tokenized(args: argparse.Namespace) -> List[List[str]]:
    path = Path(args.train)
    if args.train_format != "tokenized":
        raise ValueError("tokenized format is required to load tokens directly")
    total = _count_lines(path)
    iterator = _iter_tokenized(path)
    return list(tqdm(iterator, desc="Loading tokens", unit="lines", total=total))


def main() -> int:
    args = parse_args()

    if args.train_format == "tokenized":
        sequences = _load_tokenized(args)
        tokenizer = None
        print("Loaded tokenized training data.")
    else:
        texts = _load_texts(args)
        if args.tokenizer == "bpe":
            if args.bpe_vocab_size is None:
                raise SystemExit("--bpe-vocab-size is required for BPE")
            tokenizer = BPETokenizer(vocab_size=args.bpe_vocab_size)
            tokenizer.train(tqdm(texts, desc="Training BPE", unit="lines"))
        else:
            tokenizer = get_tokenizer(args.tokenizer)
        print("Tokenizer ready. Tokenizing training data...")
        sequences = [tokenizer.tokenize(text) for text in tqdm(texts, desc="Tokenizing", unit="lines")]

    smoothing = get_smoothing(args.smoothing)
    model = NGramLanguageModel(
        n=args.n,
        smoothing=smoothing,
        add_bos_eos=args.add_bos_eos,
    )
    model.fit(tqdm(sequences, desc="Training LM", unit="seq"))
    print("Model trained. Enter prompts to infer.")

    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            break
        prompt = line.strip()
        if not prompt:
            continue

        if tokenizer is None:
            prompt_tokens = prompt.split()
        else:
            prompt_tokens = tokenizer.tokenize(prompt)

        if args.generate:
            generated = model.generate(prompt_tokens, max_tokens=args.max_tokens)
            formatted = _format_generated(generated, model.eos_token)
            print(" ".join(formatted))
        else:
            suggestions = model.predict_top_k(prompt_tokens, k=args.top_k)
            for token, prob in suggestions:
                print(f"{token}\t{prob:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
