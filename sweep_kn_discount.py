"""Sweep Kneser-Ney discount values and report validation perplexity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List

from language_models import KneserNeySmoothing, MLESmoothing, NGramLanguageModel

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


def _iter_tokenized(path: Path) -> Iterator[List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield line.split()


def _iter_jsonl(path: Path, field: str) -> Iterator[List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(field, "")
            if not text:
                continue
            yield text.split()


def load_sequences(path: str, fmt: str, field: str, label: str) -> List[List[str]]:
    data_path = Path(path)
    if fmt == "tokenized":
        total = _count_lines(data_path)
        iterator = _iter_tokenized(data_path)
    elif fmt == "jsonl":
        total = _count_lines(data_path, jsonl_field=field)
        iterator = _iter_jsonl(data_path, field=field)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return list(tqdm(iterator, desc=label, unit="lines", total=total))


def parse_discounts(value: str) -> List[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("--discounts must contain at least one value")
    discounts = [float(item) for item in items]
    for discount in discounts:
        if not (0.0 < discount < 1.0):
            raise ValueError(f"discount must be between 0 and 1: {discount}")
    return discounts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Kneser-Ney discount values and report perplexity.",
    )
    parser.add_argument("--train", required=True, help="Training data path.")
    parser.add_argument("--val", required=True, help="Validation data path.")
    parser.add_argument("--train-format", choices=["tokenized", "jsonl"], default="tokenized")
    parser.add_argument("--val-format", choices=["tokenized", "jsonl"], default="tokenized")
    parser.add_argument("--jsonl-field", default="text", help="JSONL field for text.")
    parser.add_argument("--n", type=int, default=4, help="n-gram order.")
    parser.add_argument(
        "--discounts",
        default="0.5,0.6,0.7,0.75,0.8,0.9",
        help="Comma-separated list of discount values.",
    )
    parser.add_argument("--add-bos-eos", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    discounts = parse_discounts(args.discounts)

    train_sequences = load_sequences(
        args.train,
        args.train_format,
        args.jsonl_field,
        label="Loading train",
    )
    val_sequences = load_sequences(
        args.val,
        args.val_format,
        args.jsonl_field,
        label="Loading val",
    )

    model = NGramLanguageModel(
        n=args.n,
        smoothing=MLESmoothing(),
        add_bos_eos=args.add_bos_eos,
    )
    model.fit(tqdm(train_sequences, desc="Training counts", unit="seq"))

    oov_count = 0
    token_count = 0
    for sequence in tqdm(val_sequences, desc="OOV scan", unit="seq"):
        for token in sequence:
            token_count += 1
            if token not in model.vocab:
                oov_count += 1
    if token_count:
        oov_rate = (oov_count / token_count) * 100.0
    else:
        oov_rate = 0.0
    print(f"\nOOV tokens in val: {oov_count}/{token_count} ({oov_rate:.2f}%)")

    results: List[tuple[float, float]] = []
    for discount in discounts:
        smoothing = KneserNeySmoothing(discount=discount)
        smoothing.fit(model.counts, len(model.vocab))
        model.smoothing = smoothing
        ppl = model.perplexity(tqdm(val_sequences, desc=f"Perplexity D={discount}", unit="seq"))
        results.append((discount, ppl))

    print("\nDiscount\tPerplexity")
    for discount, ppl in results:
        print(f"{discount:.4f}\t{ppl:.6f}")

    best = min(results, key=lambda item: item[1])
    print(f"\nBest: D={best[0]:.4f} with perplexity {best[1]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
