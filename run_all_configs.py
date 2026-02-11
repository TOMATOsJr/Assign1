"""Train and etestuate all tokenizer x smoothing configurations on tokenized files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from language_models import (
    KneserNeySmoothing,
    MLESmoothing,
    NGramLanguageModel,
    WittenBellSmoothing,
    read_tokenized_lines,
)

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - required dependency
    raise RuntimeError("tqdm is required for progress bars. Please install tqdm.") from exc


def get_smoothing(name: str):
    if name == "mle":
        return MLESmoothing()
    if name == "kneser_ney":
        return KneserNeySmoothing()
    if name == "witten_bell":
        return WittenBellSmoothing()
    raise ValueError(f"Unsupported smoothing: {name}")


def build_path(base: str, tokenizer: str, split: str) -> Path:
    return Path(f"{base}_{tokenizer}_{split}")


def train_and_etest(
    train_path: Path,
    test_path: Path,
    n: int,
    smoothing_name: str,
    add_bos_eos: bool,
) -> float:
    print(f"Loading train: {train_path}")
    train_sequences = read_tokenized_lines(str(train_path))
    print(f"Loading test: {test_path}")
    test_sequences = read_tokenized_lines(str(test_path))

    model = NGramLanguageModel(
        n=n,
        smoothing=get_smoothing(smoothing_name),
        add_bos_eos=add_bos_eos,
    )
    model.fit(tqdm(train_sequences, desc="Training", unit="seq", total=len(train_sequences)))
    return model.perplexity(tqdm(test_sequences, desc="Perplexity", unit="seq", total=len(test_sequences)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and etestuate all tokenizer x smoothing configs on tokenized files.",
    )
    parser.add_argument(
        "--base",
        default="cc100_en_tokens",
        help="Base path prefix (e.g., cc100_en_tokens).",
    )
    parser.add_argument("--n", type=int, default=4, help="n-gram order.")
    parser.add_argument("--add-bos-eos", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tokenizers = ["regex", "bpe"]
    smoothings = ["kneser_ney", "witten_bell"]

    results: List[Tuple[str, str, str]] = []

    for tokenizer in tokenizers:
        train_path = build_path(args.base, tokenizer, "train")
        test_path = build_path(args.base, tokenizer, "test")

        if not train_path.exists() or not test_path.exists():
            results.append((tokenizer, "-", "missing train/test"))
            continue

        for smoothing in smoothings:
            print(f"\nRunning config: tokenizer={tokenizer}, smoothing={smoothing}")
            try:
                ppl = train_and_etest(
                    train_path=train_path,
                    test_path=test_path,
                    n=args.n,
                    smoothing_name=smoothing,
                    add_bos_eos=args.add_bos_eos,
                )
            except Exception as exc:  # noqa: BLE001
                results.append((tokenizer, smoothing, f"error: {exc}"))
                print(f"Failed: tokenizer={tokenizer}, smoothing={smoothing} ({exc})")
                continue
            results.append((tokenizer, smoothing, f"{ppl:.6f}"))
            print(f"Done: tokenizer={tokenizer}, smoothing={smoothing} -> ppl={ppl:.6f}")

    print("tokenizer\tsmoothing\tperplexity")
    for tokenizer, smoothing, metric in results:
        print(f"{tokenizer}\t{smoothing}\t{metric}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
