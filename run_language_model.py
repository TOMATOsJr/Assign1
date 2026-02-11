"""CLI for training and evaluating n-gram language models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, List

from language_models import (
    EOS_TOKEN,
    KneserNeySmoothing,
    MLESmoothing,
    NGramLanguageModel,
    WittenBellSmoothing,
)

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
        description="Train and evaluate an n-gram language model.",
    )
    parser.add_argument("--train", help="Training data path.")
    parser.add_argument("--train-format", choices=["tokenized", "jsonl"], default="tokenized")
    parser.add_argument("--val", help="Validation data path.")
    parser.add_argument("--val-format", choices=["tokenized", "jsonl"], default=None)
    parser.add_argument("--jsonl-field", default="text", help="JSONL field for text.")
    parser.add_argument("--n", type=int, default=4, help="n-gram order.")
    parser.add_argument(
        "--smoothing",
        choices=["mle", "kneser_ney", "witten_bell"],
        default="mle",
    )
    parser.add_argument("--add-bos-eos", action="store_true")
    parser.add_argument("--prompt", help="Prompt text for generation.")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--stop-token", default=EOS_TOKEN)
    parser.add_argument("--save-model", help="Path to save trained model JSON.")
    parser.add_argument("--load-model", help="Path to load a saved model JSON.")
    parser.add_argument("--autocomplete", action="store_true", help="Run interactive autocomplete.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k suggestions for autocomplete.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.load_model:
        model = NGramLanguageModel.load(args.load_model)
        print(f"Loaded model from {args.load_model}")
    else:
        if not args.train:
            raise SystemExit("--train is required unless --load-model is provided")
        train_sequences = load_sequences(
            args.train,
            args.train_format,
            args.jsonl_field,
            label="Loading train",
        )
        smoothing = get_smoothing(args.smoothing)
        model = NGramLanguageModel(
            n=args.n,
            smoothing=smoothing,
            add_bos_eos=args.add_bos_eos,
        )
        model.fit(tqdm(train_sequences, desc="Training", unit="seq"))
        if args.save_model:
            model.save(args.save_model)

    if args.val:
        val_format = args.val_format or args.train_format
        val_sequences = load_sequences(
            args.val,
            val_format,
            args.jsonl_field,
            label="Loading val",
        )
        try:
            ppl = model.perplexity(tqdm(val_sequences, desc="Perplexity", unit="seq"))
        except NotImplementedError as exc:
            print(f"Perplexity not available: {exc}", file=sys.stderr)
        else:
            print(f"Perplexity: {ppl:.6f}")

    if args.prompt:
        prompt_tokens = args.prompt.split()
        try:
            generated = model.generate(
                prompt_tokens,
                max_tokens=args.max_tokens,
                stop_token=args.stop_token,
            )
        except NotImplementedError as exc:
            print(f"Generation not available: {exc}", file=sys.stderr)
        else:
            print(" ".join(generated))

    if args.autocomplete:
        print("Autocomplete mode. Enter a prefix and press Enter (Ctrl+D to exit).")
        while True:
            try:
                line = input("> ")
            except EOFError:
                print()
                break
            prefix = line.strip().split()
            if not prefix:
                continue
            suggestions = model.predict_top_k(prefix, k=args.top_k)
            for token, prob in suggestions:
                print(f"{token}\t{prob:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
