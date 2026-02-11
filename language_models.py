"""Non-neural probabilistic 4-gram language models."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
import json
import pickle
import math
from typing import Iterable, List, Optional, Sequence, Tuple


BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


def read_tokenized_lines(path: str) -> List[List[str]]:
    """Read tokenized text where each line is a space-separated sequence."""
    sequences: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            sequences.append(line.split())
    return sequences


def read_jsonl_text(path: str, field: str = "text") -> List[List[str]]:
    """Read JSONL and split the selected field on whitespace."""
    sequences: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(field, "")
            if not text:
                continue
            sequences.append(text.split())
    return sequences


class LanguageModel:
    """Base interface for probabilistic language models."""

    def fit(self, sequences: Iterable[Sequence[str]]) -> None:
        raise NotImplementedError

    def score(self, sequence: Sequence[str]) -> float:
        raise NotImplementedError

    def predict_next(self, prefix: Sequence[str], return_prob: bool = False):
        raise NotImplementedError

    def generate(
        self,
        prefix: Sequence[str],
        max_tokens: int = 50,
        stop_token: str = EOS_TOKEN,
    ) -> List[str]:
        raise NotImplementedError

    def perplexity(self, sequences: Iterable[Sequence[str]]) -> float:
        raise NotImplementedError


@dataclass
class NGramCounts:
    ngram: Counter
    context: Counter
    ngram_by_order: dict[int, Counter] = field(default_factory=dict)
    context_by_order: dict[int, Counter] = field(default_factory=dict)


class SmoothingStrategy:
    """Base class for smoothing strategies used by n-gram models."""

    name = "base"

    def fit(self, counts: NGramCounts, vocab_size: int) -> None:
        self.counts = counts
        self.vocab_size = vocab_size

    def prob(self, context: Tuple[str, ...], token: str) -> float:
        raise NotImplementedError


class MLESmoothing(SmoothingStrategy):
    name = "mle"

    def prob(self, context: Tuple[str, ...], token: str) -> float:
        denom = self.counts.context.get(context, 0)
        if denom == 0:
            return 0.0
        numer = self.counts.ngram.get(context + (token,), 0)
        return numer / denom


class KneserNeySmoothing(SmoothingStrategy):
    name = "kneser_ney"

    def __init__(self, discount: float = 0.9) -> None:
        if not (0.0 < discount < 1.0):
            raise ValueError("discount must be between 0 and 1")
        self.discount = discount

    def fit(self, counts: NGramCounts, vocab_size: int) -> None:
        super().fit(counts, vocab_size)
        self.ngram_by_order = counts.ngram_by_order
        self.context_by_order = counts.context_by_order
        self.max_order = max(self.ngram_by_order.keys()) if self.ngram_by_order else 1

        self.continuation_counts: dict[int, dict[Tuple[str, ...], int]] = {}
        for order in range(2, self.max_order + 1):
            followers: dict[Tuple[str, ...], set[str]] = defaultdict(set)
            for ngram in self.ngram_by_order.get(order, {}):
                context = ngram[:-1]
                token = ngram[-1]
                followers[context].add(token)
            self.continuation_counts[order - 1] = {
                context: len(tokens) for context, tokens in followers.items()
            }

        self.unigram_continuations: dict[str, int] = {}
        self.total_bigram_types = 0
        bigram_types = self.ngram_by_order.get(2, {})
        if bigram_types:
            predecessors: dict[str, set[str]] = defaultdict(set)
            for ngram in bigram_types:
                token = ngram[1]
                prev = ngram[0]
                predecessors[token].add(prev)
            self.unigram_continuations = {
                token: len(prevs) for token, prevs in predecessors.items()
            }
            self.total_bigram_types = sum(self.unigram_continuations.values())

        self.lambda_by_order: dict[int, dict[Tuple[str, ...], float]] = {}
        for order in range(1, self.max_order):
            lambda_for_order: dict[Tuple[str, ...], float] = {}
            continuation = self.continuation_counts.get(order, {})
            context_counts = self.context_by_order.get(order, {})
            for context, unique_followers in continuation.items():
                denom = context_counts.get(context, 0)
                if denom > 0:
                    lambda_for_order[context] = (self.discount * unique_followers) / denom
            self.lambda_by_order[order] = lambda_for_order

        self._kn_prob_cached.cache_clear()

    def prob(self, context: Tuple[str, ...], token: str) -> float:
        order = min(len(context) + 1, self.max_order)
        if order <= 1:
            return self._unigram_prob(token)
        context = context[-(self.max_order - 1) :]
        return self._kn_prob_cached(context, token, order)

    def _unigram_prob(self, token: str) -> float:
        if self.total_bigram_types == 0:
            return 0.0
        return self.unigram_continuations.get(token, 0) / self.total_bigram_types

    @lru_cache(maxsize=1_000_000)
    def _kn_prob_cached(self, context: Tuple[str, ...], token: str, order: int) -> float:
        ngram_by_order = self.ngram_by_order
        context_by_order = self.context_by_order
        lambda_by_order = self.lambda_by_order
        discount = self.discount

        prob = 0.0
        weight = 1.0
        cur_context = context[-(order - 1) :]
        cur_order = order

        while True:
            if cur_order == 1:
                prob += weight * self._unigram_prob(token)
                break

            ngram = cur_context + (token,)
            ngram_count = ngram_by_order.get(cur_order, {}).get(ngram, 0)
            context_count = context_by_order.get(cur_order - 1, {}).get(cur_context, 0)
            if context_count == 0:
                if len(cur_context) == 0:
                    prob += weight * self._unigram_prob(token)
                    break
                cur_context = cur_context[1:]
                cur_order -= 1
                continue

            first_term = max(ngram_count - discount, 0.0) / context_count
            prob += weight * first_term

            lambda_context = lambda_by_order.get(cur_order - 1, {}).get(cur_context, 0.0)
            weight *= lambda_context
            if weight == 0.0:
                break

            cur_context = cur_context[1:]
            cur_order -= 1

        return prob


class WittenBellSmoothing(SmoothingStrategy):
    name = "witten_bell"

    def fit(self, counts: NGramCounts, vocab_size: int) -> None:
        super().fit(counts, vocab_size)
        self.ngram_by_order = counts.ngram_by_order
        self.context_by_order = counts.context_by_order
        self.max_order = max(self.ngram_by_order.keys()) if self.ngram_by_order else 1

        self.total_unigrams = sum(self.ngram_by_order.get(1, {}).values())
        self.types_by_context: dict[int, dict[Tuple[str, ...], int]] = {}
        for order in range(2, self.max_order + 1):
            followers: dict[Tuple[str, ...], set[str]] = defaultdict(set)
            for ngram in self.ngram_by_order.get(order, {}):
                context = ngram[:-1]
                token = ngram[-1]
                followers[context].add(token)
            self.types_by_context[order - 1] = {
                context: len(tokens) for context, tokens in followers.items()
            }

        self._wb_prob_cached.cache_clear()

    def prob(self, context: Tuple[str, ...], token: str) -> float:
        order = min(len(context) + 1, self.max_order)
        if order <= 1:
            return self._unigram_prob(token)
        context = context[-(self.max_order - 1) :]
        return self._wb_prob_cached(context, token, order)

    def _unigram_prob(self, token: str) -> float:
        if self.total_unigrams == 0:
            return 0.0
        return self.ngram_by_order.get(1, {}).get((token,), 0) / self.total_unigrams

    @lru_cache(maxsize=1_000_000)
    def _wb_prob_cached(self, context: Tuple[str, ...], token: str, order: int) -> float:
        ngram_by_order = self.ngram_by_order
        context_by_order = self.context_by_order
        types_by_context = self.types_by_context

        cur_context = context[-(order - 1) :]
        cur_order = order

        while True:
            if cur_order == 1:
                return self._unigram_prob(token)

            ngram = cur_context + (token,)
            ngram_count = ngram_by_order.get(cur_order, {}).get(ngram, 0)
            context_count = context_by_order.get(cur_order - 1, {}).get(cur_context, 0)

            if context_count == 0:
                if len(cur_context) == 0:
                    return self._unigram_prob(token)
                cur_context = cur_context[1:]
                cur_order -= 1
                continue

            unique_followers = types_by_context.get(cur_order - 1, {}).get(cur_context, 0)
            denom = context_count + unique_followers
            if denom == 0:
                return self._unigram_prob(token)

            if ngram_count > 0:
                return ngram_count / denom

            backoff_weight = unique_followers / denom
            cur_context = cur_context[1:]
            cur_order -= 1
            if backoff_weight == 0.0:
                return self._unigram_prob(token)
            return backoff_weight * self._wb_prob_cached(cur_context, token, cur_order)


class NGramLanguageModel(LanguageModel):
    """Probabilistic n-gram LM with greedy generation."""

    def __init__(
        self,
        n: int = 4,
        smoothing: Optional[SmoothingStrategy] = None,
        add_bos_eos: bool = False,
        add_unk_token: bool = True,
        bos_token: str = BOS_TOKEN,
        eos_token: str = EOS_TOKEN,
        unk_token: str = UNK_TOKEN,
    ) -> None:
        if n < 2:
            raise ValueError("n must be >= 2")
        self.n = n
        self.add_bos_eos = add_bos_eos
        self.add_unk_token = add_unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.smoothing = smoothing or MLESmoothing()
        self.vocab: set[str] = set()
        self.counts = NGramCounts(Counter(), Counter())
        self.counts.ngram_by_order[self.n] = self.counts.ngram
        self.counts.context_by_order[self.n - 1] = self.counts.context

    def fit(self, sequences: Iterable[Sequence[str]]) -> None:
        for seq in sequences:
            tokens = list(seq)
            if self.add_bos_eos:
                tokens = [self.bos_token] + tokens + [self.eos_token]
            self.vocab.update(tokens)
            self._update_counts(tokens)
        if self.add_unk_token:
            self.vocab.add(self.unk_token)
            self._ensure_unk_counts()
        self.smoothing.fit(self.counts, len(self.vocab))

    def save(self, path: str) -> None:
        payload = self.to_dict()
        with open(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "NGramLanguageModel":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        return cls.from_dict(payload)

    def to_dict(self) -> dict:
        def _serialize_counter(counter: Counter) -> List[List[object]]:
            return [[list(k), v] for k, v in counter.items()]

        def _serialize_by_order(by_order: dict[int, Counter]) -> dict:
            return {str(order): _serialize_counter(counter) for order, counter in by_order.items()}

        return {
            "n": self.n,
            "add_bos_eos": self.add_bos_eos,
            "add_unk_token": self.add_unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "vocab": sorted(self.vocab),
            "smoothing": {
                "name": self.smoothing.name,
                "params": {"discount": getattr(self.smoothing, "discount", None)},
            },
            "counts": {
                "ngram_by_order": _serialize_by_order(self.counts.ngram_by_order),
                "context_by_order": _serialize_by_order(self.counts.context_by_order),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "NGramLanguageModel":
        def _deserialize_counter(items: List[List[object]]) -> Counter:
            counter = Counter()
            for key_list, value in items:
                counter[tuple(key_list)] = value
            return counter

        def _deserialize_by_order(by_order: dict) -> dict[int, Counter]:
            result: dict[int, Counter] = {}
            for order_str, items in by_order.items():
                result[int(order_str)] = _deserialize_counter(items)
            return result

        smoothing_info = payload.get("smoothing", {})
        name = smoothing_info.get("name", "mle")
        params = smoothing_info.get("params", {})
        if name == "kneser_ney":
            smoothing = KneserNeySmoothing(discount=params.get("discount", 0.75))
        elif name == "witten_bell":
            smoothing = WittenBellSmoothing()
        else:
            smoothing = MLESmoothing()

        model = cls(
            n=payload["n"],
            smoothing=smoothing,
            add_bos_eos=payload.get("add_bos_eos", False),
            add_unk_token=payload.get("add_unk_token", True),
            bos_token=payload.get("bos_token", BOS_TOKEN),
            eos_token=payload.get("eos_token", EOS_TOKEN),
            unk_token=payload.get("unk_token", UNK_TOKEN),
        )

        ngram_by_order = _deserialize_by_order(payload["counts"]["ngram_by_order"])
        context_by_order = _deserialize_by_order(payload["counts"]["context_by_order"])
        model.counts = NGramCounts(
            ngram=ngram_by_order.get(model.n, Counter()),
            context=context_by_order.get(model.n - 1, Counter()),
            ngram_by_order=ngram_by_order,
            context_by_order=context_by_order,
        )
        model.vocab = set(payload.get("vocab", []))
        model.smoothing.fit(model.counts, len(model.vocab))
        return model

    def score(self, sequence: Sequence[str]) -> float:
        tokens = self._map_unknowns(sequence)
        if len(tokens) < 2:
            return float("-inf")
        total_log_prob = 0.0
        for i in range(len(tokens) - 1):
            context = self._context(tokens[: i + 1])
            token = tokens[i + 1]
            prob = self.smoothing.prob(context, token)
            if prob <= 0.0:
                return float("-inf")
            total_log_prob += math.log(prob)
        return total_log_prob

    def predict_next(self, prefix: Sequence[str], return_prob: bool = False):
        context = self._context(prefix)
        best_token = None
        best_prob = -1.0
        for token in self.vocab:
            prob = self.smoothing.prob(context, token)
            if prob > best_prob:
                best_prob = prob
                best_token = token
        if return_prob:
            return best_token, best_prob
        return best_token

    def predict_top_k(self, prefix: Sequence[str], k: int = 5) -> List[Tuple[str, float]]:
        context = self._context(prefix)
        candidates: List[Tuple[str, float]] = []
        for token in self.vocab:
            prob = self.smoothing.prob(context, token)
            candidates.append((token, prob))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:k]

    def generate(
        self,
        prefix: Sequence[str],
        max_tokens: int = 50,
        stop_token: str = EOS_TOKEN,
    ) -> List[str]:
        tokens = list(prefix)
        for _ in range(max_tokens):
            next_token = self.predict_next(tokens)
            if next_token is None:
                break
            tokens.append(next_token)
            if next_token == stop_token:
                break
        return tokens

    def perplexity(self, sequences: Iterable[Sequence[str]]) -> float:
        total_log_prob = 0.0
        total_predicted = 0
        for sequence in sequences:
            tokens = self._map_unknowns(sequence)
            if len(tokens) < 2:
                continue
            for i in range(len(tokens) - 1):
                context = self._context(tokens[: i + 1])
                token = tokens[i + 1]
                prob = self.smoothing.prob(context, token)
                if prob <= 0.0:
                    return float("inf")
                total_log_prob += math.log(prob)
                total_predicted += 1
        if total_predicted == 0:
            return float("inf")
        return math.exp(-total_log_prob / total_predicted)

    def _update_counts(self, tokens: Sequence[str]) -> None:
        n = self.n
        for order in range(1, n + 1):
            if order == n:
                ngram_counter = self.counts.ngram
                context_counter = self.counts.context
            else:
                ngram_counter = self.counts.ngram_by_order.setdefault(order, Counter())
                context_counter = self.counts.context_by_order.setdefault(order - 1, Counter())

            for i in range(len(tokens) - order + 1):
                ngram = tuple(tokens[i : i + order])
                ngram_counter[ngram] += 1
                if order > 1:
                    context = ngram[:-1]
                    context_counter[context] += 1

    def _context(self, prefix: Sequence[str]) -> Tuple[str, ...]:
        tokens = self._map_unknowns(prefix)
        needed = self.n - 1
        context = tokens[-needed:]
        if len(context) < needed:
            pad_count = needed - len(context)
            if self.bos_token in self.vocab:
                context = [self.bos_token] * pad_count + context
        return tuple(context)

    def _ensure_unk_counts(self) -> None:
        unigram_counter = self.counts.ngram_by_order.setdefault(1, Counter())
        if (self.unk_token,) not in unigram_counter:
            unigram_counter[(self.unk_token,)] = 1

        if self.n >= 2:
            bigram_counter = self.counts.ngram_by_order.setdefault(2, Counter())
            context_counter = self.counts.context_by_order.setdefault(1, Counter())
            bigram = (self.unk_token, self.unk_token)
            if bigram not in bigram_counter:
                bigram_counter[bigram] = 1
                context_counter[(self.unk_token,)] = context_counter.get((self.unk_token,), 0) + 1

    def _map_unknowns(self, sequence: Sequence[str]) -> List[str]:
        if not self.add_unk_token:
            return [t for t in sequence if t in self.vocab]
        return [t if t in self.vocab else self.unk_token for t in sequence]



