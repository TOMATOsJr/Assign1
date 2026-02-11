"""Tokenizer interface and stubs.

Simplifying assumptions (to be refined in implementations):
- Tokenization only (no encode/decode).
- Tokens are strings, not ids.
- Special tokens like <EOS> and <DASH> should be preserved as whole tokens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

try:
	from tqdm import tqdm
except ImportError:
	tqdm = None


@dataclass
class BaseTokenizer(ABC):
	"""Common interface for all tokenizers."""

	name: str
	config: Dict[str, Any] = field(default_factory=dict)

	@abstractmethod
	def tokenize(self, text: str) -> List[str]:
		"""Tokenize text into a list of string tokens."""
		raise NotImplementedError


class WhitespaceTokenizer(BaseTokenizer):
	"""Whitespace-based tokenizer stub.

	TODO: Implement whitespace splitting with special-token handling and
	splitting of punctuation/special characters as standalone tokens.
	"""

	def __init__(self, **config: Any) -> None:
		super().__init__(name="whitespace", config=config)

	def tokenize(self, text: str) -> List[str]:
		return text.strip().split()


class RegexTokenizer(BaseTokenizer):
	"""Regex-based tokenizer stub.

	TODO: Define a regex pattern and implement tokenization.
	"""

	def __init__(self, pattern: str | None = None, **config: Any) -> None:
		config = dict(config)
		if pattern is not None:
			config["pattern"] = pattern
		super().__init__(name="regex", config=config)
		self.pattern = pattern

	def tokenize(self, text: str) -> List[str]:
		raise NotImplementedError("RegexTokenizer.tokenize is not implemented yet")


class BPETokenizer(BaseTokenizer):
	"""BPE tokenizer stub.

	TODO: Implement training and tokenization using BPE merges/vocab.
	"""

	def __init__(self, vocab_size: int | None = None, **config: Any) -> None:
		config = dict(config)
		if vocab_size is not None:
			config["vocab_size"] = vocab_size
		super().__init__(name="bpe", config=config)
		self.vocab_size = vocab_size
		self.special_tokens = {"<BOS>", "<EOS>", "<DASH>"}
		self.merges: List[Tuple[str, str]] = []
		self.merge_ranks: Dict[Tuple[str, str], int] = {}
		self.bpe_cache: Dict[str, List[str]] = {}

	def train(self, texts: Iterable[str]) -> None:
		if self.vocab_size is None:
			raise ValueError("vocab_size must be set before training.")

		word_freq: Counter[str] = Counter()
		for text in texts:
			for token in text.strip().split():
				if token:
					word_freq[token] += 1

		word_symbols: Dict[str, Tuple[str, ...]] = {}
		for word in word_freq:
			if word in self.special_tokens:
				continue
			word_symbols[word] = tuple(word) + ("</w>",)

		vocab = set(self.special_tokens)
		for symbols in word_symbols.values():
			vocab.update(symbols)

		num_merges = self.vocab_size - len(vocab)
		print(f"Starting BPE training: Learning {num_merges} merges...")

		# Build initial pair counts and pair-to-words mapping
		pair_counts: Counter[Tuple[str, str]] = Counter()
		pair_to_words: Dict[Tuple[str, str], set[str]] = {}

		for word, symbols in word_symbols.items():
			if len(symbols) < 2:
				continue
			for i in range(len(symbols) - 1):
				pair = (symbols[i], symbols[i + 1])
				pair_counts[pair] += word_freq[word]
				if pair not in pair_to_words:
					pair_to_words[pair] = set()
				pair_to_words[pair].add(word)

		merge_iter = range(num_merges)
		if tqdm is not None:
			merge_iter = tqdm(merge_iter, desc="Learning merges", unit="merge")

		for _ in merge_iter:
			if len(vocab) >= self.vocab_size or not pair_counts:
				break

			# Find best pair
			best_pair = pair_counts.most_common(1)[0][0]
			self.merges.append(best_pair)
			self.merge_ranks[best_pair] = len(self.merges) - 1
			merged_token = "".join(best_pair)
			vocab.add(merged_token)

			# Only update words that contain the best pair
			affected_words = pair_to_words.get(best_pair, set()).copy()

			for word in affected_words:
				old_symbols = word_symbols[word]
				new_symbols = self._merge_symbols(old_symbols, best_pair)

				if old_symbols == new_symbols:
					continue

				# Remove old pairs from counts
				if len(old_symbols) >= 2:
					for i in range(len(old_symbols) - 1):
						old_pair = (old_symbols[i], old_symbols[i + 1])
						pair_counts[old_pair] -= word_freq[word]
						if pair_counts[old_pair] <= 0:
							del pair_counts[old_pair]
							if old_pair in pair_to_words:
								del pair_to_words[old_pair]
						elif old_pair in pair_to_words:
							pair_to_words[old_pair].discard(word)

				# Add new pairs to counts
				word_symbols[word] = new_symbols
				if len(new_symbols) >= 2:
					for i in range(len(new_symbols) - 1):
						new_pair = (new_symbols[i], new_symbols[i + 1])
						pair_counts[new_pair] += word_freq[word]
						if new_pair not in pair_to_words:
							pair_to_words[new_pair] = set()
						pair_to_words[new_pair].add(word)

		print(f"BPE training complete. Learned {len(self.merges)} merges.")
		self.bpe_cache.clear()

	def tokenize(self, text: str) -> List[str]:
		output: List[str] = []
		for token in text.strip().split():
			if token in self.special_tokens:
				output.append(token)
				continue
			output.extend(self._bpe(token))
		return output

	def _merge_symbols(self, symbols: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
		first, second = pair
		merged: List[str] = []
		i = 0
		while i < len(symbols):
			if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
				merged.append(first + second)
				i += 2
			else:
				merged.append(symbols[i])
				i += 1
		return tuple(merged)

	def _get_pairs(self, word: Tuple[str, ...]) -> set[Tuple[str, str]]:
		pairs: set[Tuple[str, str]] = set()
		if len(word) < 2:
			return pairs
		prev = word[0]
		for char in word[1:]:
			pairs.add((prev, char))
			prev = char
		return pairs

	def _bpe(self, token: str) -> List[str]:
		cached = self.bpe_cache.get(token)
		if cached is not None:
			return cached

		if not self.merge_ranks:
			pieces = list(token)
			self.bpe_cache[token] = pieces
			return pieces

		word: Tuple[str, ...] = tuple(token) + ("</w>",)
		while True:
			pairs = self._get_pairs(word)
			if not pairs:
				break
			best = min(pairs, key=lambda p: self.merge_ranks.get(p, float("inf")))
			if best not in self.merge_ranks:
				break
			word = self._merge_symbols(word, best)

		if word:
			last = word[-1]
			if last == "</w>":
				word = word[:-1]
			elif last.endswith("</w>"):
				stripped = last[: -len("</w>")]
				if stripped:
					word = word[:-1] + (stripped,)
				else:
					word = word[:-1]
		pieces = list(word)
		self.bpe_cache[token] = pieces
		return pieces


def get_tokenizer(kind: str, **kwargs: Any) -> BaseTokenizer:
	"""Factory helper for constructing tokenizer stubs."""

	kind_lower = kind.strip().lower()
	if kind_lower in {"whitespace", "ws"}:
		return WhitespaceTokenizer(**kwargs)
	if kind_lower in {"regex", "re"}:
		return RegexTokenizer(**kwargs)
	if kind_lower in {"bpe"}:
		return BPETokenizer(**kwargs)
	raise ValueError(f"Unknown tokenizer kind: {kind}")
