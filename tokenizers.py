"""Tokenizer interface and stubs.

Simplifying assumptions (to be refined in implementations):
- Tokenization only (no encode/decode).
- Tokens are strings, not ids.
- Special tokens like <EOS> and <DASH> should be preserved as whole tokens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


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

	def tokenize(self, text: str) -> List[str]:
		raise NotImplementedError("BPETokenizer.tokenize is not implemented yet")


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
