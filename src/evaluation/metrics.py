"""Pure evaluation metrics for sequence and RNA secondary-structure outputs."""

from __future__ import annotations

from statistics import mean, median
from typing import Iterable, Sequence

BasePair = tuple[int, int]

_BRACKET_OPEN_TO_CLOSE = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">",
}
_BRACKET_CLOSE_TO_OPEN = {close: open_ for open_, close in _BRACKET_OPEN_TO_CLOSE.items()}
_LETTER_OPEN_TO_CLOSE = {chr(code): chr(code + 32) for code in range(ord("A"), ord("Z") + 1)}
_LETTER_CLOSE_TO_OPEN = {close: open_ for open_, close in _LETTER_OPEN_TO_CLOSE.items()}
_OPEN_TO_CLOSE = {**_BRACKET_OPEN_TO_CLOSE, **_LETTER_OPEN_TO_CLOSE}
_CLOSE_TO_OPEN = {**_BRACKET_CLOSE_TO_OPEN, **_LETTER_CLOSE_TO_OPEN}


def sequence_recovery(predicted: str, native: str) -> float:
    """Return exact-position sequence recovery after strict equal-length validation."""

    if len(predicted) != len(native):
        raise ValueError("sequence_recovery requires predicted and native sequences with equal length")
    if not native:
        return 1.0
    matches = sum(1 for predicted_base, native_base in zip(predicted, native) if predicted_base == native_base)
    return matches / len(native)


def metric_mean(values: Iterable[float]) -> float:
    """Aggregate metric values with the paper-style empty input convention."""

    items = list(values)
    return mean(items) if items else 0.0


def metric_median(values: Iterable[float]) -> float:
    """Aggregate metric values with the paper-style empty input convention."""

    items = list(values)
    return median(items) if items else 0.0


def length_bucket(length: int) -> str:
    """Return the paper length bucket: short <= 50, medium 51..99, long >= 100."""

    if length < 0:
        raise ValueError("length must be non-negative")
    if length <= 50:
        return "short"
    if length <= 99:
        return "medium"
    return "long"


def parse_extended_dot_bracket(structure: str) -> set[BasePair]:
    """Parse extended dot-bracket notation into zero-based base-pair indices.

    Supports (), [], {}, <>, and A/a through Z/z pseudoknot levels.
    """

    stacks: dict[str, list[int]] = {open_: [] for open_ in _OPEN_TO_CLOSE}
    pairs: set[BasePair] = set()
    for index, char in enumerate(structure):
        if char == ".":
            continue
        if char in _OPEN_TO_CLOSE:
            stacks[char].append(index)
            continue
        if char in _CLOSE_TO_OPEN:
            open_char = _CLOSE_TO_OPEN[char]
            if not stacks[open_char]:
                raise ValueError(f"unmatched closing token {char!r} at position {index}")
            pairs.add((stacks[open_char].pop(), index))
            continue
        raise ValueError(f"unsupported dot-bracket token {char!r} at position {index}")

    unclosed = [(open_char, positions[-1]) for open_char, positions in stacks.items() if positions]
    if unclosed:
        open_char, index = unclosed[0]
        raise ValueError(f"unmatched opening token {open_char!r} at position {index}")
    return pairs


def base_pair_precision(predicted: set[BasePair], native: set[BasePair]) -> float:
    """Return base-pair precision with explicit empty-set semantics."""

    if not predicted:
        return 1.0
    return len(predicted & native) / len(predicted)


def base_pair_recall(predicted: set[BasePair], native: set[BasePair]) -> float:
    """Return base-pair recall with explicit empty-set semantics."""

    if not native:
        return 1.0
    return len(predicted & native) / len(native)


def base_pair_f1(predicted: set[BasePair], native: set[BasePair]) -> float:
    """Return base-pair F1 with explicit empty-set semantics."""

    precision = base_pair_precision(predicted, native)
    recall = base_pair_recall(predicted, native)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def base_pair_scores(predicted: set[BasePair], native: set[BasePair]) -> dict[str, float]:
    """Return precision, recall, and F1 for parsed base-pair sets."""

    return {
        "precision": base_pair_precision(predicted, native),
        "recall": base_pair_recall(predicted, native),
        "f1": base_pair_f1(predicted, native),
    }


def lcs_normalized_similarity(left: str, right: str) -> float:
    """Return LCS length normalized by the shorter input length."""

    if not left and not right:
        return 1.0
    denominator = min(len(left), len(right))
    if denominator == 0:
        return 0.0
    return _lcs_length(left, right) / denominator


def intdiv(sequences: Sequence[str]) -> float:
    """Return IntDiv = 1 - 1/n^2 * sum ordered pair similarities, diagonal included."""

    n = len(sequences)
    if n == 0:
        return 0.0
    similarity_sum = sum(
        lcs_normalized_similarity(left, right)
        for left in sequences
        for right in sequences
    )
    return 1.0 - similarity_sum / float(n * n)


def _lcs_length(left: str, right: str) -> int:
    if len(left) < len(right):
        shorter, longer = left, right
    else:
        shorter, longer = right, left

    previous = [0] * (len(shorter) + 1)
    for longer_char in longer:
        current = [0]
        northwest = 0
        for index, shorter_char in enumerate(shorter, start=1):
            north = previous[index]
            west = current[index - 1]
            current.append(northwest + 1 if longer_char == shorter_char else max(north, west))
            northwest = north
        previous = current
    return previous[-1]


__all__ = [
    "BasePair",
    "base_pair_f1",
    "base_pair_precision",
    "base_pair_recall",
    "base_pair_scores",
    "intdiv",
    "lcs_normalized_similarity",
    "length_bucket",
    "metric_mean",
    "metric_median",
    "parse_extended_dot_bracket",
    "sequence_recovery",
]
