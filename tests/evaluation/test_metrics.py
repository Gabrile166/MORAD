from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    base_pair_f1,
    base_pair_precision,
    base_pair_recall,
    base_pair_scores,
    intdiv,
    lcs_normalized_similarity,
    length_bucket,
    metric_mean,
    metric_median,
    parse_extended_dot_bracket,
    sequence_recovery,
)


def test_sequence_recovery_requires_equal_lengths() -> None:
    assert sequence_recovery("AUGC", "AUGU") == pytest.approx(0.75)
    assert sequence_recovery("", "") == pytest.approx(1.0)

    with pytest.raises(ValueError, match="equal length"):
        sequence_recovery("AUG", "AUGC")


def test_metric_aggregators_support_mean_and_median() -> None:
    values = [1.0, 0.5, 0.0]

    assert metric_mean(values) == pytest.approx(0.5)
    assert metric_median(values) == pytest.approx(0.5)
    assert metric_mean([]) == pytest.approx(0.0)
    assert metric_median([]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("length", "bucket"),
    [
        (0, "short"),
        (50, "short"),
        (51, "medium"),
        (99, "medium"),
        (100, "long"),
    ],
)
def test_length_bucket_matches_paper_boundaries(length: int, bucket: str) -> None:
    assert length_bucket(length) == bucket


def test_length_bucket_rejects_negative_lengths() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        length_bucket(-1)


def test_parse_extended_dot_bracket_supports_brackets_and_pseudoknots() -> None:
    structure = "([{<AB..ba>}])"

    assert parse_extended_dot_bracket(structure) == {
        (0, 13),
        (1, 12),
        (2, 11),
        (3, 10),
        (4, 9),
        (5, 8),
    }


def test_parse_extended_dot_bracket_rejects_invalid_or_unbalanced_input() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        parse_extended_dot_bracket("(.)&")
    with pytest.raises(ValueError, match="unmatched closing"):
        parse_extended_dot_bracket(".)")
    with pytest.raises(ValueError, match="unmatched opening"):
        parse_extended_dot_bracket("(.")


def test_base_pair_scores_handle_partial_overlap() -> None:
    predicted = {(0, 5), (1, 4), (2, 3)}
    native = {(0, 5), (1, 4), (6, 7), (8, 9)}

    assert base_pair_precision(predicted, native) == pytest.approx(2 / 3)
    assert base_pair_recall(predicted, native) == pytest.approx(0.5)
    assert base_pair_f1(predicted, native) == pytest.approx(4 / 7)
    assert base_pair_scores(predicted, native) == {
        "precision": pytest.approx(2 / 3),
        "recall": pytest.approx(0.5),
        "f1": pytest.approx(4 / 7),
    }


def test_base_pair_scores_define_empty_set_semantics() -> None:
    assert base_pair_scores(set(), set()) == {
        "precision": pytest.approx(1.0),
        "recall": pytest.approx(1.0),
        "f1": pytest.approx(1.0),
    }
    assert base_pair_scores(set(), {(0, 1)}) == {
        "precision": pytest.approx(1.0),
        "recall": pytest.approx(0.0),
        "f1": pytest.approx(0.0),
    }
    assert base_pair_scores({(0, 1)}, set()) == {
        "precision": pytest.approx(0.0),
        "recall": pytest.approx(1.0),
        "f1": pytest.approx(0.0),
    }


def test_lcs_normalized_similarity_uses_shorter_sequence_denominator() -> None:
    assert lcs_normalized_similarity("ABCDEF", "ACE") == pytest.approx(1.0)
    assert lcs_normalized_similarity("GATTACA", "GCATGCU") == pytest.approx(4 / 7)
    assert lcs_normalized_similarity("", "") == pytest.approx(1.0)
    assert lcs_normalized_similarity("", "AUGC") == pytest.approx(0.0)


def test_intdiv_uses_ordered_pairs_with_diagonal() -> None:
    sequences = ["AA", "AB"]

    assert intdiv(sequences) == pytest.approx(1.0 - (1.0 + 0.5 + 0.5 + 1.0) / 4.0)
    assert intdiv(["AA", "AA"]) == pytest.approx(0.0)
    assert intdiv([]) == pytest.approx(0.0)
