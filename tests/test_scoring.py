"""Exact-match scoring: the parser must accept the answer formats models actually produce."""
import pytest

from tokens_per_task.ledger import SUPPLIERS
from tokens_per_task.scoring import parse_number, score_numeric, score_t1, score_t3


@pytest.mark.parametrize("reply,key,ok", [
    ("so 9*2 = 18\nAnswer: 18", "18", True),
    ("**Answer:** $1,234", "1234", True),
    ("The answer is \\boxed{42}.", "42", True),          # no marker: last number
    ("Answer: 18 (9 eggs x 2)", "18", True),
    ("Answer: 17.5", "18", False),
    ("Answer: -3", "-3", True),
    ("no number here", "5", False),
])
def test_numeric(reply, key, ok):
    assert score_numeric(reply, key)[0] is ok


@pytest.mark.parametrize("reply,ok", [
    ("card_arrival", True),
    ("  `Card_Arrival`.\n", True),
    ("The label is card_arrival", False),               # extra words are wrong
])
def test_label(reply, ok):
    assert score_t1(reply, "card_arrival")[0] is ok


def test_t3_numbers_within_one_percent():
    q = {"answer_type": "number", "answer": 12345.67}
    assert score_t3("Answer: 12345.67", q, SUPPLIERS)[0]
    assert score_t3("Answer: 12.345,67 EUR", q, SUPPLIERS)[0]
    assert score_t3("Answer: 12,400", q, SUPPLIERS)[0]
    assert not score_t3("Answer: 12600", q, SUPPLIERS)[0]
    assert score_t3("Answer: -12.3%", {"answer_type": "number", "answer": -12.3}, SUPPLIERS)[0]


def test_t3_supplier_names():
    q = {"answer_type": "supplier", "answer": "Moreau Fournitures SARL"}
    assert score_t3("Answer: Moreau Fournitures SARL", q, SUPPLIERS)[0]
    assert score_t3("Answer: **moreau fournitures sarl** (EUR 50,000)", q, SUPPLIERS)[0]
    assert not score_t3("Answer: Lindqvist Datasystem AB", q, SUPPLIERS)[0]


@pytest.mark.parametrize("tok,val", [("1.234,56", 1234.56), ("1,234.56", 1234.56), ("12,5", 12.5),
                                     ("1,234", 1234), ("−7", -7), ("1,2,3", None)])
def test_parse_number(tok, val):
    assert parse_number(tok) == val
