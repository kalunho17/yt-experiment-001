"""T3 environment: deterministic ledger, verified answer keys, safe tools."""
import json

import pytest

from tokens_per_task.config import GENERATED_DIR
from tokens_per_task.ledger import build_ledger, execute_tool, verify_question_with_tools


@pytest.fixture(scope="module")
def ledger():
    return build_ledger()


def test_shape(ledger):
    assert len(ledger["invoices"]) == 500
    assert len(ledger["suppliers"]) == 12
    assert len(ledger["questions"]) == 20


def test_every_key_recomputed_from_tool_outputs(ledger):
    for q in ledger["questions"]:
        ok, value, n_calls = verify_question_with_tools(ledger, q)
        assert ok, (q["id"], q["answer"], value)
        assert 2 <= n_calls <= 6


def test_matches_published_files(ledger):
    pub_ledger = json.loads((GENERATED_DIR / "t3_ledger.json").read_text(encoding="utf-8"))
    pub_questions = json.loads((GENERATED_DIR / "t3_questions.json").read_text(encoding="utf-8"))
    assert pub_ledger == {"suppliers": ledger["suppliers"], "invoices": ledger["invoices"]}
    assert pub_questions == json.loads(json.dumps(ledger["questions"]))


@pytest.mark.parametrize("name,args", [
    ("calculator", {"expression": "__import__('os').system('dir')"}),
    ("calculator", {"expression": "2 ** 10000000"}),
    ("calculator", {"expression": "1/0"}),
    ("calculator", {"expression": "1,234.5 + 1"}),        # thousands separator -> list -> error
    ("get_invoices", {"supplier": "Acme Ltd", "quarter": "Q1"}),
    ("get_invoices", {"supplier": "Brandt Werkzeugbau GmbH"}),
    ("delete_everything", {}),
])
def test_tools_return_errors_not_exceptions(ledger, name, args):
    assert "error" in json.loads(execute_tool(ledger, name, json.dumps(args)))


def test_tools_are_lenient_on_format(ledger):
    d = json.loads(execute_tool(ledger, "get_invoices",
                                json.dumps({"supplier": "BRANDT WERKZEUGBAU GMBH", "quarter": "2025-Q1"})))
    assert d["supplier"] == "Brandt Werkzeugbau GmbH" and d["quarter"] == "Q1 2025"
    assert json.loads(execute_tool(ledger, "calculator", '{"expression": "0.1 + 0.2"}'))["result"] == 0.3
    assert "error" in json.loads(execute_tool(ledger, "calculator", "not json"))
