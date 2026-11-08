"""T3 environment: a synthetic supplier-invoice ledger, 20 analyst questions and three tools.

The ledger (500 invoices, 12 suppliers, calendar year 2025) and the questions are generated
deterministically from ``SAMPLE_SEED``. Every answer key is computed from the ledger, and
:func:`verify_question_with_tools` recomputes each key independently, using only the tool
outputs a model would see (run ``python run.py check``).
"""
from __future__ import annotations

import ast
import datetime as dt
import json
import math
import operator
import random
import re
import threading
from decimal import Decimal

from .config import SAMPLE_SEED
from .scoring import norm_name

SUPPLIERS = [
    "Brandt Werkzeugbau GmbH", "Nordhafen Logistik GmbH", "Kessler Industriebedarf KG",
    "Alpenblick Elektrotechnik AG", "Van Dijk Verpakkingen B.V.", "Rheintal Facility Services GmbH",
    "Lindqvist Datasystem AB", "Moreau Fournitures SARL", "Hansekontor Office Supplies GmbH",
    "Weidmann Maschinenbau GmbH & Co. KG", "Castellano Servizi S.r.l.", "Falkenried IT-Beratung GmbH",
]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
_LEDGER_CACHE: dict | None = None
_LEDGER_LOCK = threading.Lock()


def quarter_of(date: dt.date) -> str:
    return f"Q{(date.month - 1) // 3 + 1}"


# =============================================================================
# Ledger and questions
# =============================================================================
def build_ledger() -> dict:
    """500 invoices, 12 suppliers, 2025, seed 42; also the 20 questions with answer keys.

    The random stream is shared between invoices and questions, so the order of draws below
    must not change.
    """
    global _LEDGER_CACHE
    with _LEDGER_LOCK:
        if _LEDGER_CACHE is not None:
            return _LEDGER_CACHE
        rng = random.Random(SAMPLE_SEED)
        weights = [rng.uniform(0.6, 1.6) for _ in SUPPLIERS]
        scale = [rng.choice([450, 800, 1200, 1800, 2500, 3500, 5000, 7000]) for _ in SUPPLIERS]
        raw = []
        start = dt.date(2025, 1, 1)
        for _ in range(500):
            si = rng.choices(range(len(SUPPLIERS)), weights=weights)[0]
            date = start + dt.timedelta(days=rng.randrange(365))
            cents = max(5000, int(round(scale[si] * rng.lognormvariate(0, 0.55) * 100)))
            raw.append((date, SUPPLIERS[si], cents))
        raw.sort(key=lambda r: (r[0], r[1], r[2]))
        invoices = [{"invoice_id": f"INV-2025-{i + 1:04d}", "date": d.isoformat(), "supplier": s,
                     "quarter": quarter_of(d), "amount_cents": c} for i, (d, s, c) in enumerate(raw)]
        ledger = {"suppliers": list(SUPPLIERS), "invoices": invoices}
        ledger["questions"] = build_questions(ledger, rng)
        _LEDGER_CACHE = ledger
        return ledger


def cents_of(ledger: dict, supplier: str, quarters) -> list[int]:
    qs = set(quarters)
    return [i["amount_cents"] for i in ledger["invoices"] if i["supplier"] == supplier and i["quarter"] in qs]


def build_questions(ledger: dict, rng: random.Random) -> list[dict]:
    """20 questions of 8 kinds. ``expected_tool_calls`` is the (min, max) a careful agent needs."""
    sup = ledger["suppliers"]
    total = lambda s, q: sum(cents_of(ledger, s, [q]))  # noqa: E731
    halves = {"first half of 2025 (Q1 and Q2)": ["Q1", "Q2"], "second half of 2025 (Q3 and Q4)": ["Q3", "Q4"]}
    plan = (["total"] * 4 + ["highest"] * 2 + ["lowest"] + ["diff"] * 3 + ["qoq"] * 3
            + ["count_above"] * 3 + ["max_invoice"] * 2 + ["average"] * 2)
    rng.shuffle(plan)
    out, seen = [], set()
    for kind in plan:
        for _ in range(1000):
            q = rng.choice(QUARTERS)
            if kind == "total":
                s = rng.choice(sup)
                key = (kind, s, q)
                if key in seen or not cents_of(ledger, s, [q]):
                    continue
                text = f"What was the total amount invoiced by {s} in {q} 2025, in EUR?"
                ans, atype, params, calls = total(s, q) / 100, "number", {"supplier": s, "quarter": q}, (2, 2)
            elif kind in ("highest", "lowest"):
                ss = sorted(rng.sample(sup, 3))
                key = (kind, tuple(ss), q)
                tots = {s: total(s, q) for s in ss}
                vals = sorted(tots.values())
                # skip near-ties so the key is robust
                if key in seen or vals[0] == 0 or any(b - a < 0.02 * b for a, b in zip(vals, vals[1:])):
                    continue
                pick = max if kind == "highest" else min
                word = "highest" if kind == "highest" else "lowest"
                text = (f"Of the suppliers {ss[0]}, {ss[1]} and {ss[2]}, which one had the {word} "
                        f"total invoiced amount in {q} 2025?")
                ans, atype, params, calls = pick(tots, key=tots.get), "supplier", {"suppliers": ss, "quarter": q}, (3, 6)
            elif kind == "diff":
                a, b = rng.sample(sup, 2)
                key = (kind, a, b, q)
                if key in seen or not cents_of(ledger, a, [q]) or not cents_of(ledger, b, [q]):
                    continue
                text = (f"In {q} 2025, how much more did {a} invoice than {b}? Give {a}'s total minus "
                        f"{b}'s total in EUR (negative if {b}'s total is higher).")
                ans, atype, params, calls = (total(a, q) - total(b, q)) / 100, "number", {"a": a, "b": b, "quarter": q}, (3, 5)
            elif kind == "qoq":
                s = rng.choice(sup)
                qi = rng.randrange(3)
                q1, q2 = QUARTERS[qi], QUARTERS[qi + 1]
                key = (kind, s, q1)
                base, new = total(s, q1), total(s, q2)
                if key in seen or base == 0 or new == 0:
                    continue
                text = (f"By what percentage did the total amount invoiced by {s} change from {q1} to {q2} "
                        "2025? Give the change in percent, rounded to one decimal (negative for a decrease).")
                ans = float(round((Decimal(new) - Decimal(base)) / Decimal(base) * 100, 1))
                atype, params, calls = "number", {"supplier": s, "from": q1, "to": q2}, (3, 5)
            elif kind == "count_above":
                s = rng.choice(sup)
                label, qs = rng.choice(list(halves.items()))
                thr = rng.choice([1000, 2000, 2500, 5000])
                key = (kind, s, label)
                n = sum(1 for c in cents_of(ledger, s, qs) if c > thr * 100)
                if key in seen or n == 0 or n == len(cents_of(ledger, s, qs)):
                    continue
                text = (f"How many invoices from {s} in the {label} had an amount above "
                        f"EUR {thr:,}?")
                ans, atype, params, calls = n, "number", {"supplier": s, "quarters": qs, "threshold": thr}, (2, 3)
            elif kind == "max_invoice":
                s = rng.choice(sup)
                label, qs = rng.choice(list(halves.items()))
                key = (kind, s, label)
                cs = cents_of(ledger, s, qs)
                if key in seen or not cs:
                    continue
                text = f"What was the largest single invoice amount from {s} in the {label}, in EUR?"
                ans, atype, params, calls = max(cs) / 100, "number", {"supplier": s, "quarters": qs}, (2, 3)
            elif kind == "average":
                s = rng.choice(sup)
                key = (kind, s, q)
                cs = cents_of(ledger, s, [q])
                if key in seen or len(cs) < 2:
                    continue
                text = f"What was the average invoice amount from {s} in {q} 2025, in EUR rounded to cents?"
                ans = float(round(Decimal(sum(cs)) / Decimal(len(cs)) / 100, 2))
                atype, params, calls = "number", {"supplier": s, "quarter": q}, (2, 3)
            else:
                raise ValueError(kind)
            seen.add(key)
            out.append({"id": f"t3-{len(out) + 1:02d}", "kind": kind, "question": text,
                        "answer_type": atype, "answer": ans, "params": params,
                        "expected_tool_calls": list(calls)})
            break
        else:
            raise RuntimeError(f"could not build a '{kind}' question")
    return out


# =============================================================================
# Tools
# =============================================================================
def _find_supplier(ledger, name) -> str | None:
    n = norm_name(str(name))
    for s in ledger["suppliers"]:
        if norm_name(s) == n:
            return s
    return None


def _norm_quarter(q) -> str | None:
    s = str(q).upper().replace("2025", "")
    m = re.search(r"[1-4]", s)
    return f"Q{m.group(0)}" if m else None


def tool_list_suppliers(ledger) -> str:
    return json.dumps({"suppliers": ledger["suppliers"]}, ensure_ascii=False, separators=(",", ":"))


def tool_get_invoices(ledger, supplier, quarter) -> str:
    s = _find_supplier(ledger, supplier)
    if s is None:
        return json.dumps({"error": f"Unknown supplier '{supplier}'. Call list_suppliers for the exact names."})
    q = _norm_quarter(quarter)
    if q is None:
        return json.dumps({"error": f"Unknown quarter '{quarter}'. Use Q1, Q2, Q3 or Q4."})
    rows = [[i["invoice_id"], i["date"], i["amount_cents"] / 100]
            for i in ledger["invoices"] if i["supplier"] == s and i["quarter"] == q]
    return json.dumps({"supplier": s, "quarter": f"{q} 2025", "currency": "EUR", "count": len(rows),
                       "columns": ["invoice_id", "date", "amount_eur"], "rows": rows},
                      ensure_ascii=False, separators=(",", ":"))


_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "sum": sum}


def safe_eval(expr: str):
    """Evaluate an arithmetic expression via a whitelisted AST walk (no ``eval``)."""
    expr = str(expr).replace("×", "*").replace("÷", "/").replace("−", "-")
    expr = expr.replace("€", "").replace("$", "").strip()
    if len(expr) > 5000:
        raise ValueError("expression too long")
    tree = ast.parse(expr, mode="eval")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(b) > 100 or abs(a) > 1e6):
                raise ValueError("power too large")
            return _BINOPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UNOPS:
            return _UNOPS[type(n.op)](ev(n.operand))
        if isinstance(n, (ast.List, ast.Tuple)):
            return [ev(e) for e in n.elts]
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS
                and not n.keywords):
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"unsupported syntax ({type(n).__name__}); use numbers, + - * / ** % ( ) "
                         "and round/abs/min/max/sum")

    res = ev(tree)
    if isinstance(res, list):
        raise ValueError("result is a list; do not use commas as thousands separators")
    if isinstance(res, float) and not math.isfinite(res):
        raise ValueError("result is not finite")
    return res


def tool_calculator(expression) -> str:
    try:
        res = safe_eval(expression)
    except ZeroDivisionError:
        return json.dumps({"error": "division by zero"})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"cannot evaluate: {e}"})
    if isinstance(res, float):
        res = int(res) if res.is_integer() and abs(res) < 1e15 else round(res, 10)
    return json.dumps({"result": res})


TOOLS = [
    {"type": "function", "function": {
        "name": "list_suppliers", "description": "List the exact names of all suppliers in the 2025 ledger.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_invoices",
        "description": "Get all invoices of one supplier in one quarter of 2025 (invoice id, date, amount in EUR).",
        "parameters": {"type": "object", "properties": {
            "supplier": {"type": "string", "description": "Exact supplier name as returned by list_suppliers."},
            "quarter": {"type": "string", "enum": QUARTERS, "description": "Quarter of 2025."}},
            "required": ["supplier", "quarter"]}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '1200.5 + 830.25' or 'round((5300 - 4800) / 4800 * 100, 1)'. "
                       "Supports + - * / ** % ( ) and round, abs, min, max, sum. No thousands separators.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "The expression to evaluate."}},
            "required": ["expression"]}}},
]


def execute_tool(ledger: dict, name: str, arguments) -> str:
    """Run one tool call; errors are returned to the model as JSON, never raised."""
    try:
        args = arguments if isinstance(arguments, dict) else json.loads(arguments or "{}")
        if not isinstance(args, dict):
            raise ValueError
    except Exception:  # noqa: BLE001
        return json.dumps({"error": "arguments are not a valid JSON object"})
    try:
        if name == "list_suppliers":
            return tool_list_suppliers(ledger)
        if name == "get_invoices":
            if "supplier" not in args or "quarter" not in args:
                return json.dumps({"error": "get_invoices needs 'supplier' and 'quarter'"})
            return tool_get_invoices(ledger, args["supplier"], args["quarter"])
        if name == "calculator":
            if "expression" not in args:
                return json.dumps({"error": "calculator needs 'expression'"})
            return tool_calculator(args["expression"])
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"tool failed: {e}"})
    return json.dumps({"error": f"unknown tool '{name}'; available: list_suppliers, get_invoices, calculator"})


# =============================================================================
# Independent verification of the answer keys
# =============================================================================
def verify_question_with_tools(ledger: dict, q: dict) -> tuple[bool, object, int]:
    """Recompute an answer key only from tool outputs (independent of :func:`build_questions`).

    Returns (matches, recomputed value, number of tool calls used).
    """
    calls = 0

    def rows(s, qs):
        nonlocal calls
        out = []
        for qq in qs:
            calls += 1
            d = json.loads(execute_tool(ledger, "get_invoices", json.dumps({"supplier": s, "quarter": qq})))
            assert "error" not in d, d
            out += [Decimal(str(r[2])) for r in d["rows"]]
        return out

    def calc(expr):
        nonlocal calls
        calls += 1
        d = json.loads(execute_tool(ledger, "calculator", json.dumps({"expression": expr})))
        assert "error" not in d, d
        return Decimal(str(d["result"]))

    def tot(s, qq):
        r = rows(s, [qq])
        dec = sum(r, Decimal(0))
        via_calc = calc(" + ".join(str(x) for x in r))
        assert abs(dec - via_calc) < Decimal("0.005"), (dec, via_calc)
        return dec

    p, k = q["params"], q["kind"]
    if k == "total":
        v = tot(p["supplier"], p["quarter"])
    elif k in ("highest", "lowest"):
        t = {s: tot(s, p["quarter"]) for s in p["suppliers"]}
        v = (max if k == "highest" else min)(t, key=t.get)
    elif k == "diff":
        v = tot(p["a"], p["quarter"]) - tot(p["b"], p["quarter"])
    elif k == "qoq":
        a, b = tot(p["supplier"], p["from"]), tot(p["supplier"], p["to"])
        v = round((b - a) / a * 100, 1)
    elif k == "count_above":
        v = sum(1 for x in rows(p["supplier"], p["quarters"]) if x > p["threshold"])
    elif k == "max_invoice":
        v = max(rows(p["supplier"], p["quarters"]))
    elif k == "average":
        r = rows(p["supplier"], [p["quarter"]])
        v = round(sum(r, Decimal(0)) / len(r), 2)
        via_calc = calc(f"round(({' + '.join(str(x) for x in r)}) / {len(r)}, 2)")
        assert abs(v - via_calc) < Decimal("0.006"), (v, via_calc)
    else:
        raise ValueError(k)
    if q["answer_type"] == "supplier":
        return v == q["answer"], v, calls
    return abs(float(v) - float(q["answer"])) < 0.006, float(v), calls
