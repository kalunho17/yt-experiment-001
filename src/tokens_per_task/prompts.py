"""Prompt templates, one per task. Single user turn, no system prompt.

These strings are part of the experiment: changing them changes token counts.
"""
from __future__ import annotations


def t1_prompt(text: str, labels: list[str]) -> str:
    """T1 Classify (Banking77): pick one of 77 intent labels."""
    return ("Classify the customer message below into exactly one of these 77 intent labels:\n\n"
            + "\n".join(labels)
            + f"\n\nCustomer message: \"{text}\"\n\n"
            "Reply with the label only, exactly as written in the list, and nothing else.")


def t2_prompt(question: str) -> str:
    """T2 Calculate (GSM8K): a multi-step word problem with a numeric answer."""
    return ("Solve the following problem. End your reply with a final line of the form "
            "'Answer: <number>' (the number only, no units).\n\n"
            f"Problem: {question}")


def t3_prompt(question: str) -> str:
    """T3 Agent: an analyst question answered by calling tools on a synthetic ledger."""
    return ("You are a financial analyst with read access to a company's supplier invoice ledger "
            "for 2025 (amounts in EUR). Quarters: Q1 = Jan-Mar, Q2 = Apr-Jun, Q3 = Jul-Sep, "
            "Q4 = Oct-Dec 2025. Use the tools to look up the invoices and to do the arithmetic; "
            "do not guess numbers.\n\n"
            f"Question: {question}\n\n"
            "When you have the result, end your reply with a final line of the form "
            "'Answer: <value>', where <value> is a plain number without units or thousands "
            "separators, or a supplier name exactly as list_suppliers returns it.")


def t4_prompt(problem: str) -> str:
    """T4 Hard maths (MATH-500, level 5, integer answers)."""
    return ("Solve the following competition maths problem. The answer is an integer. End your reply "
            "with a final line of the form 'Answer: <integer>'.\n\n"
            f"Problem: {problem}")
