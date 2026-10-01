"""Smart Expense Processing Workflow (LangGraph).

A user submits an expense in USD. The graph:
  1. add_tax         adds 10% tax
  2. convert_to_inr  converts the taxed total to Indian rupees
  3. a conditional edge routes the request by the submitted amount:
       <= 100 USD          -> auto_approve
       100 < amount <= 1000 -> manager_approval
       > 1000 USD          -> finance_approval
  4. report          prints the final decision and converted amount

Usage:
    python expense_workflow.py              # run the demo expenses
    python expense_workflow.py 450          # process one expense of $450
    python expense_workflow.py --graph      # print the graph as a Mermaid diagram
"""

import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

TAX_RATE = 0.10
USD_TO_INR = 88.00          # fixed exchange rate (1 USD = 88 INR)
AUTO_APPROVE_LIMIT = 100    # USD
MANAGER_LIMIT = 1000        # USD


class ExpenseState(TypedDict, total=False):
    """The data passed between nodes. Each node returns only the fields it adds."""
    amount_usd: float   # submitted expense, before tax
    tax_usd: float
    total_usd: float    # amount + tax
    total_inr: float
    decision: str
    approver: str


# ---------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------
def add_tax(state: ExpenseState) -> ExpenseState:
    tax = round(state["amount_usd"] * TAX_RATE, 2)
    return {"tax_usd": tax, "total_usd": round(state["amount_usd"] + tax, 2)}


def convert_to_inr(state: ExpenseState) -> ExpenseState:
    return {"total_inr": round(state["total_usd"] * USD_TO_INR, 2)}


def auto_approve(state: ExpenseState) -> ExpenseState:
    return {"decision": "Auto Approved", "approver": "System (no review needed)"}


def manager_approval(state: ExpenseState) -> ExpenseState:
    return {"decision": "Sent for Manager Approval", "approver": "Line Manager"}


def finance_approval(state: ExpenseState) -> ExpenseState:
    return {"decision": "Sent for Finance Department Approval", "approver": "Finance Department"}


def report(state: ExpenseState) -> ExpenseState:
    print(f"  Expense submitted : ${state['amount_usd']:,.2f}")
    print(f"  Tax (10%)         : ${state['tax_usd']:,.2f}")
    print(f"  Total (USD)       : ${state['total_usd']:,.2f}")
    print(f"  Total (INR)       : ₹{state['total_inr']:,.2f}   (at ₹{USD_TO_INR:.2f} per $1)")
    print(f"  Decision          : {state['decision']}")
    print(f"  Handled by        : {state['approver']}")
    return {}


# ---------------------------------------------------------------
# Routing (used by the conditional edge)
# ---------------------------------------------------------------
def route_expense(state: ExpenseState) -> str:
    """Pick the approval node from the submitted (pre-tax) amount."""
    amount = state["amount_usd"]
    if amount <= AUTO_APPROVE_LIMIT:
        return "auto_approve"
    if amount <= MANAGER_LIMIT:
        return "manager_approval"
    return "finance_approval"


# ---------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------
def build_graph():
    graph = StateGraph(ExpenseState)
    for node in (add_tax, convert_to_inr, auto_approve, manager_approval, finance_approval,
                 report):
        graph.add_node(node.__name__, node)

    graph.add_edge(START, "add_tax")
    graph.add_edge("add_tax", "convert_to_inr")
    graph.add_conditional_edges("convert_to_inr", route_expense,
                                ["auto_approve", "manager_approval", "finance_approval"])
    for approval in ("auto_approve", "manager_approval", "finance_approval"):
        graph.add_edge(approval, "report")
    graph.add_edge("report", END)
    return graph.compile()


workflow = build_graph()


def process_expense(amount_usd):
    print(f"\nProcessing expense of ${amount_usd:,.2f}")
    # stream() yields each node's update as it runs, so we can show the path taken
    path = []
    final_state = {"amount_usd": amount_usd}
    for update in workflow.stream({"amount_usd": amount_usd}):
        for node, changes in update.items():
            path.append(node)
            final_state.update(changes or {})
    print(f"  Path              : {' -> '.join(path)}")
    return final_state


def parse_amount(text):
    try:
        amount = float(text.replace("$", "").replace(",", ""))
    except ValueError:
        raise SystemExit(f"Not a valid amount: {text!r}")
    if amount <= 0:
        raise SystemExit("The expense amount must be greater than 0.")
    return amount


if __name__ == "__main__":
    if "--graph" in sys.argv:
        print(workflow.get_graph().draw_mermaid())
    elif len(sys.argv) > 1:
        process_expense(parse_amount(sys.argv[1]))
    else:
        # One expense for each route, plus the exact boundaries
        for amount in (45, 100, 450, 1000, 2500):
            process_expense(amount)
