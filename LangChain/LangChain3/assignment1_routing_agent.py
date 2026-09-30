"""Assignment 1: The "Confused Agent" Routing Challenge.

Two similar billing tools: refund_order (money back for one past charge) and
cancel_subscription (stop all future charges). The agent chooses between them using ONLY
the tools' docstrings: LangChain turns each docstring into the tool description and argument
schema sent to the model, and the model decides which tool to call. There is no if/else
routing anywhere in this file.

Usage:
    python assignment1_routing_agent.py                        # the two required tests
    python assignment1_routing_agent.py "Stop taking money from my account! jo@x.com"
"""

import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from llm_setup import CHAT_MODEL, get_llm


@tool(parse_docstring=True)
def refund_order(transaction_id: str) -> str:
    """Refund ONE specific past payment: return the money for a charge that already happened.

    Use this when the customer wants money BACK for a particular transaction they were
    already charged, for example: "give it back", "I want my money back for last month",
    "refund this charge", "I was charged by mistake", "that payment was wrong".
    The request refers to a single past charge, usually with a transaction ID (e.g.
    TXN991), an amount, or a date.

    Do NOT use this to stop future billing or to close an account: refunding a payment
    does not cancel the subscription. For "stop charging me" requests, use
    cancel_subscription instead.

    Args:
        transaction_id: The ID of the past transaction to refund, without a leading "#"
            (e.g. "TXN991").
    """
    return f"Refund issued for transaction {transaction_id}. Funds return in 5-7 business days."


@tool(parse_docstring=True)
def cancel_subscription(email: str) -> str:
    """Cancel a recurring subscription: stop ALL FUTURE payments and end the service.

    Use this when the customer wants future charges to stop or wants to leave the service,
    for example: "stop charging me", "stop taking money from my account", "I don't want to
    use your software anymore", "cancel my plan", "end my membership", "unsubscribe me".
    The account is identified by the customer's email address.

    Do NOT use this to return money for a charge that already happened: cancelling does
    not refund past payments. For "give my money back" requests about a specific charge,
    use refund_order instead.

    Args:
        email: The email address of the account whose subscription should be cancelled
            (e.g. "john@email.com").
    """
    return f"Subscription for {email} cancelled. No further payments will be taken."


TOOLS = [refund_order, cancel_subscription]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}  # to run whichever tool the model chose

# A neutral system prompt: it gives no routing hints; the docstrings do all the work
SYSTEM_PROMPT = ("You are a billing support agent for a SaaS platform. Handle the customer's "
                 "request by calling the single most appropriate tool.")

TEST_CASES = [
    ("Cancel Test", "I don't want to use your software anymore, stop charging john@email.com.",
     "cancel_subscription"),
    ("Refund Test", "My last charge of $50 on ID #TXN991 was a mistake, give it back.",
     "refund_order"),
]


def route(agent, user_message):
    """Let the model pick a tool; run it. Returns (tool name, arguments, tool output)."""
    reply = agent.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(user_message)])
    if not reply.tool_calls:
        return None, None, reply.content
    call = reply.tool_calls[0]
    output = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
    return call["name"], call["args"], output


if __name__ == "__main__":
    print(f"Model: {CHAT_MODEL}")
    print("Tools (descriptions come straight from the docstrings):")
    for t in TOOLS:
        print(f"  - {t.name}({', '.join(t.args)}): {t.description.split(". ")[0]}.")

    if len(sys.argv) > 1:
        cases = [("Custom Test", " ".join(sys.argv[1:]), None)]
    else:
        cases = TEST_CASES

    agent = get_llm(temperature=0).bind_tools(TOOLS)  # the model, given both tool schemas
    passed = 0
    for title, message, expected in cases:
        tool_name, args, output = route(agent, message)
        print(f"\n{title}")
        print(f'  User:          "{message}"')
        print(f"  Tool selected: {tool_name or '(no tool - model replied in text)'}")
        print(f"  Arguments:     {args}")
        print(f"  Tool output:   {output}")
        if expected:
            # Grading only - this compares the result AFTER the model has routed
            ok = tool_name == expected
            passed += ok
            print(f"  Expected:      {expected} -> {'PASS' if ok else 'FAIL'}")

    if len(sys.argv) == 1:
        print(f"\n{passed}/{len(cases)} tests passed"
              + ("" if passed == len(cases) else " - the docstrings need work"))
