"""Assignment 2: Agent Resilience - The Broken API Challenge.

A ReAct-style financial agent (Thought -> Action -> Observation, repeated) with two tools:
  - get_internal_stock_price: the primary source, deliberately broken (always times out)
  - search_public_web: a working backup that returns a mock web search result
The agent tries the primary database first, sees the failure, and recovers on its own by
switching to the backup. The printed trace proves each step.
"""

import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from llm_setup import CHAT_MODEL, get_llm

QUESTION = "What is the current stock price of Apple?"
MAX_STEPS = 6


@tool
def get_internal_stock_price(ticker: str) -> str:
    """PRIMARY source for stock prices: the company's internal market database.
    Always try this tool first for any stock price question. Input: a stock ticker
    symbol, e.g. "AAPL" for Apple."""
    time.sleep(1)  # simulate waiting on a database that never answers
    return "Error: Database Timeout"


# Mock search results (a real version would call a web search API)
MOCK_WEB_RESULTS = {
    "apple": "Apple Inc. (AAPL) stock is at $170.",
    "aapl": "Apple Inc. (AAPL) stock is at $170.",
    "microsoft": "Microsoft Corp. (MSFT) stock is at $420.",
}


@tool
def search_public_web(query: str) -> str:
    """BACKUP source: searches the public web and returns the top result. Use this when
    the internal database is unavailable or returns an error. Input: a search query,
    e.g. "Apple stock price"."""
    for keyword, result in MOCK_WEB_RESULTS.items():
        if keyword in query.lower():
            return f"Top web result: {result}"
    return "Top web result: no stock price found for that query."


TOOLS = {t.name: t for t in (get_internal_stock_price, search_public_web)}

SYSTEM_PROMPT = f"""You are a financial assistant agent. You answer questions by reasoning
step by step and using tools. You have these tools:

{chr(10).join(f"- {name}: {' '.join(t.description.split())}" for name, t in TOOLS.items())}

If a tool returns an error, do not give up: think about why it failed and try another
tool that can answer the question.

As soon as a tool gives you the information you need, reply with the Final Answer. Do not
repeat a tool call you have already made, and only claim what your observations show.

Reply in EXACTLY this format, and stop after "Action Input":
Thought: <your reasoning about what to do next>
Action: <one tool name>
Action Input: <the input for the tool>

When you have the answer, reply instead with:
Thought: <your reasoning>
Final Answer: <the answer>"""

FORMAT_REMINDER = ("Invalid format. Reply with 'Thought:', then either 'Action:' and "
                   "'Action Input:', or 'Final Answer:'.")


def parse_reply(text):
    """Split the model's reply into (thought, tool name, tool input, final answer)."""
    thought = re.search(r"Thought:\s*(.*?)(?=\n\s*(?:Action:|Final Answer:)|\Z)", text, re.S)
    action = re.search(r"Action:\s*(.+?)\s*\n\s*Action Input:\s*(.+)", text)
    final = re.search(r"Final Answer:\s*(.+)", text, re.S)
    thought = thought.group(1).strip() if thought else text.split("Action:")[0].strip()
    if action:
        name = action.group(1).strip().strip("[]\"'`")
        tool_input = action.group(2).strip().splitlines()[0].strip().strip("\"'`")
        return thought, name, tool_input, None
    if final:
        return thought, None, None, final.group(1).strip()
    return thought, None, None, None


def run_agent(question):
    """Run the agent loop; return (final answer, list of (tool, input, observation) steps)."""
    llm = get_llm(temperature=0).bind(stop=["\nObservation:", "Observation:"])
    scratchpad = ""
    history = []

    for step in range(1, MAX_STEPS + 1):
        messages = [SystemMessage(SYSTEM_PROMPT),
                    HumanMessage(f"Question: {question}\n\n{scratchpad}".rstrip())]
        reply = llm.invoke(messages).content.strip()
        thought, tool_name, tool_input, final_answer = parse_reply(reply)
        print(f"Thought {step}: {thought}")

        if final_answer is not None:
            print(f"Final Answer: {final_answer}")
            return final_answer, history

        previous = next((obs for name, arg, obs in history
                         if name == tool_name and arg == tool_input), None)
        if tool_name is None:
            observation = FORMAT_REMINDER
        elif tool_name not in TOOLS:
            observation = f"Error: unknown tool {tool_name!r}. Use one of {list(TOOLS)}."
        elif previous is not None:
            # Don't let the agent loop on calls it has already made
            print(f'Action {step}: {tool_name}("{tool_input}")')
            observation = (f"You already called {tool_name} with this input and got: "
                           f"{previous} Use the observations you have and give the Final Answer.")
        else:
            print(f'Action {step}: {tool_name}("{tool_input}")')
            observation = TOOLS[tool_name].invoke(tool_input)
            history.append((tool_name, tool_input, observation))
        print(f"Observation {step}: {observation}")
        scratchpad += (f"Thought: {thought}\nAction: {tool_name}\nAction Input: {tool_input}\n"
                       f"Observation: {observation}\n")

    print(f"Final Answer: (no answer after {MAX_STEPS} steps)")
    return None, history


if __name__ == "__main__":
    print(f"Model: {CHAT_MODEL}")
    print(f"Question: {QUESTION}\n")
    answer, history = run_agent(QUESTION)

    # Check the trace against the success criteria
    tools_used = [name for name, _, _ in history]
    first_internal = bool(tools_used) and tools_used[0] == "get_internal_stock_price"
    saw_error = any(name == "get_internal_stock_price" and obs.startswith("Error")
                    for name, _, obs in history)
    recovered = ("search_public_web" in tools_used and "get_internal_stock_price" in tools_used
                 and tools_used.index("search_public_web")
                 > tools_used.index("get_internal_stock_price"))
    answered = answer is not None and "170" in answer

    print("\nResilience checklist:")
    for label, ok in [("Tried the internal database first", first_internal),
                      ("Observed the timeout error", saw_error),
                      ("Switched to search_public_web afterwards", recovered),
                      ("Answered with the backup's price ($170)", answered)]:
        print(f"  [{'x' if ok else ' '}] {label}")
