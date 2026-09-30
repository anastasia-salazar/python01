"""Assignment 1: A Self-Correcting LangChain Agent.

A ReAct-style agent (Thought -> Action -> Observation, repeated) with two limited tools:
CalculatorTool (math only) and SearchTool (Wikipedia facts only). The loop prints every
step of the agent's reasoning.

Self-correction happens in three ways:
  - If the agent sends words to CalculatorTool (e.g. "Einstein's birth year * 5"), the tool
    returns an error telling it to search first.
  - If the agent sends a number to CalculatorTool that it never looked up (e.g. a birth
    year from its own memory), the agent loop rejects it: facts must come from SearchTool.
  - If the agent's reply is badly formatted, it is told the correct format and retries.

Usage:
    python assignment1_self_correcting_agent.py
    python assignment1_self_correcting_agent.py "Divide the birth year of Marie Curie by 3"
"""

import re
import sys

from langchain_core.messages import HumanMessage, SystemMessage

from llm_setup import CHAT_MODEL, get_llm
from tools import TOOLS

CHALLENGE_PROMPT = "Multiply the birth year of Albert Einstein by 5."
EXPECTED_ANSWER = "9395"
MAX_STEPS = 8

SYSTEM_PROMPT = f"""You are a careful problem-solving agent. You answer questions by reasoning
step by step and using tools. You have these tools:

{chr(10).join(f"- {name}: {tool_.description}" for name, tool_ in TOOLS.items())}

Rules:
- Never rely on your own memory for facts such as dates or numbers. Every fact must come
  from a SearchTool observation.
- CalculatorTool only accepts numbers, e.g. "1879 * 5". Look up missing numbers first.
- Use one tool per step, then wait for its Observation.

Reply in EXACTLY this format, and stop after "Action Input":
Thought: <your reasoning about what to do next>
Action: <SearchTool or CalculatorTool>
Action Input: <the input for the tool>

When you have the final result, reply instead with:
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
    if action:  # an action takes priority: the agent must act before it can finish
        name = action.group(1).strip().strip("[]\"'`")
        tool_input = action.group(2).strip().splitlines()[0].strip().strip("\"'`")
        return thought, name, tool_input, None
    if final:
        return thought, None, None, final.group(1).strip()
    return thought, None, None, None


def unverified_numbers(expression, known_text):
    """Numbers in a calculation that appear neither in the question nor in any observation."""
    known = set(re.findall(r"\d+(?:\.\d+)?", known_text))
    return [n for n in re.findall(r"\d+(?:\.\d+)?", expression) if n not in known]


def run_agent(question):
    llm = get_llm(temperature=0).bind(stop=["\nObservation:", "Observation:"])
    scratchpad = ""          # the agent's history, shown to it on every step
    known_text = question    # everything the agent is allowed to take numbers from

    for step in range(1, MAX_STEPS + 1):
        messages = [SystemMessage(SYSTEM_PROMPT),
                    HumanMessage(f"Question: {question}\n\n{scratchpad}".rstrip())]
        reply = llm.invoke(messages).content.strip()
        thought, tool_name, tool_input, final_answer = parse_reply(reply)
        print(f"Thought {step}: {thought}")

        if final_answer is not None:
            print(f"Final Answer: [{final_answer}]")
            return final_answer

        if tool_name is None:
            observation = FORMAT_REMINDER
            print(f"Observation {step}: [{observation}]")
            scratchpad += f"{reply}\nObservation: {observation}\n"
            continue

        print(f'Action {step}: [{tool_name}: "{tool_input}"]')

        if tool_name not in TOOLS:
            observation = f"Error: unknown tool {tool_name!r}. Use one of {list(TOOLS)}."
        elif tool_name == "CalculatorTool" and unverified_numbers(tool_input, known_text):
            missing = unverified_numbers(tool_input, known_text)
            observation = (f"Error: the number(s) {missing} were not given in the question or "
                           "found with SearchTool. Do not use facts from memory; look them up "
                           "with SearchTool first.")
        else:
            observation = TOOLS[tool_name].invoke(tool_input)
            known_text += " " + observation

        shown = observation if len(observation) <= 220 else observation[:220] + "..."
        print(f"Observation {step}: [{shown}]")
        scratchpad += (f"Thought: {thought}\nAction: {tool_name}\nAction Input: {tool_input}\n"
                       f"Observation: {observation}\n")

    print(f"Final Answer: [No answer after {MAX_STEPS} steps]")
    return None


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or CHALLENGE_PROMPT
    print(f"Model: {CHAT_MODEL}")
    print(f"Question: {question}\n")
    answer = run_agent(question)

    if question == CHALLENGE_PROMPT:
        correct = answer is not None and EXPECTED_ANSWER in answer.replace(",", "")
        print(f"\nExpected {EXPECTED_ANSWER}: {'correct' if correct else 'INCORRECT'}")
