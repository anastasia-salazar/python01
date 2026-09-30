"""Assignment 4: The "Fast & Grounded" System.

Grounding: a strict system prompt makes the LLM answer only from the provided context, and
reply exactly "I do not have enough information" when the answer isn't there.

Latency: answers are cached in a plain dictionary (query_cache). Before calling the LLM,
the exact question is looked up in the cache; a hit returns immediately with no API call.
"""

import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from llm_setup import CHAT_MODEL, get_llm

CONTEXT = (Path(__file__).resolve().parent / "knowledge_base.txt").read_text(encoding="utf-8")
NO_INFO = "I do not have enough information"

SYSTEM_PROMPT = f"""You are a strictly grounded assistant. You answer ONLY from the CONTEXT
below, which is the only information you have.

Rules:
1. If the CONTEXT contains the answer, answer concisely using only facts from it.
2. If the CONTEXT does not contain the answer, reply with exactly this sentence and nothing
   else: {NO_INFO}
3. Never use outside knowledge, never guess, never add explanations to rule 2's reply.

CONTEXT:
{CONTEXT}"""

query_cache = {}
llm = get_llm(temperature=0)
llm_calls = 0


def enforce_exact_refusal(answer):
    """Normalize small variations like a trailing period or quotes to the exact phrase.

    Only cosmetic differences are normalized; any extra words are left visible, so a
    grounding failure is never hidden.
    """
    cleaned = answer.strip().strip("\"'").rstrip(".!").strip()
    return NO_INFO if cleaned.lower() == NO_INFO.lower() else answer.strip()


def ask(question):
    """Return (answer, came_from_cache). The exact question string is the cache key."""
    global llm_calls
    if question in query_cache:
        return query_cache[question], True

    llm_calls += 1
    reply = llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]).content
    answer = enforce_exact_refusal(reply)
    query_cache[question] = answer
    return answer, False


def run_scenario(title, question):
    print(f"\n{title}")
    print(f"  Question: {question}")
    start = time.perf_counter()
    answer, from_cache = ask(question)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if from_cache:
        print(f"  Returned from Cache: {answer}")
    else:
        print(f"  LLM Response: {answer}")
    print(f"  Time: {elapsed_ms:,.1f} ms   (LLM calls so far: {llm_calls}, "
          f"cache size: {len(query_cache)})")
    return answer


if __name__ == "__main__":
    print(f"Model: {CHAT_MODEL}")
    valid_question = "How many days of paid vacation do full-time employees get per year?"

    run_scenario("Scenario 1 (First Ask):", valid_question)
    run_scenario("Scenario 2 (Cache Hit):", valid_question)
    grounding = run_scenario("Scenario 3 (Grounding Test):",
                             "What is the recipe for a chocolate cake?")

    print(f'\nGrounding check - response is exactly "{NO_INFO}": {grounding == NO_INFO}')
