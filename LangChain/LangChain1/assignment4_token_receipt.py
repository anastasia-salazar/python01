"""Assignment 4: The Watchful Eye (Callbacks & Logging).

Runs the Assignment 1 review-cleaner chain with a custom LangChain callback handler that
counts the tokens used, then prints a receipt.

get_openai_callback only works with OpenAI models, so this uses a handler written for any
chat model: after each LLM call it reads the token counts the model reports
(Ollama reports prompt and completion tokens for every call).

Cost: the company Ollama server is self-hosted, so there is no per-token charge and the
cost is $0.00. To estimate what the same usage would cost on a paid API, set
PROMPT_PRICE_PER_1M and COMPLETION_PRICE_PER_1M (US dollars per million tokens) in .env.
"""

import os

from langchain_core.callbacks import BaseCallbackHandler

from assignment1_review_cleaner import TEST_REVIEW, build_review_chain
from llm_setup import CHAT_MODEL

PROMPT_PRICE_PER_1M = float(os.getenv("PROMPT_PRICE_PER_1M", "0"))
COMPLETION_PRICE_PER_1M = float(os.getenv("COMPLETION_PRICE_PER_1M", "0"))


class TokenUsageHandler(BaseCallbackHandler):
    """Adds up the tokens reported by every LLM call made while it is attached."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_calls = 0

    def on_llm_end(self, response, **kwargs):
        # Called by LangChain after each LLM call finishes
        self.llm_calls += 1
        for generation_list in response.generations:
            for generation in generation_list:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    self.prompt_tokens += usage.get("input_tokens", 0)
                    self.completion_tokens += usage.get("output_tokens", 0)

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    @property
    def total_cost(self):
        return (self.prompt_tokens * PROMPT_PRICE_PER_1M
                + self.completion_tokens * COMPLETION_PRICE_PER_1M) / 1_000_000


def print_receipt(handler):
    width = 44
    pricing = ("self-hosted, no per-token charge"
               if PROMPT_PRICE_PER_1M == COMPLETION_PRICE_PER_1M == 0
               else f"${PROMPT_PRICE_PER_1M}/${COMPLETION_PRICE_PER_1M} per 1M tokens")
    lines = [
        ("Model", CHAT_MODEL),
        ("LLM calls", f"{handler.llm_calls}"),
        ("Prompt Tokens", f"{handler.prompt_tokens:,}"),
        ("Completion Tokens", f"{handler.completion_tokens:,}"),
    ]
    print("\n+" + "-" * width + "+")
    print("|" + "TOKEN USAGE RECEIPT".center(width) + "|")
    print("+" + "-" * width + "+")
    for label, value in lines:
        print(f"| {label:<20}{value:>{width - 22}} |")
    print("|" + " " * 22 + "-" * (width - 24) + "  |")
    print(f"| {'Total Tokens Used':<20}{handler.total_tokens:>{width - 22},} |")
    print(f"| {'Total Cost':<20}{f'${handler.total_cost:.6f}':>{width - 22}} |")
    print("+" + "-" * width + "+")
    print(f"  Pricing: {pricing}")


if __name__ == "__main__":
    chain = build_review_chain()
    handler = TokenUsageHandler()

    # Attach the callback for this run only
    result = chain.invoke({"messy_review": TEST_REVIEW}, config={"callbacks": [handler]})

    print(f"Cleaned output:\n{result.strip()}")
    print_receipt(handler)
