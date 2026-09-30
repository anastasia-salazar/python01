"""Assignment 1: The "Messy Data" Cleaner (PromptTemplates & Parsers).

Turns an emotional, rambling product review into one clean line:
    Sentiment: [Positive/Negative], Core Issue: [brief summary]
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from llm_setup import CHAT_MODEL, get_llm

TEST_REVIEW = ("I bought this blender yesterday and it's absolutely terrible! The lid flew off "
               "while I was making a smoothie and my whole kitchen is covered in spinach. "
               "I want a refund!")

# 1. A PromptTemplate with the {messy_review} variable
review_prompt = PromptTemplate.from_template(
    """You extract the core information from customer product reviews.

Read the review below and respond with ONLY one line in exactly this format:
Sentiment: [Positive/Negative], Core Issue: [Brief summary of the problem]

Rules:
- Sentiment must be exactly "Positive" or "Negative".
- Core Issue is a brief summary of at most 12 words.
- No other text, no quotes, no explanation.

Review: {messy_review}"""
)


def build_review_chain():
    """2-3. Prompt -> LLM wrapper -> StrOutputParser, joined with LCEL pipes."""
    return review_prompt | get_llm(temperature=0) | StrOutputParser()


if __name__ == "__main__":
    chain = build_review_chain()
    print(f"Model: {CHAT_MODEL}")
    print(f"\nMessy review:\n{TEST_REVIEW}\n")
    result = chain.invoke({"messy_review": TEST_REVIEW})
    print(f"Cleaned output:\n{result.strip()}")
