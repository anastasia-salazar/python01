"""Assignment 2: The Marketing Assembly Line (Multi-step LCEL Chains).

Chain 1 writes a catchy 5-word English slogan for a product; Chain 2 translates that slogan
into French. Both are combined into one runnable sequence with the pipe operator (|).

Usage:
    python assignment2_marketing_chain.py
    python assignment2_marketing_chain.py "EcoCharge solar phone charger"
"""

import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from llm_setup import CHAT_MODEL, get_llm

DEFAULT_PRODUCT = "AquaPure, a self-cleaning smart water bottle"

llm = get_llm(temperature=0.7)  # a little creativity for slogans
parser = StrOutputParser()

# Chain 1: {product_name} -> 5-word English slogan
slogan_prompt = PromptTemplate.from_template(
    "Write a catchy marketing slogan for this product: {product_name}\n"
    "The slogan must be exactly 5 words, in English.\n"
    "Return only the slogan: no quotes, no explanation."
)
slogan_chain = slogan_prompt | llm | parser

# Chain 2: {slogan} -> French translation
translate_prompt = PromptTemplate.from_template(
    "Translate this marketing slogan into French, keeping it catchy:\n{slogan}\n"
    "Return only the French slogan: no quotes, no explanation."
)
translate_chain = translate_prompt | llm | parser

# Combined sequence: Chain 1's output becomes the {slogan} input of Chain 2.
# RunnablePassthrough.assign keeps the English slogan alongside the French one.
marketing_chain = (
    {"slogan": slogan_chain}
    | RunnablePassthrough.assign(french_slogan=translate_chain)
)


if __name__ == "__main__":
    product = " ".join(sys.argv[1:]) or DEFAULT_PRODUCT
    print(f"Model:   {CHAT_MODEL}")
    print(f"Product: {product}\n")

    result = marketing_chain.invoke({"product_name": product})

    print(f"English slogan (Chain 1): {result['slogan'].strip()}")
    print(f"French slogan  (Chain 2): {result['french_slogan'].strip()}")
