"""Assignment 3: Solving "Context Poisoning" with Metadata Filtering.

Two contradictory policy documents are stored in a FAISS vector store, each tagged with the
year it was issued. A plain semantic search returns both (the outdated 2022 policy can
"poison" the answer). The custom retrieval function filters on the year metadata, so only
the 2024 policy ever reaches the LLM.

Usage:
    python assignment3_context_poisoning.py              # filter to 2024
    python assignment3_context_poisoning.py 2022         # try the old policy instead
"""

import sys
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm_setup import CHAT_MODEL, EMBEDDING_MODEL, get_embeddings, get_llm

POLICY_DIR = Path(__file__).resolve().parent / "policies"
POLICY_FILES = {2022: "wfh_policy_2022.txt", 2024: "wfh_policy_2024.txt"}
USER_QUERY = "What is the WFH policy?"


def load_policies():
    """Load each policy file and attach its year as metadata."""
    documents = []
    for year, filename in POLICY_FILES.items():
        for doc in TextLoader(str(POLICY_DIR / filename), encoding="utf-8").load():
            doc.metadata["year"] = year
            documents.append(doc)
    return documents


vector_store = FAISS.from_documents(load_policies(), get_embeddings())


def retrieve(user_query, filter_year, k=1):
    """Semantic search restricted to documents whose metadata year equals filter_year."""
    results = vector_store.similarity_search(user_query, k=k, filter={"year": filter_year})
    # Defense in depth: never pass on a document from another year, even if the store did
    return [doc for doc in results if doc.metadata.get("year") == filter_year]


answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You answer questions about company policy using ONLY the policy text "
               "provided. Do not mention or assume any other version of the policy."),
    ("human", "Policy ({year}):\n{context}\n\nQuestion: {question}"),
])
answer_chain = answer_prompt | get_llm(temperature=0) | StrOutputParser()


def answer(user_query, filter_year):
    docs = retrieve(user_query, filter_year)
    if not docs:
        return docs, f"No policy found for {filter_year}."
    context = "\n".join(doc.page_content for doc in docs)
    return docs, answer_chain.invoke({"year": filter_year, "context": context,
                                      "question": user_query}).strip()


if __name__ == "__main__":
    filter_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    print(f"Chat model: {CHAT_MODEL}   Embedding model: {EMBEDDING_MODEL}\n")

    # The problem: without a filter, both contradictory policies come back
    print("Without a filter (the context-poisoning problem):")
    for doc in vector_store.similarity_search(USER_QUERY, k=2):
        print(f"  [{doc.metadata['year']}] {doc.page_content.strip()}")

    # The fix: filter on metadata
    docs, final_answer = answer(USER_QUERY, filter_year)
    print("\n" + "=" * 70)
    print(f'User Query: "{USER_QUERY}"')
    print(f"Active Filter: Year: {filter_year}")
    print("Retrieved Context:")
    for doc in docs:
        print(f"  [{doc.metadata['year']}] {Path(doc.metadata['source']).name}: "
              f"{doc.page_content.strip()}")
    excluded = sorted(year for year in POLICY_FILES if year != filter_year)
    print(f"  Excluded by filter: {', '.join(f'{y} policy' for y in excluded)} "
          f"(retrieved docs from other years: "
          f"{sum(doc.metadata['year'] != filter_year for doc in docs)})")
    print(f"LLM Final Answer: {final_answer}")
    print("=" * 70)
