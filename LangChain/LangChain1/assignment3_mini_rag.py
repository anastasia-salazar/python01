"""Assignment 3: Mini-RAG (Retrieval-Augmented Generation).

Loads the confidential board game rules in game_rules.txt, splits them into chunks, embeds
the chunks into an in-memory FAISS vector store, retrieves the chunks relevant to a question,
and has the LLM answer using only those chunks.

Usage:
    python assignment3_mini_rag.py
    python assignment3_mini_rag.py "What happens on a volcano tile?"
"""

import sys
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_setup import CHAT_MODEL, EMBEDDING_MODEL, get_embeddings, get_llm

RULES_FILE = Path(__file__).resolve().parent / "game_rules.txt"
DEFAULT_QUESTION = "How many points is the golden token worth?"

# 1. Document Loader: read the text file into LangChain Document objects
documents = TextLoader(str(RULES_FILE), encoding="utf-8").load()

# 2. Text Splitter: small chunks so each holds roughly one rule
splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
chunks = splitter.split_documents(documents)

# 3. Embeddings + in-memory vector store
vector_store = FAISS.from_documents(chunks, get_embeddings())
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 4. RAG chain: retrieve chunks -> put them in the prompt -> LLM -> text
rag_prompt = PromptTemplate.from_template(
    """Answer the question using ONLY the game rules below. If the rules don't contain the
answer, say "The rules don't say."

Game rules:
{context}

Question: {question}
Answer in one sentence:"""
)


def format_chunks(docs):
    return "\n".join(doc.page_content for doc in docs)


llm = get_llm(temperature=0)
rag_chain = (
    {"context": retriever | format_chunks, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    print(f"Chat model: {CHAT_MODEL}   Embedding model: {EMBEDDING_MODEL}")
    print(f"Loaded {len(documents)} document, split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, start=1):
        print(f"  [{i}] {chunk.page_content}")

    print(f"\nQuestion: {question}")

    # Without retrieval, the model has never seen these made-up rules
    no_context = (PromptTemplate.from_template("Answer in one sentence: {question}")
                  | llm | StrOutputParser()).invoke({"question": question})
    print(f"\nWithout RAG (model's own knowledge):\n  {no_context.strip()}")

    print("\nRetrieved chunks (lower score = closer match):")
    for doc, score in vector_store.similarity_search_with_score(question, k=2):
        print(f"  score {score:.3f}: {doc.page_content}")

    answer = rag_chain.invoke(question)
    print(f"\nWith RAG (answer from the retrieved rules):\n  {answer.strip()}")
