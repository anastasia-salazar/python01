"""Shared model setup for all four assignments.

Uses Ollama, per the course note about corporate blocks on OpenAI/Gemini keys. By default it
connects to the company-hosted Ollama server; to use Ollama installed on your own machine
instead, set LLM_API_URL=http://localhost:11434 in .env and leave the credentials empty.

Settings are loaded from a .env file with python-dotenv - no keys are hardcoded here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv(Path(__file__).resolve().parent / ".env")

COMPANY_URL = "https://aimodels.jadeglobal.com:8082"
API_URL = os.getenv("LLM_API_URL", COMPANY_URL)
CHAT_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


def _client_kwargs():
    """HTTP Basic Auth for the company server; nothing for a local Ollama."""
    username, password = os.getenv("LLM_USERNAME"), os.getenv("LLM_PASSWORD")
    if username and password:
        return {"auth": (username, password), "timeout": 300}
    if API_URL.rstrip("/") == COMPANY_URL:
        raise SystemExit("Missing LLM_USERNAME / LLM_PASSWORD. Copy .env.example to .env "
                         "and fill them in (or point LLM_API_URL at a local Ollama).")
    return {"timeout": 300}


def get_llm(temperature=0.0):
    """The chat model wrapper (LangChain's ChatOllama)."""
    return ChatOllama(model=CHAT_MODEL, base_url=API_URL, temperature=temperature,
                      client_kwargs=_client_kwargs())


def get_embeddings():
    """The embedding model used to turn text chunks into vectors."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=API_URL,
                            client_kwargs=_client_kwargs())
