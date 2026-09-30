"""Configuration: reads credentials and model settings from the environment / .env."""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_API_URL = "https://aimodels.jadeglobal.com:8082"
DEFAULT_MODEL_A = "llama3.1:8b"
DEFAULT_MODEL_B = "qwen3:14b"


class ConfigError(Exception):
    """Required settings are missing."""


@dataclass
class ModelConfig:
    name: str      # "Model A" or "Model B"
    url: str       # base URL of the API serving this model
    model: str     # model name on that server
    username: str
    password: str


def load_env_file(path):
    """Read KEY=VALUE lines from a .env file into os.environ (real env vars win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_model_configs():
    """Return (Model A config, Model B config).

    Each model can point at its own API (MODEL_A_URL / MODEL_B_URL); both
    default to the shared company endpoint in LLM_API_URL.
    """
    load_env_file(Path(__file__).resolve().parent / ".env")

    username = os.environ.get("LLM_USERNAME")
    password = os.environ.get("LLM_PASSWORD")
    missing = [var for var, value in [("LLM_USERNAME", username), ("LLM_PASSWORD", password)]
               if not value]
    if missing:
        raise ConfigError(f"Missing credential(s): {', '.join(missing)}. "
                          "Set them as environment variables or in a .env file.")

    base_url = os.environ.get("LLM_API_URL", DEFAULT_API_URL)
    model_a = ModelConfig(
        name="Model A",
        url=os.environ.get("MODEL_A_URL", base_url).rstrip("/"),
        model=os.environ.get("MODEL_A", DEFAULT_MODEL_A),
        username=username,
        password=password,
    )
    model_b = ModelConfig(
        name="Model B",
        url=os.environ.get("MODEL_B_URL", base_url).rstrip("/"),
        model=os.environ.get("MODEL_B", DEFAULT_MODEL_B),
        username=username,
        password=password,
    )
    return model_a, model_b
