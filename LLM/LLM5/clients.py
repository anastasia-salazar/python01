"""API clients: the only code that talks to the LLM servers over HTTP."""

import sys
import time

import requests

MAX_API_ATTEMPTS = 3
REQUEST_TIMEOUT = 300  # seconds; the API guide recommends 120-300 for long generations


class LLMAPIError(Exception):
    """The API call itself failed (network error, bad status, bad payload)."""


class OllamaChatClient:
    """Client for one model on an Ollama-compatible /api/chat endpoint."""

    def __init__(self, config, logger):
        self.name = config.name
        self.model = config.model
        self.url = config.url
        self.auth = (config.username, config.password)  # HTTP Basic Auth
        self.logger = logger

    def __repr__(self):
        return f"{self.name} ({self.model})"

    def service_is_up(self):
        """Hit the /health endpoint (no auth needed) to see if the service is running."""
        try:
            return requests.get(f"{self.url}/health", timeout=10).status_code == 200
        except requests.RequestException:
            return False

    def chat(self, messages, step):
        """Send messages and return the raw reply text.

        Network errors, rate limits (429) and server errors (5xx) are retried
        with increasing waits. Every prompt and raw output is logged.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",  # Ollama constrains the output to valid JSON
            "options": {"temperature": 0.4, "num_ctx": 8192},
        }
        self.logger.log("prompt", step=step, model=self.name, model_id=self.model,
                        messages=messages)

        problem = None
        for attempt in range(1, MAX_API_ATTEMPTS + 1):
            try:
                response = requests.post(f"{self.url}/api/chat", json=payload, auth=self.auth,
                                         timeout=REQUEST_TIMEOUT)
            except requests.RequestException as error:
                problem = f"network error: {error}"
            else:
                if response.status_code == 200:
                    try:
                        content = response.json()["message"]["content"]
                    except (ValueError, KeyError, TypeError):
                        self.logger.log("api_error", step=step, model=self.name,
                                        error="unexpected response format", body=response.text)
                        raise LLMAPIError(f"{self}: unexpected API response format: "
                                          f"{response.text[:300]}")
                    self.logger.log("raw_output", step=step, model=self.name, attempt=attempt,
                                    content=content)
                    return content
                if response.status_code == 401:
                    self.logger.log("api_error", step=step, model=self.name, status=401)
                    raise LLMAPIError(f"{self}: API returned 401 - username or password is wrong")
                if response.status_code != 429 and response.status_code < 500:
                    # 400/403/404 (e.g. unknown model) won't fix themselves, so don't retry
                    self.logger.log("api_error", step=step, model=self.name,
                                    status=response.status_code, body=response.text)
                    raise LLMAPIError(f"{self}: API returned {response.status_code}: "
                                      f"{response.text[:300]}")
                problem = f"API returned {response.status_code}"

            self.logger.log("api_error", step=step, model=self.name, attempt=attempt,
                            error=problem)
            if attempt < MAX_API_ATTEMPTS:
                wait = 2 ** attempt
                print(f"  {self}: {problem}; retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)

        status = "is up" if self.service_is_up() else "appears DOWN (or you're off the VPN)"
        raise LLMAPIError(f"{self}: API call failed after {MAX_API_ATTEMPTS} attempts "
                          f"({problem}). Health check: service {status}.")


def create_model_a_client(config, logger):
    return OllamaChatClient(config, logger)


def create_model_b_client(config, logger):
    return OllamaChatClient(config, logger)
