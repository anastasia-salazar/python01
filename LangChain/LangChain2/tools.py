"""The agent's two custom tools: a calculator and a Wikipedia search.

Each tool is deliberately limited, as the assignment requires:
  - CalculatorTool only does arithmetic on numbers; it cannot look anything up.
  - SearchTool only retrieves facts; it does no math.
"""

import ast
import operator
import re
import time

import requests
from langchain_core.tools import tool

# ---------------------------------------------------------------
# Calculator: + - * / on numbers only
# ---------------------------------------------------------------
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub,
             ast.Mult: operator.mul, ast.Div: operator.truediv}


def _evaluate(node):
    """Safely evaluate a parsed arithmetic expression (no eval(), no names, no functions)."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate(node.operand)
    raise ValueError("unsupported expression")


@tool("CalculatorTool")
def calculator_tool(expression: str) -> str:
    """Does arithmetic (add, subtract, multiply, divide) on numbers, e.g. "1879 * 5".
    Only accepts digits and + - * / ( ). It CANNOT look up facts, names or dates."""
    expression = expression.strip().strip("\"'`")
    if re.search(r"[A-Za-z]", expression):
        return ("Error: CalculatorTool only accepts numbers and + - * /. It cannot look up "
                f"facts like {expression!r}. Find the missing number with SearchTool first.")
    try:
        result = _evaluate(ast.parse(expression, mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero."
    except (SyntaxError, ValueError):
        return f"Error: {expression!r} is not a valid arithmetic expression."
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# ---------------------------------------------------------------
# Search: Wikipedia
# ---------------------------------------------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "LangChainCourseAgent/1.0 (student project)"}
MAX_CHARS = 400


def _wiki(params):
    """Call the Wikipedia API, waiting and retrying if it rate-limits us (HTTP 429)."""
    for attempt in range(4):
        response = requests.get(WIKI_API, params={"action": "query", "format": "json", **params},
                                headers=HEADERS, timeout=20)
        if response.status_code != 429:
            break
        wait = int(response.headers.get("Retry-After", 0)) or 2 ** (attempt + 1)
        time.sleep(min(wait, 30))
    response.raise_for_status()
    return response.json()["query"]


def _title_words(title):
    return re.findall(r"[a-z0-9]+", re.sub(r"\s*\(.*\)", "", title).lower())


def _choose_article(query):
    """Pick the Wikipedia article that best matches the query.

    Wikipedia's search ranks loosely: for "Albert Einstein birth year" it returns
    "Hans Albert Einstein" (his son, born 1904) first. So candidates come from the search
    results plus names in the query (capitalized phrases), and a candidate only counts if
    every word of its title appears in the query. The most specific match wins.
    """
    hits = [hit["title"] for hit in _wiki({"list": "search", "srsearch": query, "srlimit": 5})["search"]]
    names = re.findall(r"[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*", query)
    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))

    best, best_score = None, 0
    for title in hits + names:
        words = _title_words(title)
        if words and set(words) <= query_words and len(words) > best_score:
            best, best_score = title, len(words)
    return best or (hits[0] if hits else None)


@tool("SearchTool")
def search_tool(query: str) -> str:
    """Looks up facts on Wikipedia, e.g. "Albert Einstein birth year".
    Returns the opening of the most relevant article. It CANNOT do math."""
    query = query.strip().strip("\"'`")
    try:
        title = _choose_article(query)
        if not title:
            return f"No Wikipedia results for {query!r}."
        pages = _wiki({"prop": "extracts", "exintro": 1, "explaintext": 1, "redirects": 1,
                       "titles": title})["pages"]
        page = next(iter(pages.values()))
        if "missing" in page or not page.get("extract"):
            return f"No Wikipedia article found for {query!r}."
    except requests.RequestException as error:
        return f"Error: Wikipedia search failed ({error})."
    text = " ".join(page["extract"].split())
    return f"{page['title']}: {text[:MAX_CHARS]}"


TOOLS = {tool_.name: tool_ for tool_ in (calculator_tool, search_tool)}
