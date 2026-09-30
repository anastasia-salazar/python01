"""Validation: required structure for each step, and relevance to the scenario."""

import re
from collections import Counter

from parsing import ResponseError, parse_json_response
from prompts import build_relevance_prompt


# ---------------------------------------------------------------
# Field rules
# ---------------------------------------------------------------
class Text:
    """A non-empty string with a word limit."""

    def __init__(self, max_words):
        self.max_words = max_words

    def check(self, value, path):
        if not isinstance(value, str) or not value.strip():
            return [f"'{path}' must be a non-empty string"]
        if len(value.split()) > self.max_words:
            return [f"'{path}' has {len(value.split())} words (max {self.max_words})"]
        return []


class TextList:
    """A list of non-empty strings with a size range."""

    def __init__(self, min_items, max_items):
        self.min_items, self.max_items = min_items, max_items

    def check(self, value, path):
        if not isinstance(value, list) or not all(isinstance(item, str) and item.strip()
                                                  for item in value):
            return [f"'{path}' must be an array of non-empty strings"]
        if not self.min_items <= len(value) <= self.max_items:
            return [f"'{path}' has {len(value)} items (must be {self.min_items}-{self.max_items})"]
        return []


class Choice:
    """One of a fixed set of strings."""

    def __init__(self, *options):
        self.options = options

    def check(self, value, path):
        if value not in self.options:
            return [f"'{path}' must be one of {list(self.options)}, got {value!r}"]
        return []


class Integer:
    """A whole number in a range."""

    def __init__(self, low, high):
        self.low, self.high = low, high

    def check(self, value, path):
        # bool is a subclass of int in Python, so rule it out explicitly
        if not isinstance(value, int) or isinstance(value, bool):
            return [f"'{path}' must be an integer"]
        if not self.low <= value <= self.high:
            return [f"'{path}' must be between {self.low} and {self.high}, got {value}"]
        return []


class ObjectList:
    """A list of objects, each with exactly the given fields."""

    def __init__(self, min_items, max_items, fields):
        self.min_items, self.max_items, self.fields = min_items, max_items, fields

    def check(self, value, path):
        if not isinstance(value, list):
            return [f"'{path}' must be an array of objects"]
        errors = []
        if not self.min_items <= len(value) <= self.max_items:
            errors.append(f"'{path}' has {len(value)} items "
                          f"(must be {self.min_items}-{self.max_items})")
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                errors.append(f"'{path}[{i}]' must be an object")
            else:
                errors += check_fields(item, self.fields, f"{path}[{i}]")
        return errors


SEVERITY = Choice("low", "medium", "high")

SCHEMAS = {
    "proposal": {
        "summary": Text(60),
        "proposal": Text(200),
        "reasoning": TextList(3, 6),
        "assumptions": TextList(1, 5),
    },
    "stress_test": {
        "overall_assessment": Text(60),
        "weaknesses": TextList(2, 6),
        "risks": ObjectList(2, 6, {"risk": Text(60), "severity": SEVERITY}),
        "edge_cases": TextList(1, 5),
        "counterarguments": TextList(1, 5),
        "questions": TextList(1, 3),
    },
    "revision": {
        "revised_proposal": Text(200),
        "changes_made": TextList(1, 6),
        "defended_points": TextList(0, 4),
        "mitigations": ObjectList(1, 6, {"risk": Text(60), "mitigation": Text(80)}),
        "answers_to_questions": TextList(1, 3),
        "remaining_limitations": TextList(0, 4),
    },
    "evaluation": {
        "robustness_score": Integer(1, 10),
        "verdict": Choice("robust", "needs_minor_changes", "needs_major_changes", "not_viable"),
        "summary": Text(100),
        "addressed_risks": TextList(0, 6),
        "remaining_risks": ObjectList(0, 5, {"risk": Text(60), "severity": SEVERITY}),
    },
}


def check_fields(data, fields, path=""):
    """Check an object has exactly the required fields, and each passes its rule."""
    errors = []
    prefix = f"{path}." if path else ""
    missing = fields.keys() - data.keys()
    extra = data.keys() - fields.keys()
    if missing:
        errors.append(f"{path or 'response'} is missing field(s): {sorted(missing)}")
    if extra:
        errors.append(f"{path or 'response'} has unexpected field(s): {sorted(extra)}")
    for name, rule in fields.items():
        if name in data:
            errors += rule.check(data[name], prefix + name)
    return errors


def validate_structure(data, step):
    """Raise ResponseError listing every problem if the response breaks its schema."""
    errors = check_fields(data, SCHEMAS[step])
    if errors:
        raise ResponseError("; ".join(errors))
    return data


# ---------------------------------------------------------------
# Relevance to the scenario
# ---------------------------------------------------------------
STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for from
further had has have having he her here hers him his how i if in into is it its itself just
me more most my no nor not now of off on once only or other our ours out over own same she
should so some such than that the their theirs them then there these they this those
through to too under until up very was we were what when where which while who whom why
will with would you your yours new one two three use using need needs want wants make like
get well many much may might must shall us company team plan current currently per year
years month months week weeks day days person people considering consider handle
handles going ahead expects expected within across
""".split())

TOP_KEYWORDS = 12        # how many of the scenario's main words to look for
MIN_KEYWORD_SHARE = 0.3  # a response must mention at least 30% of them


def scenario_keywords(scenario):
    """The scenario's most frequent meaningful words (its main subject matter)."""
    words = [w for w in re.findall(r"[a-z][a-z0-9-]+", scenario.lower())
             if w not in STOPWORDS and len(w) > 2]
    return [word for word, _ in Counter(words).most_common(TOP_KEYWORDS)]


def stem(word):
    """Crude stem, so "migrate" matches "migration" and "migrating"."""
    return word[:max(4, len(word) - 3)]


def response_text(data):
    """All text in a (possibly nested) response, as one lowercase string."""
    if isinstance(data, dict):
        return " ".join(response_text(v) for v in data.values())
    if isinstance(data, list):
        return " ".join(response_text(v) for v in data)
    return str(data).lower()


def keyword_coverage(scenario, data):
    """Return (share of scenario keywords the response mentions, the keywords)."""
    keywords = scenario_keywords(scenario)
    if not keywords:
        return 1.0, keywords
    text = response_text(data)
    found = sum(1 for word in keywords if stem(word) in text)
    return found / len(keywords), keywords


def check_relevance(scenario, data, judge_client, step):
    """Make sure the response addresses the scenario.

    First a quick keyword-coverage check. If the response mentions too few of the
    scenario's main words, another model judges relevance (it may use synonyms).
    """
    coverage, keywords = keyword_coverage(scenario, data)
    if coverage >= MIN_KEYWORD_SHARE:
        judge_client.logger.log("relevance_check", step=step, method="keywords",
                                coverage=round(coverage, 2), keywords=keywords, relevant=True)
        return

    raw = judge_client.chat(build_relevance_prompt(scenario, data), step=f"{step}:relevance")
    verdict = parse_json_response(raw)
    relevant = verdict.get("relevant")
    reason = verdict.get("reason", "")
    judge_client.logger.log("relevance_check", step=step, method=f"judge ({judge_client})",
                            coverage=round(coverage, 2), keywords=keywords,
                            relevant=relevant, reason=reason)
    if relevant is not True:
        raise ResponseError(f"response does not address the scenario ({reason})")
