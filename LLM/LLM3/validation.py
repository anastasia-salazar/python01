"""Validation: required structure for each step, topic relevance, and the final output."""

import re

from parsing import ResponseError, parse_json_response
from prompts import build_relevance_prompt

# Required fields for each step.
#   ("text", max_words)          -> non-empty string with a word limit
#   ("list", min_items, max_items) -> list of non-empty strings
SCHEMAS = {
    "initial": {
        "position": ("text", 60),
        "explanation": ("text", 150),
        "key_arguments": ("list", 3, 5),
    },
    "critique": {
        "assessment": ("text", 60),
        "critique": ("text", 150),
        "counterpoints": ("list", 2, 5),
        "questions": ("list", 1, 3),
    },
    "reply": {
        "response_to_critique": ("text", 150),
        "concessions": ("list", 0, 3),
        "final_position": ("text", 60),
    },
    "conclusion": {
        "conclusion": ("text", 100),
    },
}

FINAL_OUTPUT_FIELDS = ["topic", "models", "model_a_initial_response", "model_b_critique",
                       "model_a_final_reply", "conclusion"]

# Small words that don't tell us what a topic is about
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "should", "that",
    "the", "their", "there", "this", "to", "was", "we", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your", "more", "than", "vs", "versus",
}


def validate_structure(data, step):
    """Check that the response has exactly the required fields, types and limits."""
    schema = SCHEMAS[step]
    errors = []

    missing = schema.keys() - data.keys()
    extra = data.keys() - schema.keys()
    if missing:
        errors.append(f"missing field(s): {sorted(missing)}")
    if extra:
        errors.append(f"unexpected field(s): {sorted(extra)}")

    for field, rule in schema.items():
        if field not in data:
            continue
        value = data[field]
        if rule[0] == "text":
            max_words = rule[1]
            if not isinstance(value, str) or not value.strip():
                errors.append(f"'{field}' must be a non-empty string")
            elif len(value.split()) > max_words:
                errors.append(f"'{field}' has {len(value.split())} words (max {max_words})")
        else:
            min_items, max_items = rule[1], rule[2]
            if not isinstance(value, list) or not all(isinstance(item, str) and item.strip()
                                                      for item in value):
                errors.append(f"'{field}' must be an array of non-empty strings")
            elif not min_items <= len(value) <= max_items:
                errors.append(f"'{field}' has {len(value)} items "
                              f"(must be {min_items}-{max_items})")

    if errors:
        raise ResponseError("; ".join(errors))
    return data


def topic_keywords(topic):
    words = re.findall(r"[a-z0-9]+", topic.lower())
    return {word for word in words if word not in STOPWORDS and len(word) > 2}


def response_text(data):
    """All the text in a response, joined into one lowercase string."""
    parts = []
    for value in data.values():
        parts.extend(value if isinstance(value, list) else [str(value)])
    return " ".join(parts).lower()


def mentions_topic(topic, data):
    """Cheap check: does the response use any of the topic's key words?

    Compares word stems, so "automation" also matches "automated" and "automating".
    """
    text = response_text(data)
    for keyword in topic_keywords(topic):
        stem = keyword[:max(4, len(keyword) - 3)]
        if stem in text:
            return True
    return False


def check_relevance(topic, data, judge_client, step):
    """Make sure the response is about the topic.

    First a quick keyword check; if the response never uses the topic's key
    words, a model is asked to judge relevance (it may use synonyms).
    Returns a short description of how relevance was confirmed.
    """
    if not topic_keywords(topic) or mentions_topic(topic, data):
        judge_client.logger.log("relevance_check", step=step, method="keywords", relevant=True)
        return "keywords"

    raw = judge_client.chat(build_relevance_prompt(topic, data), step=f"{step}:relevance")
    verdict = parse_json_response(raw)
    relevant = verdict.get("relevant")
    reason = verdict.get("reason", "")
    judge_client.logger.log("relevance_check", step=step, method=f"judge ({judge_client})",
                            relevant=relevant, reason=reason)
    if relevant is not True:
        raise ResponseError(f"response is not relevant to the topic '{topic}' ({reason})")
    return "judge"


def validate_final_output(result):
    """Check the final discussion object before it is returned."""
    missing = [field for field in FINAL_OUTPUT_FIELDS if field not in result]
    if missing:
        raise ResponseError(f"final output missing field(s): {missing}")
    if not isinstance(result["topic"], str) or not result["topic"].strip():
        raise ResponseError("final output 'topic' must be a non-empty string")
    validate_structure(result["model_a_initial_response"], "initial")
    validate_structure(result["model_b_critique"], "critique")
    validate_structure(result["model_a_final_reply"], "reply")
    validate_structure({"conclusion": result["conclusion"]}, "conclusion")
    return result
