"""The five LLM agents. Each is a LangGraph node: it reads what it needs from the shared
state, makes one structured LLM call, and returns its result to be merged into the state."""

import json
import sys

from langchain_core.messages import HumanMessage, SystemMessage

from llm_setup import get_llm
from schemas import (ArchitectureOutput, DocumentAnalysis, RequirementsOutput, RiskOutput,
                     TestCaseOutput)

MAX_ATTEMPTS = 3  # retries if a reply fails schema validation or the API call errors


def ask_structured(schema, system_prompt, user_prompt):
    """One LLM call whose reply is constrained to, and validated against, `schema`."""
    llm = get_llm(temperature=0).with_structured_output(schema)
    llm = llm.with_retry(stop_after_attempt=MAX_ATTEMPTS)
    result = llm.invoke([SystemMessage(system_prompt), HumanMessage(user_prompt)])
    return result.model_dump()


def as_json(data):
    return json.dumps(data, indent=2, ensure_ascii=False)


def log(agent, message):
    print(f"  [{agent}] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------
# Document Analyzer
# ---------------------------------------------------------------
def document_analyzer(state):
    log("Document Analyzer", "reading the document...")
    result = ask_structured(
        DocumentAnalysis,
        "You are a business analyst who reviews software requirements documents (SRS). "
        "Analyze the document's structure and content. Be factual.",
        f"Analyze this document. Decide whether it is a software requirements document, "
        f"summarize it, list its stakeholders and sections, and list standard SRS sections "
        f"that are missing.\n\n<document>\n{state['srs_text']}\n</document>",
    )
    log("Document Analyzer", f"done: '{result['project_name']}', is SRS: {result['is_srs']}")
    return {"analysis": result}


# ---------------------------------------------------------------
# Branch 1: Requirement Agent -> Architecture Agent
# ---------------------------------------------------------------
def requirement_agent(state):
    log("Requirement Agent", "extracting requirements...")
    result = ask_structured(
        RequirementsOutput,
        "You are a requirements engineer. Extract requirements precisely, keeping each "
        "requirement's ID exactly as written in the document. Never invent requirements.",
        f"Extract every functional and non-functional requirement from this SRS, with its "
        f"ID, type, a one-sentence description and a priority. Then list any requirements "
        f"that are vague or untestable, and why.\n\n<srs>\n{state['srs_text']}\n</srs>",
    )
    log("Requirement Agent", f"done: {len(result['requirements'])} requirements, "
                             f"{len(result['ambiguities'])} ambiguities")
    return {"requirements": result}


def architecture_agent(state):
    log("Architecture Agent", "designing the architecture...")
    result = ask_structured(
        ArchitectureOutput,
        "You are a software architect. Propose a practical architecture that satisfies the "
        "given requirements and the project's constraints (team size, deadline, integrations).",
        f"Design an architecture for this system. For every component, list the requirement "
        f"IDs it serves; together the components should cover every requirement.\n\n"
        f"<srs>\n{state['srs_text']}\n</srs>\n\n"
        f"Requirements extracted by the Requirement Agent:\n"
        f"{as_json(state['requirements']['requirements'])}",
    )
    log("Architecture Agent", f"done: {len(result['components'])} components")
    return {"architecture": result}


# ---------------------------------------------------------------
# Branch 2: Risk Agent -> Test Case Agent
# ---------------------------------------------------------------
def risk_agent(state):
    log("Risk Agent", "identifying risks...")
    result = ask_structured(
        RiskOutput,
        "You are a project risk analyst. Identify concrete, specific risks for this project, "
        "not generic ones. Number them R-1, R-2, ...",
        f"Identify the main technical, security, schedule, requirements and operational "
        f"risks in this SRS. For each, give severity, likelihood, a mitigation, and the "
        f"related requirement IDs.\n\n<srs>\n{state['srs_text']}\n</srs>",
    )
    log("Risk Agent", f"done: {len(result['risks'])} risks")
    return {"risks": result}


def test_case_agent(state):
    log("Test Case Agent", "writing test cases...")
    result = ask_structured(
        TestCaseOutput,
        "You are a QA engineer. Write clear, concrete test cases. Number them TC-1, TC-2, ... "
        "and reference requirement IDs exactly as written in the SRS.",
        f"Write test cases for this SRS. Cover every functional requirement (FR-...) with at "
        f"least one test, and add tests that would catch the high-severity risks below.\n\n"
        f"<srs>\n{state['srs_text']}\n</srs>\n\n"
        f"Risks identified by the Risk Agent:\n{as_json(state['risks']['risks'])}",
    )
    log("Test Case Agent", f"done: {len(result['test_cases'])} test cases")
    return {"test_cases": result}
