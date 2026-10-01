"""AI Document Processing Workflow (LangGraph).

Processes a Software Requirements Specification (SRS) with multiple agents:

    Input SRS -> Document Analyzer ─┬─> Requirement Agent -> Architecture Agent ─┐
                                    └─> Risk Agent        -> Test Case Agent    ─┴─>
                Merge Results (+ validation) -> Human Review (HITL) -> Final Report

The two branches run in parallel. Merge Results waits for both, then cross-checks the
agents' work against the SRS. Human Review pauses the graph (LangGraph interrupt) until a
person approves or rejects. The final report is saved as Markdown and JSON in reports/.

Usage:
    python srs_workflow.py                       # process sample_srs.md, ask for review
    python srs_workflow.py my_srs.md             # process your own SRS file
    python srs_workflow.py --auto-approve        # skip the interactive review
    python srs_workflow.py --graph               # print the graph as a Mermaid diagram
"""

import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents import (architecture_agent, document_analyzer, requirement_agent, risk_agent,
                    test_case_agent)
from llm_setup import CHAT_MODEL

HERE = Path(__file__).resolve().parent
REPORTS_DIR = HERE / "reports"
DEFAULT_SRS = HERE / "sample_srs.md"


class WorkflowState(TypedDict, total=False):
    """Shared state. Each node adds its own key, so parallel branches never collide."""
    srs_name: str
    srs_text: str
    analysis: dict        # Document Analyzer
    requirements: dict    # Requirement Agent
    architecture: dict    # Architecture Agent
    risks: dict           # Risk Agent
    test_cases: dict      # Test Case Agent
    validation: dict      # Merge Results
    review: dict          # Human Review
    report_path: str      # Final Report


# ---------------------------------------------------------------
# Merge Results: combine both branches and validate them against the SRS
# ---------------------------------------------------------------
def normalize_id(req_id):
    """Treat "FR-1", "fr-01" and "FR-01" as the same requirement ID."""
    match = re.fullmatch(r"\s*(N?FR)\s*-\s*0*(\d+)\s*", req_id, re.IGNORECASE)
    return f"{match.group(1).upper()}-{int(match.group(2))}" if match else req_id.strip().upper()


def merge_results(state):
    srs_ids = {normalize_id(i): i for i in re.findall(r"\bN?FR-\d+\b", state["srs_text"])}
    functional = {key for key, original in srs_ids.items() if original.startswith("FR")}

    extracted = {normalize_id(r["id"]) for r in state["requirements"]["requirements"]}
    mapped = {normalize_id(i) for c in state["architecture"]["components"]
              for i in c["requirement_ids"]}
    tests = state["test_cases"]["test_cases"]
    tested = {normalize_id(i) for t in tests for i in t["requirement_ids"]}
    high_risks = {r["id"] for r in state["risks"]["risks"] if r["severity"] == "high"}
    risks_tested = {i for t in tests for i in t["risk_ids"]}

    def names(keys):
        return ", ".join(sorted(srs_ids.get(k, k) for k in keys)) or "-"

    checks = [
        ("Requirement Agent found every requirement in the SRS",
         not (set(srs_ids) - extracted), f"missing: {names(set(srs_ids) - extracted)}"),
        ("Requirement Agent invented no requirements",
         not (extracted - set(srs_ids)), f"not in SRS: {names(extracted - set(srs_ids))}"),
        ("Architecture covers every requirement",
         not (set(srs_ids) - mapped), f"not mapped to a component: {names(set(srs_ids) - mapped)}"),
        ("Every functional requirement has a test case",
         not (functional - tested), f"untested: {names(functional - tested)}"),
        ("Every high-severity risk has a test case",
         not (high_risks - risks_tested), f"untested: {', '.join(sorted(high_risks - risks_tested))}"),
        ("Test cases only reference real requirements",
         not (tested - set(srs_ids)), f"unknown IDs: {names(tested - set(srs_ids))}"),
    ]
    validation = {
        "checks": [{"check": name, "passed": ok, "detail": "" if ok else detail}
                   for name, ok, detail in checks],
        "issues": [f"{name}: {detail}" for name, ok, detail in checks if not ok],
        "ambiguities": state["requirements"]["ambiguities"],
    }
    passed = sum(c["passed"] for c in validation["checks"])
    print(f"  [Merge Results] {passed}/{len(checks)} validation checks passed", file=sys.stderr)
    return {"validation": validation}


# ---------------------------------------------------------------
# Human Review (HITL): pause the graph until a person decides
# ---------------------------------------------------------------
def human_review(state):
    analysis = state["analysis"]
    packet = {
        "project": analysis["project_name"],
        "summary": analysis["summary"],
        "counts": {
            "requirements": len(state["requirements"]["requirements"]),
            "components": len(state["architecture"]["components"]),
            "risks": len(state["risks"]["risks"]),
            "test_cases": len(state["test_cases"]["test_cases"]),
        },
        "high_risks": [f"{r['id']}: {r['description']}" for r in state["risks"]["risks"]
                       if r["severity"] == "high"],
        "validation": state["validation"],
    }
    # interrupt() saves the state and stops here; the value passed to
    # Command(resume=...) becomes the return value when the graph continues
    decision = interrupt(packet)
    return {"review": decision}


# ---------------------------------------------------------------
# Final Report
# ---------------------------------------------------------------
def cell(text):
    """Make text safe for a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def build_markdown(state):
    a = state["analysis"]
    lines = [f"# SRS Analysis Report: {a['project_name']}", "",
             f"- **Source:** {state['srs_name']}",
             f"- **Generated:** {datetime.now():%Y-%m-%d %H:%M}",
             f"- **Model:** {CHAT_MODEL}", ""]

    if not a["is_srs"]:
        return "\n".join(lines + ["## Result", "",
                                  "The Document Analyzer found that this is **not a software "
                                  "requirements document**, so no further analysis was done.",
                                  "", f"Analyzer summary: {a['summary']}"])

    review = state["review"]
    status = "APPROVED" if review["decision"] == "approved" else "REJECTED - needs revision"
    lines += [f"- **Review status:** {status}", "",
              "## 1. Document Overview", "", a["summary"], "",
              f"**Stakeholders:** {', '.join(a['stakeholders'])}", "",
              f"**Missing SRS sections:** {', '.join(a['missing_sections']) or 'none'}", ""]

    lines += ["## 2. Requirements", "", "| ID | Type | Priority | Description |",
              "|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['type']} | {r['priority']} | {cell(r['description'])} |"
              for r in state["requirements"]["requirements"]]
    lines += ["", "**Ambiguous or untestable requirements:**", ""]
    lines += [f"- {x}" for x in state["requirements"]["ambiguities"]] or ["- none"]

    arch = state["architecture"]
    lines += ["", "## 3. Proposed Architecture", "", f"**Style:** {arch['style']}", "",
              "| Component | Responsibility | Technologies | Requirements |", "|---|---|---|---|"]
    lines += [f"| {cell(c['name'])} | {cell(c['responsibility'])} | "
              f"{cell(', '.join(c['technologies']))} | {', '.join(c['requirement_ids'])} |"
              for c in arch["components"]]
    lines += ["", f"**Data flow:** {arch['data_flow']}", "", "**Key decisions:**", ""]
    lines += [f"- {d}" for d in arch["key_decisions"]]

    lines += ["", "## 4. Risks", "",
              "| ID | Severity | Likelihood | Category | Risk | Mitigation | Requirements |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['severity']} | {r['likelihood']} | {r['category']} | "
              f"{cell(r['description'])} | {cell(r['mitigation'])} | "
              f"{', '.join(r['related_requirements'])} |" for r in state["risks"]["risks"]]

    lines += ["", "## 5. Test Cases", "",
              "| ID | Type | Title | Requirements | Risks | Expected result |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {t['id']} | {t['type']} | {cell(t['title'])} | {', '.join(t['requirement_ids'])} "
              f"| {', '.join(t['risk_ids']) or '-'} | {cell(t['expected_result'])} |"
              for t in state["test_cases"]["test_cases"]]

    lines += ["", "## 6. Validation (Merge Results)", ""]
    lines += [f"- {'PASS' if c['passed'] else 'FAIL'}: {c['check']}"
              + (f" ({c['detail']})" if c["detail"] else "")
              for c in state["validation"]["checks"]]

    lines += ["", "## 7. Human Review", "", f"- **Decision:** {status}",
              f"- **Reviewer:** {review.get('reviewer') or '-'}",
              f"- **Comments:** {review.get('comments') or '-'}", ""]
    return "\n".join(lines)


def final_report(state):
    REPORTS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", state["analysis"]["project_name"].lower()).strip("_")
    base = REPORTS_DIR / f"{slug or 'srs'}_{datetime.now():%Y%m%d_%H%M%S}"

    base.with_suffix(".md").write_text(build_markdown(state), encoding="utf-8")
    data = {key: state.get(key) for key in ("srs_name", "analysis", "requirements",
                                             "architecture", "risks", "test_cases",
                                             "validation", "review")}
    base.with_suffix(".json").write_text(json.dumps(data, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    return {"report_path": str(base.with_suffix(".md"))}


# ---------------------------------------------------------------
# Graph
# ---------------------------------------------------------------
def after_analysis(state):
    """Fan out to both branches for a real SRS; otherwise go straight to the report."""
    return ["requirement_agent", "risk_agent"] if state["analysis"]["is_srs"] else "final_report"


def build_graph():
    graph = StateGraph(WorkflowState)
    for node in (document_analyzer, requirement_agent, architecture_agent, risk_agent,
                 test_case_agent, merge_results, human_review, final_report):
        graph.add_node(node.__name__, node)

    graph.add_edge(START, "document_analyzer")
    graph.add_conditional_edges("document_analyzer", after_analysis,
                                ["requirement_agent", "risk_agent", "final_report"])
    graph.add_edge("requirement_agent", "architecture_agent")   # branch 1
    graph.add_edge("risk_agent", "test_case_agent")             # branch 2
    # A list of sources means: wait until BOTH branches have finished
    graph.add_edge(["architecture_agent", "test_case_agent"], "merge_results")
    graph.add_edge("merge_results", "human_review")
    graph.add_edge("human_review", "final_report")
    graph.add_edge("final_report", END)
    # The checkpointer saves state at each step, which is what lets interrupt() pause/resume
    return graph.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------
# Running the workflow
# ---------------------------------------------------------------
def ask_reviewer(packet):
    """Show the review packet and collect the human decision."""
    print("\n" + "=" * 72)
    print("HUMAN REVIEW REQUIRED")
    print("=" * 72)
    print(f"Project: {packet['project']}\n{packet['summary']}\n")
    counts = packet["counts"]
    print(f"Found {counts['requirements']} requirements, {counts['components']} architecture "
          f"components, {counts['risks']} risks, {counts['test_cases']} test cases.")
    if packet["high_risks"]:
        print("\nHigh-severity risks:")
        for risk in packet["high_risks"]:
            print(f"  - {risk}")
    print("\nValidation checks:")
    for check in packet["validation"]["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"  [{mark}] {check['check']}" + (f" ({check['detail']})" if check["detail"] else ""))
    if packet["validation"]["ambiguities"]:
        print("\nAmbiguous requirements flagged:")
        for item in packet["validation"]["ambiguities"]:
            print(f"  - {item}")
    print("=" * 72)

    while True:
        answer = input("Approve this analysis? [a]pprove / [r]eject: ").strip().lower()
        if answer in ("a", "approve", "r", "reject"):
            break
        print("Please type a or r.")
    reviewer = input("Your name: ").strip()
    comments = input("Comments (optional): ").strip()
    return {"decision": "approved" if answer.startswith("a") else "rejected",
            "reviewer": reviewer, "comments": comments}


def run(srs_path, auto_approve=False):
    workflow = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    inputs = {"srs_name": srs_path.name, "srs_text": srs_path.read_text(encoding="utf-8")}

    print(f"Processing {srs_path.name} with {CHAT_MODEL}...", file=sys.stderr)
    while True:
        for update in workflow.stream(inputs, config, stream_mode="updates"):
            for node in update:
                if node != "__interrupt__":
                    print(f"  completed: {node}", file=sys.stderr)

        snapshot = workflow.get_state(config)
        if not snapshot.next:            # nothing left to run: finished
            break
        packet = snapshot.tasks[0].interrupts[0].value   # paused at human_review
        if auto_approve:
            decision = {"decision": "approved", "reviewer": "auto",
                        "comments": "Auto-approved with --auto-approve"}
            print("\nHuman review: auto-approved (--auto-approve)", file=sys.stderr)
        else:
            decision = ask_reviewer(packet)
        inputs = Command(resume=decision)   # continue the paused graph with the decision

    state = workflow.get_state(config).values
    print("\n" + "=" * 72)
    print(f"FINAL REPORT: {state['analysis']['project_name']}")
    if state["analysis"]["is_srs"]:
        print(f"Review decision: {state['review']['decision'].upper()}")
        checks = state["validation"]["checks"]
        print(f"Validation: {sum(c['passed'] for c in checks)}/{len(checks)} checks passed")
    else:
        print("Not a software requirements document - analysis stopped after the analyzer.")
    print(f"Saved: {state['report_path']}")
    print(f"       {Path(state['report_path']).with_suffix('.json')}")
    return state


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--graph" in sys.argv:
        print(build_graph().get_graph().draw_mermaid())
    else:
        path = Path(args[0]) if args else DEFAULT_SRS
        if not path.is_file():
            raise SystemExit(f"File not found: {path}")
        run(path, auto_approve="--auto-approve" in sys.argv)
