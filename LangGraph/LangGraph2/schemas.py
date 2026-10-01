"""Structured outputs for each agent.

Each agent's LLM call is constrained to one of these Pydantic models (Ollama's JSON-schema
mode), and the reply is validated against it, so every agent returns predictable fields.
"""

from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["high", "medium", "low"]


class DocumentAnalysis(BaseModel):
    is_srs: bool = Field(description="True if the text is a software requirements document")
    project_name: str
    summary: str = Field(description="2-3 sentence summary of what the system is for")
    stakeholders: list[str] = Field(description="User groups and other stakeholders")
    sections_found: list[str] = Field(description="Main sections present in the document")
    missing_sections: list[str] = Field(
        description="Standard SRS sections that are missing, e.g. 'Non-functional requirements'")


class Requirement(BaseModel):
    id: str = Field(description="The requirement ID exactly as written in the SRS, e.g. FR-01")
    type: Literal["functional", "non-functional"]
    description: str
    priority: Level


class RequirementsOutput(BaseModel):
    requirements: list[Requirement] = Field(min_length=1)
    ambiguities: list[str] = Field(
        description="Requirements that are vague or untestable, and why (empty if none)")


class Component(BaseModel):
    name: str
    responsibility: str
    technologies: list[str]
    requirement_ids: list[str] = Field(description="IDs of the requirements this component serves")


class ArchitectureOutput(BaseModel):
    style: str = Field(description="Overall architecture style, e.g. 'three-tier web application'")
    components: list[Component] = Field(min_length=2)
    data_flow: str = Field(description="How a typical request flows through the components")
    key_decisions: list[str] = Field(description="Important design decisions and their reasons")


class Risk(BaseModel):
    id: str = Field(description="Risk ID: R-1, R-2, ...")
    description: str
    category: Literal["technical", "security", "schedule", "requirements", "operational"]
    severity: Level
    likelihood: Level
    mitigation: str
    related_requirements: list[str] = Field(description="Related SRS requirement IDs, if any")


class RiskOutput(BaseModel):
    risks: list[Risk] = Field(min_length=1)


class TestCase(BaseModel):
    id: str = Field(description="Test case ID: TC-1, TC-2, ...")
    title: str
    type: Literal["functional", "performance", "security", "usability", "integration"]
    requirement_ids: list[str] = Field(description="SRS requirement IDs this test verifies")
    risk_ids: list[str] = Field(description="Risk IDs this test helps to catch (empty if none)")
    steps: list[str] = Field(min_length=1)
    expected_result: str


class TestCaseOutput(BaseModel):
    test_cases: list[TestCase] = Field(min_length=1)
