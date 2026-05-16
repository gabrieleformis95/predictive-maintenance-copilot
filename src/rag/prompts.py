"""Prompt templates for the explanation step.

The LLM is asked to produce a strictly-typed JSON object so the API can
consume it without parsing free-form text.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AlertCitation(BaseModel):
    source: str
    page: int | None = None
    quote: str = Field(description="Short quote (<=200 chars) supporting the recommendation")


class AnomalyAlert(BaseModel):
    """Structured anomaly alert returned to the operator."""

    severity: str = Field(description="info | warning | critical")
    probable_cause: str = Field(description="Plain-language probable cause, 1-2 sentences")
    recommended_action: str = Field(description="Concrete next step the operator should take")
    affected_sensors: list[str] = Field(default_factory=list)
    citations: list[AlertCitation] = Field(default_factory=list)


SYSTEM_PROMPT = """\
You are an industrial maintenance assistant. You receive (1) a description of
an anomaly detected by an ML system on equipment sensor data, and (2) excerpts
from maintenance manuals retrieved by a search engine.

Your job is to write a SHORT, ACTIONABLE alert for a plant operator.

Rules:
- Reply ONLY with a valid JSON object using EXACTLY these field names:
  {
    "severity": "info" | "warning" | "critical",
    "probable_cause": "<1-2 sentence plain-language cause>",
    "recommended_action": "<concrete next step for the operator>",
    "affected_sensors": ["<sensor_name>", ...],
    "citations": [
      {"source": "<filename>", "page": <int or null>, "quote": "<<=200 char quote>"}
    ]
  }
- Do not add any fields outside this schema.
- Cite each recommendation with the manual excerpt it came from.
- If the retrieved excerpts do not support an answer, set severity to "info"
  and say so in probable_cause.
- Never invent procedures.
"""


USER_PROMPT_TEMPLATE = """\
ANOMALY DESCRIPTION
-------------------
{anomaly_description}

TOP SENSOR DEVIATIONS (z-score)
-------------------------------
{sensor_deviations}

RETRIEVED MANUAL EXCERPTS
-------------------------
{retrieved_excerpts}

Now produce a JSON object conforming to the AnomalyAlert schema.
"""


def build_user_prompt(
    anomaly_description: str,
    sensor_deviations: str,
    retrieved_excerpts: str,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        anomaly_description=anomaly_description,
        sensor_deviations=sensor_deviations,
        retrieved_excerpts=retrieved_excerpts,
    )
