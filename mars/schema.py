from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


CHOICES = {"a", "b", "c", "d"}

ACTION_CONTINUE = "continue_thinking"
ACTION_ADD_DATA = "add_data"
ACTION_ANSWER = "answer"
ACTION_RANDOM = "random_fallback"
VALID_ACTIONS = {ACTION_CONTINUE, ACTION_ADD_DATA, ACTION_ANSWER, ACTION_RANDOM}


@dataclass
class Evidence:
    source_type: str
    content: str
    day: Optional[str] = None
    owner: Optional[str] = None
    time: Optional[str] = None
    path: Optional[str] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, row: Dict[str, Any]) -> "Evidence":
        return cls(
            source_type=str(row.get("source_type", "")),
            content=str(row.get("content", "")),
            day=row.get("day"),
            owner=row.get("owner"),
            time=row.get("time"),
            path=row.get("path"),
            score=float(row.get("score", 0.0) or 0.0),
            metadata=dict(row.get("metadata") or {}),
        )


@dataclass
class AgentDecision:
    action: str
    rationale: str = ""
    source_type: Optional[str] = None
    day: Optional[str] = None
    owner: Optional[str] = None
    time: Optional[str] = None
    answer: Optional[str] = None
    confidence: float = 0.0

    @classmethod
    def from_json(cls, row: Dict[str, Any]) -> "AgentDecision":
        action = str(row.get("action", ACTION_CONTINUE)).strip()
        answer = row.get("answer")
        if isinstance(answer, str):
            answer = answer.strip().lower()
        if answer not in CHOICES:
            answer = None
        return cls(
            action=action if action in VALID_ACTIONS else ACTION_CONTINUE,
            rationale=str(row.get("rationale", "")),
            source_type=row.get("source_type"),
            day=row.get("day"),
            owner=row.get("owner"),
            time=row.get("time"),
            answer=answer,
            confidence=float(row.get("confidence", 0.0) or 0.0),
        )


@dataclass
class Question:
    question_id: str
    query: str
    answers: Dict[str, str]

    @classmethod
    def from_json(cls, row: Dict[str, Any]) -> "Question":
        return cls(
            question_id=str(row.get("id", "")),
            query=str(row.get("query", "")),
            answers={str(k).lower(): str(v) for k, v in dict(row.get("answers", {})).items()},
        )

