"""Pydantic-схемы запросов и ответов API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """Тело POST /api/recommend — профиль пользователя."""

    budget: Optional[float] = Field(default=None, ge=0)
    knowledge_level: Optional[float] = Field(default=None, ge=0, le=10)
    time_availability: Optional[float] = Field(default=None, ge=0)
    career_focus: Optional[float] = Field(default=None, ge=0, le=1)
    interests: List[str] = Field(default_factory=list)
    goals: str = Field(default="")


class RecommendationMeta(BaseModel):
    """Метаданные ответа рекомендаций."""

    fuzzy_rules_count: int
    total_courses_in_db: int


class RecommendationResponse(BaseModel):
    """Ответ POST /api/recommend."""

    recommendations: List[Dict[str, Any]]
    explanations: Dict[str, Any]
    meta: RecommendationMeta


class HealthResponse(BaseModel):
    """Ответ GET /api/health."""

    status: str
    components: List[str]
    templates_path: str
    templates_exists: bool


class DialogStepRequest(BaseModel):
    """Тело POST /api/dialog/step."""

    session_id: Optional[str] = None
    user_message: str = Field(default="", min_length=0)
    prev_belief: Optional[Dict[str, Any]] = None


class DialogStepResponse(BaseModel):
    """Ответ POST /api/dialog/step."""

    session_id: str
    next_question: Optional[str] = None
    updated_belief: Dict[str, Any]
    is_ready_for_recommend: bool
    profile_preview: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class WhatIfRequest(BaseModel):
    """Тело POST /api/explain/what_if — контрфактуальный анализ (§3.3)."""

    course_id: str
    base_profile: RecommendationRequest
    changed_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="Изменённые поля профиля, напр. {\"budget\": 60000}",
    )


class WhatIfResponse(BaseModel):
    """Ответ контрфактуального анализа."""

    course_id: str
    old_rank: int
    new_rank: int
    delta_rank: int
    old_score: Optional[float] = None
    new_score: Optional[float] = None
    explanation: str
    changed_fields: Dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    """Узел графа компетенций для визуализации."""

    id: str
    label: str
    type: str


class GraphEdge(BaseModel):
    """Ребро графа компетенций."""

    source: str
    target: str
    relation: str = ""


class GraphResponse(BaseModel):
    """Ответ GET /api/graph/{course_id} (§2.1)."""

    nodes: List[GraphNode]
    edges: List[GraphEdge]
    explanation_path: str
    course_id: str
    profession: str
