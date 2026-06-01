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
