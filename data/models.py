# data/models.py
"""
Единая схема данных и модели для информационно-советующей системы.
Соответствует разделам диплома:
- §1.4: Требования к данным и валидация метаданных
- §2.2: Модель лингвистических переменных и термов
- §2.3: Правила нечёткого вывода (Mamdani)
- §2.4 / §3.2: Профиль пользователя и состояние диалога
"""

from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
import math

# ─────────────────────────────────────────────────────────────
# ENUMS (Типизация для строгой валидации)
# ─────────────────────────────────────────────────────────────
class ProviderType(str, Enum):
    MOOC = "mooc"
    UNIVERSITY = "university"
    BOOTCAMP = "bootcamp"
    CORPORATE = "corporate"
    GOVERNMENT = "government_platform"

class CourseFormat(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    HYBRID = "hybrid"

class CertificationType(str, Enum):
    NONE = "none"
    PROFESSIONAL = "professional"
    STATE = "state"

# ─────────────────────────────────────────────────────────────
# 1. ЕДИНАЯ СХЕМА КУРСА (§1.4, §3.2)
# ─────────────────────────────────────────────────────────────
class UnifiedCourse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, json_schema_extra={
        "example": {
            "title": "Python для анализа данных",
            "provider": "Stepik",
            "provider_type": "mooc",
            "price": 15000.0,
            "format": "online",
            "level": ["beginner"],
            "competencies": ["python", "pandas"],
            "certification": "professional"
        }
    })

    id: str = Field(default_factory=lambda: f"C_{uuid.uuid4().hex[:8].upper()}")
    title: str = Field(min_length=2, max_length=300)
    provider: str = Field(min_length=1, max_length=100)
    provider_type: ProviderType
    price: float = Field(default=0.0, ge=0)
    duration_weeks: Optional[int] = Field(default=None, ge=1, le=52)
    format: CourseFormat
    language: str = Field(default="ru", min_length=2, max_length=5)
    level: List[str] = Field(default=["beginner"])
    competencies: List[str] = Field(default_factory=list)
    certification: CertificationType
    url: str = Field(default="", min_length=0, max_length=500)
    description: str = Field(default="", max_length=5000)
    source_quality: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Внутренние поля для ранжирования (не обязательны при импорте)
    embedding_vector: Optional[List[float]] = Field(default=None, alias="embedding")
    fuzzy_activation: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def normalize_competencies(cls, data: Any) -> Any:
        """Приводит компетенции к нижнему регистру и удаляет дубли"""
        if isinstance(data, dict):
            comps = data.get("competencies", [])
            if comps:
                data["competencies"] = list(set(str(c).lower().strip() for c in comps if str(c).strip()))
        return data


# ─────────────────────────────────────────────────────────────
# 2. ЛИНГВИСТИЧЕСКИЕ ПЕРЕМЕННЫЕ И ТЕРМЫ (§2.2)
# ─────────────────────────────────────────────────────────────
class MembershipFunction(BaseModel):
    type: str = Field(pattern="^(triangle|trapezoid|gaussian)$")
    params: List[float]  # [a, b, c] или [a, b, c, d] или [mean, sigma]

class LinguisticTerm(BaseModel):
    name: str
    membership_function: MembershipFunction

    def evaluate(self, x: float) -> float:
        """Вычисляет μ(x) ∈ [0, 1]"""
        p = self.membership_function.params
        t = self.membership_function.type
        if t == "triangle":
            a, b, c = p
            if a <= x <= b: return (x - a) / (b - a) if b != a else 0
            if b < x <= c: return (c - x) / (c - b) if c != b else 0
            return 0.0
        elif t == "trapezoid":
            a, b, c, d = p
            if a <= x <= b: return (x - a) / (b - a) if b != a else 0
            if b < x < c: return 1.0
            if c <= x <= d: return (d - x) / (d - c) if d != c else 0
            return 0.0
        elif t == "gaussian":
            mean, sigma = p
            if sigma <= 0: return 1.0 if x == mean else 0.0
            return math.exp(-0.5 * ((x - mean) / sigma) ** 2)
        return 0.0

class LinguisticVariable(BaseModel):
    name: str
    terms: Dict[str, LinguisticTerm]
    unit: str = ""  # "RUB", "hours/week", "scale 0-10"


# ─────────────────────────────────────────────────────────────
# 3. ПРАВИЛА НЕЧЁТКОГО ВЫВОДА (§2.3)
# ─────────────────────────────────────────────────────────────
class FuzzyAntecedent(BaseModel):
    variable: str
    term: str
    operator: str = Field(default="AND", pattern="^(AND|OR)$")
    weight: float = Field(default=1.0, ge=0.0, le=1.0)

class FuzzyConsequent(BaseModel):
    variable: str
    term: str
    boost: float = Field(default=0.1, ge=0.0, le=1.0)

class FuzzyRule(BaseModel):
    id: str = Field(default_factory=lambda: f"R_{uuid.uuid4().hex[:4].upper()}")
    name: str
    antecedents: List[FuzzyAntecedent]
    consequent: FuzzyConsequent
    priority: int = Field(default=1, ge=1, le=5)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


# ─────────────────────────────────────────────────────────────
# 4. ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ И ДИАЛОГ (§2.2, §2.4, §3.3)
# ─────────────────────────────────────────────────────────────
class UserProfile(BaseModel):
    # Структурированные атрибуты
    age: Optional[int] = Field(default=None, ge=14, le=100)
    education_level: Optional[str] = None
    location: Optional[str] = None

    # Лингвистические переменные (числовые входы для fuzzy-движка)
    budget: Optional[float] = Field(default=None, ge=0)
    knowledge_level: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    time_availability: Optional[float] = Field(default=None, ge=0.0, le=40.0)
    career_focus: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # 0=personal, 0.5=academic, 1=employment

    # Текстовые предпочтения
    interests: List[str] = Field(default_factory=list)
    goals: str = Field(default="", max_length=300)
    preferred_language: str = Field(default="ru")

    # Belief State (§2.4)
    clarified_slots: List[str] = Field(default_factory=list)
    session_confidence: Dict[str, float] = Field(default_factory=dict)

    def to_fuzzy_input(self) -> Dict[str, float]:
        """Извлекает поля для нечёткого вывода, игнорируя None"""
        return {
            k: v for k, v in {
                "budget": self.budget,
                "knowledge_level": self.knowledge_level,
                "time_availability": self.time_availability,
                "career_focus": self.career_focus
            }.items() if v is not None
        }


# ─────────────────────────────────────────────────────────────
# 5. API КОНТРАКТЫ (для FastAPI)
# ─────────────────────────────────────────────────────────────
class RecommendationRequest(BaseModel):
    budget: Optional[float] = None
    knowledge_level: Optional[float] = None
    time_availability: Optional[float] = None
    career_focus: Optional[float] = None
    interests: List[str] = Field(default_factory=list)
    goals: str = Field(default="")
    top_k: int = Field(default=5, ge=1, le=20)

class ExplanationItem(BaseModel):
    type: str  # fuzzy_rule, semantic_match, constraint, evidence
    content: str
    confidence: Optional[float] = None

class RecommendationResponse(BaseModel):
    course: UnifiedCourse
    explanations: List[ExplanationItem]
    overall_confidence: float
    fuzzy_trace: Optional[List[Dict[str, Any]]] = None