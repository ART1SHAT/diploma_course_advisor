"""
Генератор объяснений (§2.3) и контрфактуальный анализ (§3.3).
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from app.user_profile import UserProfile

NEUTRAL_SCORE = 0.5
REASON_TYPES = (
    "fuzzy_rule",
    "semantic_match",
    "budget_constraint",
    "competency_align",
)


class ExplanationGenerator:
    """Формирует структурированные объяснения рекомендации."""

    def __init__(
        self,
        profile: UserProfile,
        course: Dict[str, Any],
        fuzzy_trace: List[Dict[str, Any]],
        semantic_sim: float,
    ) -> None:
        self.profile = profile
        self.course = course
        self.fuzzy_trace = fuzzy_trace or []
        self.semantic_sim = max(0.0, min(1.0, float(semantic_sim)))

    def _profile_query(self) -> str:
        goals = (self.profile.goals or "").strip()
        interests = " ".join(self.profile.interests or [])
        return f"{goals} {interests}".strip()

    def _extract_evidence(self, max_len: int = 140) -> Optional[str]:
        query = self._profile_query() or (self.profile.goals or "")
        description = str(self.course.get("description", "") or "")
        if not query or not description:
            return None

        desc = description.strip()
        desc_lower = desc.lower()
        query_lower = query.lower().strip()

        idx = desc_lower.find(query_lower)
        needle = query_lower
        if idx < 0:
            tokens = [w for w in re.split(r"\W+", query_lower) if len(w) > 4]
            for token in tokens:
                idx = desc_lower.find(token)
                if idx >= 0:
                    needle = token
                    break
            if idx < 0:
                return None

        start = max(0, idx - 50)
        end = min(len(desc), idx + len(needle) + 70)
        snippet = desc[start:end].strip()
        if start > 0:
            snippet = "…" + snippet
        if end < len(desc):
            snippet = snippet + "…"
        return snippet[:max_len] if len(snippet) > max_len else snippet

    def _reasons_fuzzy(self) -> List[Dict[str, Any]]:
        reasons: List[Dict[str, Any]] = []
        for t in self.fuzzy_trace:
            act = float(t.get("activation", 0))
            if act <= 0.1:
                continue
            reasons.append({
                "type": "fuzzy_rule",
                "text": (
                    f"Активировано правило {t.get('rule_id', '?')} "
                    f"({act:.0%}): {t.get('rule_name', '')}"
                ),
                "weight": round(act, 4),
                "rule_id": t.get("rule_id"),
                "conclusion": t.get("conclusion"),
            })
        return reasons

    def _reasons_semantic(self) -> List[Dict[str, Any]]:
        reasons: List[Dict[str, Any]] = []
        query = self._profile_query()
        description = str(self.course.get("description", "") or "").lower()
        goals = (self.profile.goals or "").strip()

        if goals and goals.lower() in description:
            reasons.append({
                "type": "semantic_match",
                "text": f"Соответствует цели: «{goals}»",
                "weight": round(self.semantic_sim, 4),
            })
        elif query and self.semantic_sim > NEUTRAL_SCORE:
            reasons.append({
                "type": "semantic_match",
                "text": "Семантическая близость описания курса к запросу",
                "weight": round(self.semantic_sim, 4),
            })
        return reasons

    def _reasons_budget(self) -> List[Dict[str, Any]]:
        if self.profile.budget is None:
            return []

        price = float(self.course.get("price", 0) or 0)
        user_budget = float(self.profile.budget)
        within = price <= user_budget if user_budget > 0 else price == 0

        if user_budget <= 0:
            text = (
                "Бесплатный курс — соответствует нулевому бюджету"
                if price == 0
                else f"Платный курс ({int(price)} ₽) при нулевом бюджете"
            )
        elif price == 0:
            text = "Бесплатный курс укладывается в бюджет"
        elif within:
            text = f"Укладывается в бюджет ({int(price)} ₽ ≤ {int(user_budget)} ₽)"
        else:
            text = f"Выше бюджета ({int(price)} ₽ > {int(user_budget)} ₽)"

        weight = 1.0 if within else 0.25
        return [{"type": "budget_constraint", "text": text, "weight": weight}]

    def _reasons_competency(self) -> List[Dict[str, Any]]:
        reasons: List[Dict[str, Any]] = []
        if not self.profile.interests:
            return reasons

        course_skills = {
            str(s).lower()
            for s in (self.course.get("skills") or [])
        }
        description = str(self.course.get("description", "") or "").lower()
        matched = [
            i for i in self.profile.interests
            if i.lower() in course_skills or i.lower() in description
        ]
        if not matched:
            return reasons

        overlap = len(matched) / max(len(self.profile.interests), 1)
        reasons.append({
            "type": "competency_align",
            "text": f"Совпадение компетенций/интересов: {', '.join(matched[:4])}",
            "weight": round(min(1.0, 0.5 + 0.5 * overlap), 4),
        })
        return reasons

    def _compute_confidence(self, reasons: List[Dict[str, Any]]) -> float:
        if not reasons:
            return NEUTRAL_SCORE
        weights = [float(r.get("weight", 0)) for r in reasons]
        fuzzy_w = [
            r["weight"] for r in reasons if r["type"] == "fuzzy_rule"
        ]
        sem_w = [
            r["weight"] for r in reasons if r["type"] == "semantic_match"
        ]
        budget_w = [
            r["weight"] for r in reasons if r["type"] == "budget_constraint"
        ]

        fuzzy_part = sum(fuzzy_w) / len(fuzzy_w) if fuzzy_w else NEUTRAL_SCORE
        sem_part = sum(sem_w) / len(sem_w) if sem_w else NEUTRAL_SCORE
        budget_part = budget_w[0] if budget_w else NEUTRAL_SCORE
        comp_part = sum(
            r["weight"] for r in reasons if r["type"] == "competency_align"
        ) / max(1, len([r for r in reasons if r["type"] == "competency_align"]))

        raw = 0.35 * sem_part + 0.30 * fuzzy_part + 0.20 * budget_part + 0.15 * comp_part
        if reasons:
            raw = 0.7 * raw + 0.3 * (sum(weights) / len(weights))
        return round(max(0.0, min(1.0, raw)), 3)

    def generate(self) -> Dict[str, Any]:
        """Возвращает {reasons, evidence, confidence}."""
        reasons: List[Dict[str, Any]] = []
        reasons.extend(self._reasons_semantic())
        reasons.extend(self._reasons_fuzzy())
        reasons.extend(self._reasons_budget())
        reasons.extend(self._reasons_competency())

        if not reasons:
            reasons.append({
                "type": "semantic_match",
                "text": "Общее соответствие профилю (нейтральный скоринг)",
                "weight": NEUTRAL_SCORE,
            })

        reasons.sort(key=lambda r: r["weight"], reverse=True)
        evidence = self._extract_evidence()
        confidence = self._compute_confidence(reasons)

        return {
            "reasons": reasons,
            "evidence": evidence,
            "confidence": confidence,
        }


def profile_to_dict(profile: UserProfile) -> Dict[str, Any]:
    return {
        "budget": profile.budget,
        "knowledge_level": profile.knowledge_level,
        "time_availability": profile.time_availability,
        "career_focus": profile.career_focus,
        "interests": profile.interests,
        "goals": profile.goals,
    }


def apply_profile_changes(
    profile: UserProfile, changes: Dict[str, Any]
) -> UserProfile:
    """Контрфактуальное изменение полей профиля."""
    data = profile_to_dict(profile)
    for key, value in changes.items():
        if key in data:
            data[key] = value
    return UserProfile(
        budget=data.get("budget"),
        knowledge_level=data.get("knowledge_level"),
        time_availability=data.get("time_availability"),
        career_focus=data.get("career_focus"),
        interests=data.get("interests") or [],
        goals=data.get("goals") or "",
    )


def build_what_if_explanation(
    field: str,
    old_value: Any,
    new_value: Any,
    delta_rank: int,
) -> str:
    """Текстовое объяснение контрфактуала для ответа API."""
    field_labels = {
        "budget": "бюджет",
        "knowledge_level": "уровень знаний",
        "time_availability": "время на обучение",
        "career_focus": "карьерная направленность",
        "goals": "цель обучения",
        "interests": "интересы",
    }
    label = field_labels.get(field, field)

    if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
        if old_value and old_value != 0:
            pct = abs((new_value - old_value) / old_value) * 100
            change = f"изменении {label} на {pct:.0f}%"
        else:
            change = f"изменении {label} до {new_value}"
    else:
        change = f"изменении {label}"

    if delta_rank > 0:
        pos_word = "позици" if abs(delta_rank) == 1 else "позиции"
        return (
            f"При {change} курс поднимется на {delta_rank} {pos_word} в рейтинге"
        )
    if delta_rank < 0:
        return (
            f"При {change} курс опустится на {abs(delta_rank)} позиций в рейтинге"
        )
    return f"При {change} позиция курса в рейтинге не изменится"


def find_course_rank(
    ranked: List[Dict[str, Any]], course_id: str
) -> Tuple[int, int]:
    """Возвращает (rank 1-based, total) или (0, total) если не в списке."""
    total = len(ranked)
    for i, c in enumerate(ranked, start=1):
        if str(c.get("id")) == str(course_id):
            return i, total
    return 0, total
