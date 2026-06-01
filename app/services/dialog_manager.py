"""
Диалоговый менеджер (§2.4): belief state, энтропия, выбор уточняющего вопроса.
"""
from __future__ import annotations

import logging
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Лингвистические термы по слотам (согласованы с fuzzy_engine)
SLOT_TERM_VOCAB: Dict[str, List[str]] = {
    "budget": ["low", "medium", "high"],
    "knowledge_level": ["beginner", "intermediate", "advanced"],
    "time_availability": ["short", "medium", "long"],
    "career_focus": ["employment", "academic", "personal"],
}

REQUIRED_SLOTS = ["budget", "career_focus"]
OPTIONAL_SLOTS = ["knowledge_level", "time_availability", "goals"]
ALL_SLOTS = REQUIRED_SLOTS + OPTIONAL_SLOTS

# Стоимость вопроса (Cost) — чем выше, тем реже спрашиваем
SLOT_QUESTION_COST: Dict[str, float] = {
    "budget": 0.25,
    "career_focus": 0.20,
    "knowledge_level": 0.15,
    "time_availability": 0.15,
    "goals": 0.10,
}

SLOT_QUESTIONS: Dict[str, str] = {
    "budget": "Какой у вас бюджет на обучение (в рублях)?",
    "career_focus": "Какова главная цель: трудоустройство, учёба или личное развитие?",
    "knowledge_level": "Оцените текущий уровень знаний по шкале от 0 (новичок) до 10.",
    "time_availability": "Сколько часов в неделю готовы уделять обучению?",
    "goals": "Сформулируйте ключевую цель обучения своими словами.",
}

LAMBDA_COST = 0.35
READINESS_CONF_THRESHOLD = 0.55
CONFLICT_TERM_THRESHOLD = 0.35

# Числовые якоря для маппинга в термы
NUMERIC_TERM_ANCHORS = {
    "budget": {"low": 15_000, "medium": 50_000, "high": 120_000},
    "knowledge_level": {"beginner": 2, "intermediate": 5, "advanced": 8},
    "time_availability": {"short": 3, "medium": 8, "long": 15},
    "career_focus": {"personal": 0.0, "academic": 0.5, "employment": 1.0},
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _normalize_distribution(terms: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in terms.values())
    if total <= 0:
        n = len(terms)
        return {k: 1.0 / n for k in terms} if n else {}
    return {k: max(0.0, v) / total for k, v in terms.items()}


def _entropy(dist: Dict[str, float]) -> float:
    if not dist:
        return 1.0
    probs = _normalize_distribution(dist)
    h = 0.0
    for p in probs.values():
        if p > 1e-9:
            h -= p * math.log2(p)
    max_h = math.log2(len(probs)) if len(probs) > 1 else 1.0
    return h / max_h if max_h > 0 else 0.0


def numeric_to_term(slot: str, value: float) -> str:
    anchors = NUMERIC_TERM_ANCHORS.get(slot, {})
    if not anchors:
        return str(value)
    best_term = min(anchors, key=lambda t: abs(anchors[t] - value))
    return best_term


def term_to_numeric(slot: str, term: str) -> Optional[float]:
    anchors = NUMERIC_TERM_ANCHORS.get(slot, {})
    return anchors.get(term)


class BeliefState:
    """
    Состояние убеждений: {slot: {terms: {term: belief}, source_conf: float, numeric?: float}}.
    Слияние уверенности: new_conf = old + conf * (1 - old).
    """

    def __init__(self) -> None:
        self.slots: Dict[str, Dict[str, Any]] = {}
        self.goals_text: str = ""
        self.interests: List[str] = []
        self.turn_count: int = 0

    def update(self, slot: str, value: Any, confidence: float) -> None:
        """Нечёткое слияние уверенности и обновление распределения по терму."""
        conf = _clamp01(float(confidence))
        slot = slot.strip()

        if slot == "goals":
            if isinstance(value, str) and value.strip():
                self.goals_text = value.strip()
            self.turn_count += 1
            return

        if slot == "interests":
            if isinstance(value, list):
                self.interests = list(value)
            elif isinstance(value, str):
                self.interests = [v.strip() for v in value.split(",") if v.strip()]
            self.turn_count += 1
            return

        if slot not in SLOT_TERM_VOCAB:
            logger.warning("Неизвестный слот belief: %s", slot)
            return

        # Определяем терм
        if isinstance(value, str) and value in SLOT_TERM_VOCAB[slot]:
            term = value
            numeric = term_to_numeric(slot, term)
        elif isinstance(value, (int, float)):
            numeric = float(value)
            term = numeric_to_term(slot, numeric)
        else:
            term = str(value)
            numeric = term_to_numeric(slot, term)

        entry = self.slots.setdefault(
            slot,
            {"terms": {t: 0.0 for t in SLOT_TERM_VOCAB[slot]}, "source_conf": 0.0},
        )
        old_conf = float(entry.get("source_conf", 0.0))
        entry["source_conf"] = _clamp01(old_conf + conf * (1.0 - old_conf))

        terms: Dict[str, float] = entry.setdefault(
            "terms", {t: 0.0 for t in SLOT_TERM_VOCAB[slot]}
        )
        boost = conf * (1.0 - terms.get(term, 0.0))
        terms[term] = _clamp01(terms.get(term, 0.0) + boost)
        # Ослабляем альтернативы
        for t in terms:
            if t != term:
                terms[t] *= 1.0 - 0.5 * conf
        entry["terms"] = _normalize_distribution(terms)
        if numeric is not None:
            entry["numeric"] = numeric
        entry["last_term"] = term
        self.turn_count += 1

    def slot_entropy(self, slot: str) -> float:
        entry = self.slots.get(slot)
        if not entry:
            return 1.0
        return _entropy(entry.get("terms", {}))

    def get_entropy(self) -> float:
        """Средняя нормированная энтропия по слотам с лингвистическими термами."""
        entropies = [self.slot_entropy(s) for s in SLOT_TERM_VOCAB]
        if self.goals_text:
            entropies.append(0.1)
        else:
            entropies.append(0.9)
        return sum(entropies) / len(entropies)

    def has_conflict(self, slot: str) -> bool:
        entry = self.slots.get(slot)
        if not entry:
            return False
        probs = sorted(entry.get("terms", {}).values(), reverse=True)
        if len(probs) < 2:
            return False
        return probs[0] >= CONFLICT_TERM_THRESHOLD and probs[1] >= CONFLICT_TERM_THRESHOLD

    def get_conflicts(self) -> List[str]:
        return [s for s in SLOT_TERM_VOCAB if self.has_conflict(s)]

    def is_slot_filled(self, slot: str) -> bool:
        if slot == "goals":
            return bool(self.goals_text.strip())
        entry = self.slots.get(slot)
        if not entry:
            return False
        return float(entry.get("source_conf", 0)) >= READINESS_CONF_THRESHOLD

    def is_ready_for_recommend(self) -> bool:
        for slot in REQUIRED_SLOTS:
            if not self.is_slot_filled(slot):
                return False
            if self.has_conflict(slot):
                return False
        return True

    def expected_information_gain(self, slot: str) -> float:
        """EIG ≈ энтропия слота × (1 − source_conf)."""
        entry = self.slots.get(slot, {})
        conf = float(entry.get("source_conf", 0.0)) if entry else 0.0
        return self.slot_entropy(slot) * (1.0 - conf)

    def next_question(self) -> Optional[str]:
        """Выбор слота: argmax(EIG − λ·Cost)."""
        if self.is_ready_for_recommend():
            return None

        conflicts = self.get_conflicts()
        if conflicts:
            slot = conflicts[0]
            return (
                f"Уточните, пожалуйста: {SLOT_QUESTIONS.get(slot, slot)} "
                f"(обнаружено противоречие в ответах)."
            )

        best_slot: Optional[str] = None
        best_score = -math.inf

        # Сначала обязательные слоты (§2.4), затем опциональные
        slot_order = REQUIRED_SLOTS + [
            s for s in OPTIONAL_SLOTS if s not in REQUIRED_SLOTS
        ]

        for slot in slot_order:
            if slot == "goals":
                if self.goals_text:
                    continue
                eig = 0.85
            else:
                if self.is_slot_filled(slot):
                    continue
                eig = self.expected_information_gain(slot)

            cost = SLOT_QUESTION_COST.get(slot, 0.2)
            score = eig - LAMBDA_COST * cost
            # Приоритет обязательных слотов
            if slot in REQUIRED_SLOTS:
                score += 0.15
            if score > best_score:
                best_score = score
                best_slot = slot

        if best_slot is None:
            return None
        return SLOT_QUESTIONS.get(best_slot)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slots": deepcopy(self.slots),
            "goals_text": self.goals_text,
            "interests": list(self.interests),
            "turn_count": self.turn_count,
            "entropy": round(self.get_entropy(), 4),
            "conflicts": self.get_conflicts(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeliefState":
        state = cls()
        if not data:
            return state
        state.slots = deepcopy(data.get("slots", {}))
        state.goals_text = data.get("goals_text", "") or ""
        state.interests = list(data.get("interests", []))
        state.turn_count = int(data.get("turn_count", 0))
        return state

    def to_profile_dict(self) -> Dict[str, Any]:
        """Маппинг belief → поля RecommendationRequest."""
        profile: Dict[str, Any] = {
            "interests": self.interests,
            "goals": self.goals_text,
        }
        for slot in SLOT_TERM_VOCAB:
            entry = self.slots.get(slot)
            if not entry:
                continue
            if "numeric" in entry:
                profile[slot] = entry["numeric"]
            else:
                terms_map = entry.get("terms", {})
                term = max(terms_map, key=terms_map.get) if terms_map else None
                if term:
                    num = term_to_numeric(slot, term)
                    if num is not None:
                        profile[slot] = num
        return profile


def apply_nlu_to_belief(
    belief: BeliefState, entities: Dict[str, Any], default_conf: float = 0.75
) -> BeliefState:
    """Применяет извлечённые сущности NLU к belief state."""
    for slot, payload in entities.items():
        if slot == "goals":
            belief.update("goals", payload.get("value", ""), payload.get("confidence", default_conf))
        elif slot == "interests":
            belief.update("interests", payload.get("value", []), payload.get("confidence", default_conf))
        else:
            value = payload.get("value")
            conf = float(payload.get("confidence", default_conf))
            if value is not None:
                belief.update(slot, value, conf)
    return belief


def process_dialog_turn(
    belief: BeliefState,
    user_message: str,
    nlu_result: Dict[str, Any],
) -> Tuple[BeliefState, Optional[str], bool, Dict[str, Any]]:
    """
    Один шаг диалога.
    Возвращает: (updated_belief, next_question, is_ready, meta).
    """
    belief = BeliefState.from_dict(belief.to_dict()) if belief else BeliefState()

    intent = nlu_result.get("intent", "unknown")
    entities = nlu_result.get("entities", {})
    parse_conf = float(nlu_result.get("confidence", 0.5))

    if entities:
        apply_nlu_to_belief(belief, entities, default_conf=parse_conf)
    elif user_message.strip() and intent != "request_recommendation":
        # Свободный текст без сущностей → цель обучения
        belief.update("goals", user_message.strip(), min(0.6, parse_conf))

    ready = belief.is_ready_for_recommend()
    if intent == "request_recommendation" and belief.is_slot_filled("budget"):
        ready = True

    question = None if ready else belief.next_question()

    meta = {
        "intent": intent,
        "parse_confidence": parse_conf,
        "entropy": round(belief.get_entropy(), 4),
        "conflicts": belief.get_conflicts(),
    }
    return belief, question, ready, meta
