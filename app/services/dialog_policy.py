"""
Политика диалога (§2.4): выбор следующего действия по EIG и порогу уверенности.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.services.belief_state import BeliefState, REQUIRED_SLOTS, SLOT_TERM_VOCAB

CONFIDENCE_THRESHOLD = 0.7

SLOT_QUESTIONS: Dict[str, str] = {
    "budget": "Какой у вас бюджет на обучение (в рублях)?",
    "career_focus": "Какова главная цель: трудоустройство, учёба или личное развитие?",
    "knowledge_level": "Оцените текущий уровень знаний по шкале от 0 (новичок) до 10.",
    "time_availability": "Сколько часов в неделю готовы уделять обучению?",
    "goals": "Сформулируйте ключевую цель обучения своими словами.",
}


def expected_information_gain(belief: BeliefState, slot: str) -> float:
    """Упрощённый EIG для MVP: 1 − текущая уверенность по слоту."""
    return 1.0 - belief.get_slot_confidence(slot)


def _required_uncertain(belief: BeliefState) -> List[str]:
    return [
        s for s in REQUIRED_SLOTS
        if belief.get_slot_confidence(s) < CONFIDENCE_THRESHOLD
    ]


def _optional_uncertain(belief: BeliefState) -> List[str]:
    optional: List[str] = []
    for slot in SLOT_TERM_VOCAB:
        if slot in REQUIRED_SLOTS:
            continue
        if belief.get_slot_confidence(slot) < CONFIDENCE_THRESHOLD:
            optional.append(slot)
    if belief.get_slot_confidence("goals") < CONFIDENCE_THRESHOLD and not belief.goals_text:
        optional.append("goals")
    return optional


def select_next_action(
    belief: BeliefState,
    nlu_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Выбор действия диалога.
    Возвращает: action, question?, slot?
    """
    intent = nlu_result.get("intent", "unknown")

    if intent == "ask_recommend" and not _required_uncertain(belief):
        return {"action": "show_recommendations", "question": None, "slot": None}

    required_uncertain = _required_uncertain(belief)
    if not required_uncertain:
        # Обязательные слоты заполнены — можно рекомендовать
        return {"action": "show_recommendations", "question": None, "slot": None}

    slot_candidates = list(required_uncertain)
    best_slot = max(slot_candidates, key=lambda s: expected_information_gain(belief, s))
    question = SLOT_QUESTIONS.get(best_slot, f"Уточните значение: {best_slot}")
    return {
        "action": "ask_question",
        "question": question,
        "slot": best_slot,
    }
