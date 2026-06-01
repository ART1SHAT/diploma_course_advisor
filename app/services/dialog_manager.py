"""
Оркестрация шага диалога (§2.4): NLU → belief → политика.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from app.services.belief_state import BeliefState
from app.services.dialog_policy import select_next_action
from app.services.nlu_parser import parse_user_message

# Реэкспорт для обратной совместимости тестов и session_store
__all__ = [
    "BeliefState",
    "apply_nlu_to_belief",
    "process_dialog_turn",
]


def apply_nlu_to_belief(
    belief: BeliefState,
    nlu_result: Dict[str, Any],
    default_conf: float = 0.75,
) -> BeliefState:
    """Применяет сущности из parse_user_message или legacy parse()."""
    entities = nlu_result.get("entities", {})
    for slot, payload in entities.items():
        if slot == "goals":
            text = payload.get("term") or payload.get("value", "")
            conf = float(payload.get("confidence", default_conf))
            belief.update("goals", str(text), 0.0, conf)
            continue
        term = payload.get("term", "")
        value = payload.get("value", 0.0)
        conf = float(payload.get("confidence", default_conf))
        if not term and "value" in payload:
            belief.update(slot, str(payload["value"]), float(payload["value"]), conf)
        else:
            belief.update(slot, str(term), float(value) if value is not None else 0.0, conf)
    return belief


def process_dialog_turn(
    belief: BeliefState,
    user_message: str,
    nlu_result: Optional[Dict[str, Any]] = None,
) -> Tuple[BeliefState, Optional[str], bool, Dict[str, Any]]:
    """
    Один шаг диалога.
    Возвращает: (belief, next_question, is_ready, meta).
    """
    belief = BeliefState.from_dict(belief.to_dict()) if belief else BeliefState()

    if nlu_result is None:
        nlu_result = parse_user_message(user_message)
    elif "raw_text" not in nlu_result:
        nlu_result = parse_user_message(user_message)

    apply_nlu_to_belief(belief, nlu_result)

    action = select_next_action(belief, nlu_result)
    ready = action["action"] == "show_recommendations"
    question = action.get("question") if action["action"] == "ask_question" else None

    meta = {
        "intent": nlu_result.get("intent"),
        "action": action["action"],
        "slot": action.get("slot"),
        "entropy": round(
            sum(belief.get_entropy(s) for s in belief.get_uncertain_slots(1.0) or ["budget"])
            / max(1, len(belief.belief_summary()) or 1),
            4,
        ),
    }
    return belief, question, ready, meta
