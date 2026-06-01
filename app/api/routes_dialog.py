"""Маршруты диалогового взаимодействия (§2.4)."""
import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.api.schemas import DialogStartResponse, DialogStepRequestV2, DialogStepResponseV2
from app.core.session_store import session_store
from app.dependencies import get_recommender
from app.services.belief_state import BeliefState
from app.services.dialog_manager import apply_nlu_to_belief
from app.services.dialog_policy import select_next_action
from app.services.nlu_parser import parse_user_message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dialog", tags=["dialog"])


def _belief_summary_dict(belief: BeliefState) -> Dict[str, Any]:
    return belief.belief_summary()


@router.post("/start", response_model=DialogStartResponse)
async def dialog_start():
    """Создаёт сессию диалога и возвращает первый уточняющий вопрос."""
    try:
        session_id = session_store.create_session(BeliefState())
        rec = session_store.get(session_id)
        if rec is None:
            raise HTTPException(status_code=500, detail="Не удалось создать сессию")
        action = select_next_action(rec.belief, {"intent": "unknown", "entities": {}})
        question = action.get("question") if action["action"] == "ask_question" else None
        return DialogStartResponse(session_id=session_id, initial_question=question)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ошибка /api/dialog/start: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/step", response_model=DialogStepResponseV2)
async def dialog_step(req: DialogStepRequestV2):
    """Шаг диалога: NLU → belief → политика → вопрос или рекомендации."""
    try:
        record = session_store.get(req.session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена или истекла")

        belief = record.belief
        nlu_result = parse_user_message(req.user_message)
        apply_nlu_to_belief(belief, nlu_result)
        action = select_next_action(belief, nlu_result)

        recommendations = None
        next_action = action["action"]
        is_ready = next_action == "show_recommendations"

        if is_ready:
            rec_engine = get_recommender()
            recommendations = rec_engine.recommend_from_belief(belief, top_k=5)

        session_store.save(req.session_id, belief)

        return DialogStepResponseV2(
            session_id=req.session_id,
            next_action=next_action,
            question=action.get("question"),
            recommendations=recommendations,
            belief_summary=_belief_summary_dict(belief),
            is_ready=is_ready,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ошибка /api/dialog/step: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
