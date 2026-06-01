"""Маршруты API: рекомендации и health check."""
import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    DialogStepRequest,
    DialogStepResponse,
    HealthResponse,
    RecommendationMeta,
    RecommendationRequest,
    RecommendationResponse,
)
from app.core.session_store import session_store
from app.dependencies import TEMPLATES_DIR, get_recommender
from app.services.dialog_manager import BeliefState, process_dialog_turn
from app.services.nlu_parser import nlu_parser
from app.user_profile import UserProfile

logger = logging.getLogger(__name__)

router = APIRouter(tags=["api"])


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend_api(req: RecommendationRequest):
    """JSON API для гибридных рекомендаций."""
    try:
        rec = get_recommender()
        profile = UserProfile(
            budget=req.budget,
            knowledge_level=req.knowledge_level,
            time_availability=req.time_availability,
            career_focus=req.career_focus,
            interests=req.interests,
            goals=req.goals,
        )
        recommendations = rec.recommend(profile, top_k=5)
        explanations = {c["id"]: rec.explain(profile, c) for c in recommendations}
        return RecommendationResponse(
            recommendations=recommendations,
            explanations=explanations,
            meta=RecommendationMeta(
                fuzzy_rules_count=5,
                total_courses_in_db=len(rec.courses),
            ),
        )
    except FileNotFoundError as e:
        logger.error("Файл курсов не найден: %s", e)
        raise HTTPException(
            status_code=500,
            detail="База курсов недоступна. Проверьте data/unified_courses.json",
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ошибка в /api/recommend: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось сформировать рекомендации: {e}",
        ) from e


@router.post("/dialog/step", response_model=DialogStepResponse)
async def dialog_step(req: DialogStepRequest):
    """Один шаг диалога: NLU → обновление belief → следующий вопрос."""
    try:
        session_id, belief = session_store.get_or_create(
            req.session_id, req.prev_belief
        )
        nlu_result = nlu_parser.parse(req.user_message)
        updated, next_q, ready, meta = process_dialog_turn(
            belief, req.user_message, nlu_result
        )
        session_store.save(session_id, updated)

        return DialogStepResponse(
            session_id=session_id,
            next_question=next_q,
            updated_belief=updated.to_dict(),
            is_ready_for_recommend=ready,
            profile_preview=updated.to_profile_dict(),
            meta=meta,
        )
    except Exception as e:
        logger.error("Ошибка в /api/dialog/step: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка диалогового шага: {e}",
        ) from e


@router.get("/health", response_model=HealthResponse)
async def health():
    """Проверка доступности сервиса и ключевых компонентов."""
    return HealthResponse(
        status="ok",
        components=["fuzzy_engine", "recommender", "course_loader"],
        templates_path=str(TEMPLATES_DIR),
        templates_exists=TEMPLATES_DIR.exists(),
    )
