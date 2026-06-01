"""Маршруты API: рекомендации и health check."""
import logging

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import (
    GraphResponse,
    HealthResponse,
    RecommendationMeta,
    RecommendationRequest,
    RecommendationResponse,
    WhatIfRequest,
    WhatIfResponse,
)
from app.dependencies import TEMPLATES_DIR, get_recommender
from app.services.explainer import (
    apply_profile_changes,
    build_what_if_explanation,
    find_course_rank,
    profile_to_dict,
)
from app.services.competency_graph import get_competency_graph
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


def _request_to_profile(req: RecommendationRequest) -> UserProfile:
    return UserProfile(
        budget=req.budget,
        knowledge_level=req.knowledge_level,
        time_availability=req.time_availability,
        career_focus=req.career_focus,
        interests=req.interests,
        goals=req.goals,
    )


@router.post("/explain/what_if", response_model=WhatIfResponse)
async def explain_what_if(req: WhatIfRequest):
    """Контрфактуальный анализ: как изменится ранг курса при изменении профиля."""
    try:
        if not req.changed_profile:
            raise HTTPException(
                status_code=422,
                detail="Укажите changed_profile с хотя бы одним полем",
            )

        rec = get_recommender()
        base = _request_to_profile(req.base_profile)
        modified = apply_profile_changes(base, req.changed_profile)

        base_ranked = rec.score_all(base)
        new_ranked = rec.score_all(modified)

        old_rank, _ = find_course_rank(base_ranked, req.course_id)
        new_rank, _ = find_course_rank(new_ranked, req.course_id)

        if old_rank == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Курс {req.course_id} не найден в рейтинге базового профиля",
            )

        delta_rank = old_rank - new_rank
        field = next(iter(req.changed_profile))
        old_val = profile_to_dict(base).get(field)
        new_val = req.changed_profile[field]

        old_score = next(
            (c["score"] for c in base_ranked if str(c["id"]) == str(req.course_id)),
            None,
        )
        new_score = next(
            (c["score"] for c in new_ranked if str(c["id"]) == str(req.course_id)),
            None,
        )

        explanation = build_what_if_explanation(
            field, old_val, new_val, delta_rank
        )

        return WhatIfResponse(
            course_id=req.course_id,
            old_rank=old_rank,
            new_rank=new_rank or old_rank,
            delta_rank=delta_rank,
            old_score=old_score,
            new_score=new_score,
            explanation=explanation,
            changed_fields=req.changed_profile,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ошибка в /api/explain/what_if: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Контрфактуальный анализ не выполнен: {e}",
        ) from e


@router.get("/graph/{course_id}", response_model=GraphResponse)
async def get_course_competency_graph(
    course_id: str,
    profession: Optional[str] = Query(
        default=None,
        description="Целевая профессия для пути (по умолчанию — первая из онтологии курса)",
    ),
):
    """Подграф компетенций для курса: узлы, рёбра и цепочка объяснения (§2.1)."""
    try:
        graph = get_competency_graph()
        if not graph.course_exists(course_id):
            raise HTTPException(
                status_code=404,
                detail=f"Курс {course_id} не найден в онтологии графа компетенций",
            )
        payload = graph.build_api_payload(course_id, profession=profession)
        return GraphResponse(**payload)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error("Ошибка в /api/graph/%s: %s", course_id, e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось построить граф компетенций: {e}",
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
