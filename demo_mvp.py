# demo_mvp.py (финальная версия)
"""
🎓 CourseAdvisor — MVP для предзащиты
Работает с courses_processed.csv из репозитория
Запуск: python demo_mvp.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import json

from app.fuzzy_engine import FuzzyInferenceEngine
from app.user_profile import UserProfile
from app.recommender import HybridRecommender

app = FastAPI(title="CourseAdvisor MVP", version="0.2.0")

# Инициализация (ленивая загрузка)
_recommender = None

def get_recommender():
    global _recommender
    if _recommender is None:
        print("🔄 Загрузка курсов из courses_processed.csv...")
        _recommender = HybridRecommender(csv_path="data/courses_processed.csv")
        print(f"✅ Загружено {_recommender.loader.load().shape[0]} курсов")
    return _recommender

class RecommendationRequest(BaseModel):
    budget: Optional[float] = Field(default=None, ge=0, description="Бюджет в рублях")
    knowledge_level: Optional[float] = Field(default=None, ge=0, le=10, description="Уровень знаний 0-10")
    time_availability: Optional[float] = Field(default=None, ge=0, description="Часов в неделю")
    career_focus: Optional[float] = Field(default=None, ge=0, le=1, description="0=personal, 0.5=academic, 1=employment")
    interests: List[str] = Field(default_factory=list, description="Список интересов")
    goals: str = Field(default="", description="Основная цель обучения")
    preferred_language: str = Field(default="ru", description="Предпочтительный язык курса")

@app.post("/api/recommend")
async def recommend(req: RecommendationRequest):
    """Основной эндпоинт рекомендаций"""
    try:
        recommender = get_recommender()
        
        profile = UserProfile(
            budget=req.budget,
            knowledge_level=req.knowledge_level,
            time_availability=req.time_availability,
            career_focus=req.career_focus,
            interests=req.interests,
            goals=req.goals
        )
        
        recommendations = recommender.recommend(profile, top_k=5)
        explanations = {
            c["id"]: recommender.explain(profile, c) 
            for c in recommendations
        }
        
        return {
            "recommendations": recommendations,
            "explanations": explanations,
            "meta": {
                "fuzzy_rules_count": len(FuzzyInferenceEngine().rules),
                "profile_params_filled": len([v for v in profile.to_fuzzy_input().values() if v is not None]),
                "total_courses_in_db": recommender.loader.load().shape[0]
            }
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/courses/stats")
async def get_course_stats():
    """Статистика по базе курсов"""
    try:
        recommender = get_recommender()
        return recommender.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {
        "status": "ok", 
        "components": ["fuzzy_engine", "recommender", "course_loader", "explanations"],
        "data_source": "courses_processed.csv"
    }

# Статика
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/demo")
async def demo_page():
    return FileResponse("frontend/index.html")

@app.get("/")
async def root():
    return {
        "message": "🎓 CourseAdvisor MVP",
        "docs": "/docs", 
        "demo": "/demo",
        "health": "/api/health",
        "stats": "/api/courses/stats"
    }

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════╗
    ║  🎓 CourseAdvisor — MVP для предзащиты v0.2    ║
    ╠════════════════════════════════════════════════╣
    ║  📊 Данные: courses_processed.csv (~4400 курсов)║
    ║  🧠 Движок: нечёткая логика + семантический    ║
    ║                                                ║
    ║  🔗 Откройте: http://localhost:8000/demo       ║
    ║                                                ║
    ║  📋 Сценарий демо:                             ║
    ║  1. Бюджет: 50000 ₽                            ║
    ║  2. Уровень: 3/10 (новичок)                    ║
    ║  3. Время: 5 ч/нед                             ║
    ║  4. Цель: трудоустройство                      ║
    ║  5. Интересы: "анализ данных, python"          ║
    ╚════════════════════════════════════════════════╝
    """)
    uvicorn.run("demo_mvp:app", host="0.0.0.0", port=8000, reload=False)