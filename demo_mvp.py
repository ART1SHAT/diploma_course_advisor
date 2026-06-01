# demo_mvp.py — Единая точка входа для предзащиты
"""
🎓 CourseAdvisor MVP
Запуск: python demo_mvp.py
Открыть: http://localhost:8000
"""

import uvicorn
import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Optional

# Импорты наших модулей
from app.fuzzy_engine import FuzzyInferenceEngine
from app.user_profile import UserProfile
from app.recommender import HybridRecommender

app = FastAPI(title="CourseAdvisor MVP", version="1.0.0")

# Пути
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

# Инициализация Jinja2
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Ленивая инициализация рекомендателя
_recommender = None

# demo_mvp.py — исправленная инициализация
_recommender = None

def get_recommender():
    global _recommender
    if _recommender is None:
        # Проверяем доступность файлов
        json_path = "data/unified_courses.json"
        if os.path.exists(json_path):
            print(f"🔄 Загрузка курсов из {json_path}...")
            _recommender = HybridRecommender(json_path=json_path)
        else:
            raise FileNotFoundError(f"❌ Не найден {json_path}")
        print(f"✅ Загружено {len(_recommender.courses)} курсов")
    return _recommender

# Модель запроса
class RecommendationRequest(BaseModel):
    budget: Optional[float] = Field(default=None, ge=0)
    knowledge_level: Optional[float] = Field(default=None, ge=0, le=10)
    time_availability: Optional[float] = Field(default=None, ge=0)
    career_focus: Optional[float] = Field(default=None, ge=0, le=1)
    interests: List[str] = Field(default_factory=list)
    goals: str = Field(default="")

# 🏠 Главная страница — отдаёт index.html
# ✅ Стало (новый стиль — 3 варианта на выбор):

# Вариант 1: позиционные аргументы (самый короткий)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

# Вариант 2: именованные аргументы (самый понятный)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html", 
        context={"request": request}
    )

# Вариант 3: универсальный (работает в старых и новых версиях)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        name="index.html",
        context={"request": request},
        request=request  # явная передача request как именованного аргумента
    )

# 🔍 API рекомендаций
@app.post("/api/recommend")
async def recommend_api(req: RecommendationRequest):
    try:
        rec = get_recommender()
        profile = UserProfile(
            budget=req.budget,
            knowledge_level=req.knowledge_level,
            time_availability=req.time_availability,
            career_focus=req.career_focus,
            interests=req.interests,
            goals=req.goals
        )
        recommendations = rec.recommend(profile, top_k=5)
        explanations = {
            c["id"]: rec.explain(profile, c) 
            for c in recommendations
        }
        return {
            "recommendations": recommendations,
            "explanations": explanations,
            "meta": {
                "fuzzy_rules_count": 5,
                "total_courses_in_db": len(rec.courses)
            }
        }
    except Exception as e:
        logging.error(f"❌ Error in /api/recommend: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 🩺 Health check
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "components": ["fuzzy_engine", "recommender", "course_loader"],
        "templates_path": str(TEMPLATES_DIR),
        "templates_exists": TEMPLATES_DIR.exists()
    }

# 📊 Статистика курсов
@app.get("/api/courses/stats")
async def course_stats():
    try:
        rec = get_recommender()
        return rec.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════╗
    ║  🎓 CourseAdvisor — MVP для предзащиты         ║
    ╠════════════════════════════════════════════════╣
    ║  ✅ Сервер запущен                              ║
    ║  🔗 Откройте: http://localhost:8000            ║
    ║                                                ║
    ║  📋 Проверка перед демо:                       ║
    ║  1. /api/health → статус компонентов          ║
    ║  2. / → веб-интерфейс                          ║
    ║  3. Заполнить профиль → "Найти программы"      ║
    ╚════════════════════════════════════════════════╝
    """)
    
    # Проверка наличия templates/index.html
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        print(f"⚠️  ВНИМАНИЕ: {index_path} не найден!")
        print(f"   Убедитесь, что файл index.html лежит в папке: {TEMPLATES_DIR}")
    
    uvicorn.run(
        "demo_mvp:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )