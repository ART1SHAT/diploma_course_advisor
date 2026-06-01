# app/main.py — исправленная версия
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Optional

from app.recommender import HybridRecommender
from app.user_profile import UserProfile

app = FastAPI(title="CourseAdvisor", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Инициализация рекомендателя (ленивая загрузка)
_recommender = None

def get_recommender():
    global _recommender
    if _recommender is None:
        print("🔄 Загрузка курсов...")
        _recommender = HybridRecommender(csv_path="data/courses_processed.csv")
        print(f"✅ Загружено {_recommender.loader.load().shape[0]} курсов")
    return _recommender

class RecommendationRequest(BaseModel):
    budget: Optional[float] = Field(default=None, ge=0)
    knowledge_level: Optional[float] = Field(default=None, ge=0, le=10)
    time_availability: Optional[float] = Field(default=None, ge=0)
    career_focus: Optional[float] = Field(default=None, ge=0, le=1)
    interests: List[str] = Field(default_factory=list)
    goals: str = Field(default="")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/recommend")
async def recommend_api(req: RecommendationRequest):
    """JSON API для рекомендаций"""
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
        explanations = {c["id"]: rec.explain(profile, c) for c in recommendations}
        return {
            "recommendations": recommendations,
            "explanations": explanations,
            "meta": {
                "fuzzy_rules_count": 5,
                "total_courses_in_db": rec.loader.load().shape[0]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "ok", "components": ["fuzzy_engine", "recommender", "course_loader"]}

# Статика
app.mount("/static", StaticFiles(directory="templates"), name="static")

if __name__ == "__main__":
    import uvicorn
    print("🎓 CourseAdvisor запущен: http://localhost:8000")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)