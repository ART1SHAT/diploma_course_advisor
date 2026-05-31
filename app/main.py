from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.recommender import CourseRecommender

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

print("Loading recommender...")
recommender = CourseRecommender()
print("Recommender loaded")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "results": None
        }
    )


@app.post("/recommend", response_class=HTMLResponse)
async def recommend(
    request: Request,
    query: str = Form(...),
    top_k: int = Form(5)
):

    results = recommender.recommend(
        query=query,
        top_k=top_k
    )

    courses = results.to_dict(
        orient="records"
    )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "results": courses,
            "query": query
        }
    )