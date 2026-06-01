"""
Точка сборки FastAPI-приложения CourseAdvisor.
Запуск через demo_mvp.py или: uvicorn app.main:app
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.api.routes import router as api_router
from app.dependencies import TEMPLATES_DIR

# Экспорт для demo_mvp и тестов
__all__ = ["app", "create_app", "TEMPLATES_DIR"]


def create_app() -> FastAPI:
    """Фабрика приложения: маршруты, шаблоны, обработчики ошибок."""
    application = FastAPI(title="CourseAdvisor MVP", version="1.0.0")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """422 — понятное сообщение при неверном JSON/полях."""
        errors = exc.errors()
        fields = ", ".join(
            ".".join(str(p) for p in err.get("loc", ()) if p != "body")
            for err in errors
        )
        message = (
            f"Некорректные данные запроса ({fields})."
            if fields
            else "Некорректные данные запроса."
        )
        return JSONResponse(
            status_code=422,
            content={"detail": message, "errors": errors},
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """500 — необработанные исключения вне HTTPException."""
        if isinstance(exc, HTTPException):
            raise exc
        return JSONResponse(
            status_code=500,
            content={"detail": f"Внутренняя ошибка сервера: {exc}"},
        )

    @application.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {"request": request},
        )

    application.include_router(api_router, prefix="/api")

    return application


app = create_app()
