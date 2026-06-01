# demo_mvp.py — тонкая обёртка для запуска MVP
"""
🎓 CourseAdvisor MVP
Запуск: python demo_mvp.py
Открыть: http://localhost:8000
"""

import uvicorn

from app.dependencies import TEMPLATES_DIR
from app.main import create_app

app = create_app()

if __name__ == "__main__":
    print(
        """
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
    """
    )

    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        print(f"⚠️  ВНИМАНИЕ: {index_path} не найден!")
        print(f"   Убедитесь, что файл index.html лежит в папке: {TEMPLATES_DIR}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
