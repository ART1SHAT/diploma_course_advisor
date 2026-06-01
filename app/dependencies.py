"""Зависимости приложения: ленивая инициализация рекомендателя."""
import os
from pathlib import Path

from app.recommender import HybridRecommender

# Корень проекта (родитель каталога app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
COURSES_JSON = PROJECT_ROOT / "data" / "unified_courses.json"

_recommender: HybridRecommender | None = None


def get_recommender() -> HybridRecommender:
    """Создаёт и кэширует HybridRecommender при первом обращении."""
    global _recommender
    if _recommender is None:
        json_path = str(COURSES_JSON)
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Не найден файл курсов: {json_path}")
        print(f"🔄 Загрузка курсов из {json_path}...")
        _recommender = HybridRecommender(json_path=json_path)
        print(f"✅ Загружено {len(_recommender.courses)} курсов")
    return _recommender
