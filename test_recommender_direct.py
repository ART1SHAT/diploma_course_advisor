# test_recommender_direct.py — исправленная версия
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO)

from app.user_profile import UserProfile
from app.recommender import HybridRecommender

# Создаём профиль
profile = UserProfile(
    budget=50000,
    knowledge_level=3,
    time_availability=5,
    career_focus=1.0,
    interests=["анализ данных", "python"],
    goals="получить практические навыки"
)

# Инициализируем рекомендатель
print("🔄 Инициализация HybridRecommender...")
rec = HybridRecommender(json_path="data/unified_courses.json")
print(f"✅ Загружено {len(rec.courses)} курсов")

# Проверка индекса (без is_ready())
if hasattr(rec, 'sem_index') and rec.sem_index is not None:
    if rec.sem_index.vectors is not None and rec.sem_index.vectors.size > 0:
        print(f"✅ Семантический индекс готов: {rec.sem_index.vectors.shape}")
    else:
        print("⚠️  Семантический индекс: используем fallback")
else:
    print("⚠️  Семантический поиск отключён")

# Тестируем рекомендацию
print("🔍 Генерация рекомендаций...")
try:
    results = rec.recommend(profile, top_k=3)
    print(f"✅ Найдено {len(results)} рекомендаций:")
    for i, c in enumerate(results, 1):
        title = c.get('title', 'Без названия')[:60]
        score = c.get('score', 0)
        print(f"  {i}. {title}... (score={score})")
    
    # Тестируем объяснение
    if results:
        print("\n📝 Объяснение для первой рекомендации:")
        exp = rec.explain(profile, results[0])
        for e in exp.get("explanations", []):
            print(f"  • {e}")
            
    print("\n🎉 ВСЁ РАБОТАЕТ! Система готова к демонстрации.")
    
except Exception as e:
    print(f"❌ Ошибка при генерации рекомендаций: {e}")
    import traceback
    traceback.print_exc()