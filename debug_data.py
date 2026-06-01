# debug_data.py
import pandas as pd
import json
from pathlib import Path

print("🔍 Диагностика данных")
print("=" * 50)

# 1. Проверяем исходный CSV
csv_path = Path("data/courses_processed.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    print(f"📄 courses_processed.csv: {len(df)} строк")
    print(f"   Колонки: {list(df.columns)}")
else:
    print("❌ courses_processed.csv не найден")

# 2. Проверяем unified JSON
json_path = Path("data/unified_courses.json")
if json_path.exists():
    with open(json_path, "r", encoding="utf-8") as f:
        courses = json.load(f)
    print(f"📦 unified_courses.json: {len(courses)} курсов")
    
    # Группировка по источнику
    providers = {}
    for c in courses:
        p = c.get("provider", "unknown")
        providers[p] = providers.get(p, 0) + 1
    print(f"   По источникам: {providers}")
    
    # Пример первой записи
    if courses:
        print(f"   Пример ID: {courses[0].get('id')}, title: {courses[0].get('title')[:50]}...")
else:
    print("❌ unified_courses.json не найден")

# 3. Проверяем, что загружает рекомендатель
print("\n🔄 Проверка CourseLoader...")
from app.course_loader import CourseLoader
loader = CourseLoader()
loaded = loader.load()
print(f"✅ CourseLoader загрузил: {len(loaded)} курсов")