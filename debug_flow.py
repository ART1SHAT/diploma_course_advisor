# debug_flow.py
import sys, json, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO)

from app.course_loader import CourseLoader
from app.semantic_search import SemanticIndex

# Загрузка курсов
loader = CourseLoader("data/unified_courses.json")
courses = loader.load()
print(f"✅ Загружено {len(courses)} курсов")
print(f"✅ Тип courses[0]: {type(courses[0])}")

# Построение индекса
idx = SemanticIndex()
idx.build(courses)

print(f"✅ self.vectors тип: {type(idx.vectors)}")
if idx.vectors is not None:
    print(f"✅ self.vectors.shape: {idx.vectors.shape}")
    print(f"✅ self.vectors.size: {idx.vectors.size}")

# Тест поиска
results = idx.search("python анализ данных", top_k=3)
print(f"✅ Поиск вернул {len(results)} результатов")
for c, score in results:
    print(f"  - {c.get('title')}: {score:.3f}")