# data/unify_pipeline.py — исправленная версия для работы с courses_processed.csv
import pandas as pd
import json
import re
from pathlib import Path
from data.models import UnifiedCourse, ProviderType, CourseFormat, CertificationType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def map_stepik_to_unified(df_stepik: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализует Stepik-данные под UnifiedCourse.
    Ожидаемые колонки: id, title, description, price, language, url, full_text, category
    """
    records = []
    skipped = 0
    
    for idx, row in df_stepik.iterrows():
        try:
            # Безопасное извлечение с дефолтами
            def get_str(key, default=""):
                val = row.get(key)
                return str(val).strip() if pd.notna(val) else default
            
            def get_float(key, default=0.0):
                val = row.get(key)
                try:
                    return float(val) if pd.notna(val) else default
                except (ValueError, TypeError):
                    return default
            
            course_id = get_str('id', f'stepik_{idx}')
            title = get_str('title', 'Без названия')
            description = get_str('description', '')
            price = get_float('price', 0)
            language = get_str('language', 'ru')
            url = get_str('url', '')
            category = get_str('category', 'Other')
            
            # Определение уровня по описанию
            desc_lower = description.lower()
            if any(w in desc_lower for w in ['продвинут', 'advanced', 'senior', 'expert']):
                level = ["advanced"]
            elif any(w in desc_lower for w in ['средн', 'intermediate', 'опыт']):
                level = ["intermediate"]
            else:
                level = ["beginner"]
            
            # Сертификация
            certification = "professional" if "сертификат" in desc_lower or "certificate" in desc_lower.lower() else "none"
            
            # Компетенции: категория + извлечение из описания
            competencies = []
            if category and category != 'Other':
                competencies.append(category.lower())
            
            # Простое извлечение навыков по ключевым словам
            skill_map = {
                'python': ['python', 'django', 'flask'],
                'sql': ['sql', 'postgresql', 'mysql', 'база данных'],
                'анализ данных': ['анализ данных', 'pandas', 'numpy', 'визуализация'],
                'веб-разработка': ['веб', 'html', 'css', 'javascript', 'frontend'],
                'машинное обучение': ['машинное обучение', 'ml', 'нейросеть', 'scikit'],
            }
            for skill, keywords in skill_map.items():
                if any(kw in desc_lower for kw in keywords) and skill not in competencies:
                    competencies.append(skill)
            
            course = UnifiedCourse(
                id=f"stepik_{course_id}",
                title=title,
                provider="Stepik",
                provider_type=ProviderType.MOOC,
                price=max(0.0, price),
                duration_weeks=None,  # нет данных в исходном CSV
                format=CourseFormat.ONLINE,
                language=language if language in ['ru', 'en'] else 'ru',
                level=level,
                competencies=list(set(competencies)),
                certification=CertificationType(certification),
                url=url if url.startswith('http') else '',
                description=description[:4999],  # ограничение длины
                source_quality=1.0
            )
            records.append(course.dict())
            
        except Exception as e:
            skipped += 1
            if skipped <= 5:  # логируем только первые 5 ошибок
                logger.warning(f"⚠️ Пропущена строка {idx}: {e}")
            continue
    
    logger.info(f"✅ Обработано: {len(records)} курсов, пропущено: {skipped}")
    return pd.DataFrame(records)

def run_unify_pipeline(output_path: str = "data/unified_courses.json"):
    base_dir = Path(__file__).parent.parent
    stepik_csv = base_dir / "data" / "courses_processed.csv"
    
    if not stepik_csv.exists():
        logger.error(f"❌ Не найден файл: {stepik_csv}")
        return []
    
    logger.info("📥 Загрузка данных Stepik...")
    df_stepik = pd.read_csv(stepik_csv, on_bad_lines="skip")
    logger.info(f"   Исходно: {len(df_stepik)} строк")
    
    logger.info("🔄 Нормализация к единой схеме...")
    df_unified = map_stepik_to_unified(df_stepik)
    
    if df_unified.empty:
        logger.error("❌ Не удалось преобразовать ни одной записи")
        return []
    
    # Сохранение
    courses_list = df_unified.to_dict(orient='records')
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(courses_list, f, ensure_ascii=False, indent=2)
        
    logger.info(f"✅ Сохранено {len(courses_list)} курсов в {output_path}")
    return courses_list

if __name__ == "__main__":
    run_unify_pipeline()