# data/unify_pipeline.py
import pandas as pd
import json
from pathlib import Path
from data.models import UnifiedCourse
from data.adapters.sdo_adapter import SDOAdapter

def map_stepik_to_unified(df_stepik: pd.DataFrame) -> pd.DataFrame:
    """Нормализует Stepik-данные"""
    records = []
    for _, row in df_stepik.iterrows():
        price = float(row.get("price", 0) or 0)
        desc = str(row.get("description", "") or "")
        cat = str(row.get("category", "Other") or "Other")
        
        level = ["beginner"]
        if any(w in desc.lower() for w in ["продвинут", "advanced"]):
            level = ["advanced"]
        elif any(w in desc.lower() for w in ["средн", "intermediate"]):
            level = ["intermediate"]
            
        records.append({
            "id": f"stepik_{int(row.get('course_id', 0))}",
            "title": str(row.get("title", "Без названия")),
            "provider": "Stepik",
            "provider_type": "mooc",
            "price": max(0.0, price),
            "duration_weeks": None,
            "format": "online",
            "language": str(row.get("language", "ru") or "ru"),
            "level": level,
            "competencies": [cat] if cat != "Other" else [],
            "certification": "professional" if "сертификат" in desc.lower() else "none",
            "url": str(row.get("url", "") or ""),
            "description": desc,
            "source_quality": 1.0
        })
    return pd.DataFrame(records)

def run_unify_pipeline(output_path: str = "data/unified_courses.json"):
    base_dir = Path(__file__).parent.parent
    stepik_csv = base_dir / "data" / "courses_processed.csv"
    
    print("📥 Этап 1: Загрузка данных Stepik...")
    try:
        df_stepik = pd.read_csv(stepik_csv, on_bad_lines="skip")
        df_unified = map_stepik_to_unified(df_stepik.head(400))
        print(f"  ✅ Stepik: {len(df_unified)} курсов")
    except Exception as e:
        print(f"  ⚠️ Ошибка Stepik: {e}")
        df_unified = pd.DataFrame()

    print("📥 Этап 2: Сбор данных с СЦОС...")
    try:
        adapter = SDOAdapter()
        sdo_courses = adapter.run(save_path="data/sdo_courses.json")
        df_sdo = pd.DataFrame([c.dict() for c in sdo_courses])
        print(f"  ✅ СЦОС: {len(df_sdo)} курсов")
    except Exception as e:
        print(f"  ⚠️ Ошибка СЦОС: {e}")
        df_sdo = pd.DataFrame()

    print("🔗 Этап 3: Слияние и валидация...")
    frames = [f for f in [df_unified, df_sdo] if not f.empty]
    if not frames:
        print("❌ Нет данных для слияния. Проверьте CSV и доступ к интернету.")
        return []

    df_all = pd.concat(frames, ignore_index=True)
    
    # Финальная валидация
    validated = []
    for _, row in df_all.iterrows():
        try:
            # Очистка от NaN перед валидацией
            clean_row = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
            validated.append(UnifiedCourse(**clean_row).dict())
        except Exception as e:
            pass
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(validated, f, ensure_ascii=False, indent=2)
        
    print(f"✅ ИТОГО: Сохранено {len(validated)} курсов в {output_path}")
    return validated

if __name__ == "__main__":
    run_unify_pipeline()