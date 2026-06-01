# app/course_loader.py — универсальный загрузчик
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

class CourseLoader:
    def __init__(self, json_path: str = "data/unified_courses.json"):
        self.data_path = Path(json_path)
        self._cache: Optional[List[Dict]] = None
        
    def load(self) -> List[Dict]:
        if self._cache is not None:
            return self._cache
            
        # Поддержка JSON и CSV
        if self.data_path.suffix == '.json':
            with open(self.data_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        elif self.data_path.suffix == '.csv':
            # Fallback: загружаем старый CSV и конвертируем
            df = pd.read_csv(self.data_path, on_bad_lines='skip')
            self._cache = []
            for _, row in df.head(400).iterrows():
                self._cache.append({
                    "id": f"stepik_{int(row.get('course_id', 0))}",
                    "title": str(row.get("title", "")),
                    "description": str(row.get("description", "")),
                    "price": float(row.get("price", 0) or 0),
                    "provider": "Stepik",
                    "provider_type": "mooc",
                    "format": "online",
                    "language": str(row.get("language", "ru") or "ru"),
                    "competencies": [str(row.get("category", ""))] if row.get("category") != "Other" else [],
                    "certification": "professional" if "сертификат" in str(row.get("description", "")).lower() else "none",
                    "url": str(row.get("url", "")),
                })
        else:
            raise ValueError(f"Неподдерживаемый формат: {self.data_path.suffix}")
            
        return self._cache
    
    def get_stats(self) -> Dict:
        courses = self.load()
        return {
            "total": len(courses),
            "providers": list(set(c.get("provider") for c in courses if c.get("provider"))),
            "avg_price": sum(c.get("price", 0) for c in courses) / max(len(courses), 1)
        }