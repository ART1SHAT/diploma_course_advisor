# app/course_loader.py — только JSON
import json
from pathlib import Path
from typing import List, Dict, Optional

class CourseLoader:
    def __init__(self, json_path: str = "data/unified_courses.json"):
        self.json_path = Path(json_path)
        self._cache: Optional[List[Dict]] = None
        
    def load(self) -> List[Dict]:
        if self._cache is None:
            with open(self.json_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        return self._cache
    
    def get_stats(self) -> Dict:
        courses = self.load()
        if not courses:
            return {"total": 0}
        providers = {}
        for c in courses:
            p = c.get("provider", "unknown")
            providers[p] = providers.get(p, 0) + 1
        return {
            "total": len(courses),
            "providers": providers,
            "avg_price": sum(c.get("price", 0) for c in courses) / len(courses)
        }