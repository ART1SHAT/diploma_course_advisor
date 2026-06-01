# app/recommender.py — Гибридный скоринг v2 (семантика + калиброванный fuzzy)
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from app.fuzzy_engine import FuzzyInferenceEngine
from app.user_profile import UserProfile
from app.course_loader import CourseLoader
from app.semantic_search import SemanticIndex

class HybridRecommender:
    def __init__(self, json_path: str = "data/unified_courses.json"):
        self.fuzzy_engine = FuzzyInferenceEngine()
        self.loader = CourseLoader(json_path)
        self.courses = self.loader.load()
        
        # Инициализация семантического индекса
        self.sem_index = SemanticIndex()
        self.sem_index.build(self.courses)
        try:
            self.sem_index.build(self.courses)
        except Exception as e:
            print(f"⚠️ Семантический индекс не построен: {e}")
            
        # Калибровка нечётких термов
        try:
            df = pd.DataFrame(self.courses)
            self.fuzzy_engine.calibrate_from_data(df)
        except Exception as e:
            print(f"⚠️ Калибровка пропущена: {e}")

    def _semantic_score(self, profile: UserProfile, course: Dict) -> float:
        """Безопасный семантический скор"""
        try:
            query = f"{profile.goals or ''} {' '.join(profile.interests or [])}".strip()
            if not query or not hasattr(self, 'sem_index') or self.sem_index is None:
                return 0.5
            
            # Ищем курс по ID в результатах поиска
            for c, score in self.sem_index.search(query, top_k=100):
                if str(c.get("id")) == str(course.get("id")):
                    return max(0.0, min(1.0, float(score)))
            return 0.0
        except Exception as e:
            logging.getLogger(__name__).warning(f"⚠️ _semantic_score fallback: {e}")
            return 0.5  # нейтральный скор при ошибке

    def _fuzzy_score(self, profile: UserProfile, course: Dict) -> float:
        """Нечёткий скор на основе правил"""
        fuzzy_input = profile.to_fuzzy_input()
        if not fuzzy_input:
            return 0.5
            
        conclusions = self.fuzzy_engine.infer(fuzzy_input)
        course_type = str(course.get("provider_type", ""))
        
        score = 0.0
        for concl_type, weight in conclusions.items():
            # Простое сопоставление типа провайдера с выводом
            mapping = {
                "practical_intensive": ["bootcamp", "corporate"],
                "theoretical_comprehensive": ["university"],
                "budget_friendly": ["government_platform"],
                "premium_certified": ["bootcamp", "university"]
            }
            if course_type in mapping.get(concl_type, []):
                score += weight * 0.9
        return min(score, 1.0)

    def recommend(self, profile: UserProfile, top_k: int = 5) -> List[Dict]:
        scored = []
        for course in self.courses:
            # Жёсткие фильтры
            price = float(course.get("price", 0) or 0)
            if profile.budget and price > profile.budget * 1.5:
                continue
                
            sem = self._semantic_score(profile, course)
            fuzzy = self._fuzzy_score(profile, course)
            
            # Гибридная формула (§1.2, §2.3)
            final = 0.55 * sem + 0.35 * fuzzy + 0.10 * (1.0 if price == 0 else 0.5)
            
            scored.append({**course, "score": round(final, 3)})
            
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]

    def explain(self, profile: UserProfile, course: Dict) -> Dict:
        trace = self.fuzzy_engine.get_trace(profile.to_fuzzy_input())
        explanations = []
        
        if profile.goals:
            if str(profile.goals).lower() in str(course.get("description", "")).lower():
                explanations.append(f"✓ Соответствует цели: «{profile.goals}»")
                
        for t in trace:
            if t["activation"] > 0.2:
                explanations.append(f"✓ {t['rule_name']} ({t['activation']:.0%})")
                
        if profile.budget:
            p = float(course.get("price", 0))
            explanations.append("✓ Бесплатный курс" if p == 0 else f"✓ В бюджете ({int(p)} ₽)")
            
        return {
            "course_id": str(course.get("id")),
            "explanations": explanations or ["✓ Общее соответствие запросу"],
            "confidence": round(sum(t["activation"] for t in trace) / max(len(trace), 1), 2),
            "fuzzy_trace": trace
        }
    
    def get_stats(self) -> Dict:
        return self.loader.get_stats()