# app/recommender.py
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import logging
from app.fuzzy_engine import FuzzyInferenceEngine
from app.user_profile import UserProfile
from app.course_loader import CourseLoader
from app.semantic_search import SemanticIndex

logger = logging.getLogger(__name__)

class HybridRecommender:
    """Гибридный рекомендатель: семантика + нечёткая логика"""
    
    def __init__(self, json_path: str = "data/unified_courses.json"):
        self.fuzzy_engine = FuzzyInferenceEngine()
        self.loader = CourseLoader(json_path)
        self.courses = self.loader.load()
        
        self.sem_index = SemanticIndex()
        try:
            self.sem_index.build(self.courses)
            logger.info(f"✅ Семантический индекс построен: {self.sem_index.vectors.shape if self.sem_index.vectors is not None else 'N/A'}")
        except Exception as e:
            logger.warning(f"⚠️ Семантический индекс не построен: {e}")
            
        try:
            df = pd.DataFrame(self.courses)
            self.fuzzy_engine.calibrate_from_data(df)
        except Exception as e:
            logger.warning(f"⚠️ Калибровка пропущена: {e}")
        
        logger.info(f"✅ HybridRecommender инициализирован: {len(self.courses)} курсов")

    def _semantic_score(self, profile: UserProfile, course: Dict) -> float:
        """Семантический скор с усилением"""
        try:
            query = f"{profile.goals or ''} {' '.join(profile.interests or [])}".strip()
            if not query:
                return 0.7
            
            for c, score in self.sem_index.search(query, top_k=100):
                if str(c.get("id")) == str(course.get("id")):
                    # Усиливаем семантический скор
                    enhanced_score = min(1.0, score * 1.3)
                    return max(0.3, enhanced_score)
            return 0.5
        except Exception as e:
            logger.warning(f"⚠️ _semantic_score fallback: {e}")
            return 0.6

    def _fuzzy_score(self, profile: UserProfile, course: Dict) -> float:
        """Нечёткий скор с учётом бюджета и соответствия"""
        try:
            fuzzy_input = profile.to_fuzzy_input()
            if not fuzzy_input:
                return 0.7
            
            conclusions = self.fuzzy_engine.infer(fuzzy_input)
            course_type = str(course.get("provider_type", ""))
            course_price = float(course.get("price", 0) or 0)
            
            score = 0.0
            max_boost = 0.0
            
            mapping = {
                "practical_intensive": ["bootcamp", "corporate"],
                "theoretical_comprehensive": ["university"],
                "budget_friendly": ["government_platform", "mooc"],
                "premium_certified": ["bootcamp", "university"]
            }
            
            for concl_type, weight in conclusions.items():
                if course_type in mapping.get(concl_type, []):
                    base_score = weight * 0.9
                    score = max(score, base_score)
                    max_boost = max(max_boost, weight)
                
                if concl_type == "budget_friendly":
                    if course_price == 0 or (profile.budget and course_price < profile.budget * 0.3):
                        score = max(score, weight * 0.7)
                elif concl_type == "premium_certified":
                    if profile.budget and course_price > profile.budget * 0.7:
                        score = max(score, weight * 0.6)
            
            # Дополнительный бонус за соответствие бюджету
            if profile.budget:
                if course_price <= profile.budget:
                    budget_match = 1.0 - (course_price / (profile.budget * 2))
                    score = max(score, budget_match * 0.5)
                else:
                    penalty = (course_price - profile.budget) / profile.budget
                    score = max(0, score - penalty * 0.3)
            
            return min(1.0, max(0.3, score + max_boost * 0.2))
        except Exception as e:
            logger.warning(f"⚠️ _fuzzy_score fallback: {e}")
            return 0.6

    def _constraint_score(self, profile: UserProfile, course: Dict) -> float:
        """Скор соответствия ограничениям (бюджет, язык)"""
        score = 1.0
        
        # 1. Проверка бюджета
        course_price = float(course.get("price", 0) or 0)
        if profile.budget and course_price > profile.budget:
            penalty = (course_price - profile.budget) / profile.budget
            score -= 0.3 * min(1.0, penalty)
        
        # 2. Безопасная проверка языка (работает с любым названием атрибута)
        course_lang = str(course.get("language", "ru") or "ru")
        # Пытаемся найти язык пользователя в любом из возможных полей
        preferred_lang = getattr(profile, "preferred_language", None) or getattr(profile, "language", None)
        
        if preferred_lang and course_lang != preferred_lang:
            score -= 0.4
        
        return max(0.0, score)

    def recommend(self, profile: UserProfile, top_k: int = 5) -> List[Dict]:
        """Генерация рекомендаций с улучшенным скорингом"""
        scored = []
        
        for course in self.courses:
            course_price = float(course.get("price", 0) or 0)
            
            # Мягкая фильтрация по бюджету (не жёсткая!)
            if profile.budget and course_price > profile.budget * 2.0:
                continue
            
            sem_score = self._semantic_score(profile, course)
            fuzzy_score = self._fuzzy_score(profile, course)
            constraint_score = self._constraint_score(profile, course)
            
            # Усиленная формула с акцентом на fuzzy
            final_score = (
                0.35 * sem_score +
                0.50 * fuzzy_score +
                0.15 * constraint_score
            )
            
            # Дополнительное усиление для курсов в бюджете
            if profile.budget and course_price <= profile.budget:
                final_score *= 1.15
            
            scored.append({**course, "score": round(min(1.0, final_score), 3)})
        
        # Сортировка и нормализация
        scored.sort(key=lambda x: x["score"], reverse=True)
        
        if scored:
            max_score = max(c["score"] for c in scored)
            min_score = min(c["score"] for c in scored[:top_k*2]) if len(scored) >= top_k*2 else min(c["score"] for c in scored)
            
            for c in scored:
                if max_score != min_score:
                    normalized = 0.4 + 0.6 * (c["score"] - min_score) / (max_score - min_score)
                else:
                    normalized = 0.85
                c["normalized_score"] = round(normalized, 3)
        
        return scored[:top_k]

    def explain(self, profile: UserProfile, course: Dict) -> Dict:
        """Генерация объяснений"""
        try:
            trace = self.fuzzy_engine.get_trace(profile.to_fuzzy_input())
            explanations = []
            
            course_price = float(course.get("price", 0) or 0)
            
            # Проверка бюджета
            if profile.budget:
                if course_price == 0:
                    explanations.append({
                        "type": "budget",
                        "text": "Бесплатный курс полностью укладывается в бюджет",
                        "confidence": 1.0
                    })
                elif course_price <= profile.budget:
                    explanations.append({
                        "type": "budget",
                        "text": f"Курс укладывается в бюджет ({int(course_price)} ₽ из {int(profile.budget)} ₽)",
                        "confidence": 0.9
                    })
            
            # Активированные правила
            for t in trace:
                if t["activation"] > 0.15:
                    explanations.append({
                        "type": "fuzzy_rule",
                        "text": f"Активировано правило {t['rule_id']} ({int(t['activation']*100)}%): {t['rule_name']}",
                        "confidence": t["activation"]
                    })
            
            # Семантическое соответствие
            if profile.goals:
                desc = str(course.get("description", "") or "").lower()
                if str(profile.goals).lower() in desc:
                    explanations.append({
                        "type": "semantic",
                        "text": "Семантическая близость описания курса к вашему запросу",
                        "confidence": 0.85
                    })
            
            # Навыки и интересы
            if profile.interests:
                course_skills = course.get("competencies", []) or []
                matched = [s for s in course_skills if any(i.lower() in s.lower() for i in profile.interests)]
                if matched:
                    explanations.append({
                        "type": "skills",
                        "text": f"Развивает навыки: {', '.join(matched[:3])}",
                        "confidence": 0.8
                    })
            
            confidence = sum(e["confidence"] for e in explanations) / max(len(explanations), 1)
            
            return {
                "course_id": str(course.get("id", "")),
                "explanations": explanations if explanations else [{"type": "default", "text": "Курс соответствует вашим критериям", "confidence": 0.6}],
                "confidence": round(confidence, 3),
                "fuzzy_trace": trace
            }
        except Exception as e:
            logger.error(f"❌ Ошибка в explain: {e}")
            return {
                "course_id": str(course.get("id", "")),
                "explanations": [{"type": "fallback", "text": "Рекомендовано на основе общего соответствия", "confidence": 0.5}],
                "confidence": 0.5,
                "fuzzy_trace": []
            }
    
    def get_stats(self) -> Dict:
        """Статистика по каталогу"""
        return self.loader.get_stats()