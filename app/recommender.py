# app/recommender.py (обновлённая версия)
from typing import List, Dict, Optional
import pandas as pd
from app.fuzzy_engine import FuzzyInferenceEngine
from app.user_profile import UserProfile
from app.course_loader import CourseLoader
from sentence_transformers import SentenceTransformer, util
import numpy as np

class HybridRecommender:
    """Рекомендатель: семантика + нечёткая логика + реальные данные"""
    
    def __init__(self, csv_path: str = "data/courses_processed.csv", 
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.fuzzy_engine = FuzzyInferenceEngine()
        self.loader = CourseLoader(csv_path)
        
        # Загрузка курсов
        df = self.loader.load(limit=500)  # MVP: первые 500 для скорости
        self.courses = self.loader.to_dict_list(df)
        
        # Инициализация эмбеддингов (кэширование)
        try:
            self.model = SentenceTransformer(model_name)
            self._precompute_embeddings()
        except Exception as e:
            print(f"⚠️ Не удалось загрузить модель эмбеддингов: {e}")
            print("💡 Используем fallback на ключевые слова")
            self.model = None
            self.course_embeddings = None
    
    def _precompute_embeddings(self):
        """Предварительное вычисление эмбеддингов описаний"""
        if self.model and self.courses:
            texts = [f"{c['title']} {c['description'][:500]}" for c in self.courses]
            self.course_embeddings = self.model.encode(texts, convert_to_tensor=True)
    
    def _semantic_score(self, profile: UserProfile, course: Dict) -> float:
        """Семантический скор: эмбеддинги + ключевые слова"""
        score = 0.0
        
        # 1. Эмбеддинг-поиск (если модель загружена)
        if self.model and self.course_embeddings is not None:
            query = f"{profile.goals} {' '.join(profile.interests)}"
            query_emb = self.model.encode(query, convert_to_tensor=True)
            
            # Находим индекс курса в списке
            try:
                idx = next(i for i, c in enumerate(self.courses) if c['id'] == course['id'])
                sim = util.cos_sim(query_emb, self.course_embeddings[idx])[0][0].item()
                score += max(0, sim) * 0.7  # 70% веса на эмбеддинги
            except StopIteration:
                pass
        
        # 2. Ключевые слова (fallback + дополнение)
        text = (course.get('description', '') + ' ' + 
                course.get('title', '') + ' ' +
                ' '.join(course.get('skills', []))).lower()
        
        for interest in profile.interests:
            if interest.lower() in text:
                score += 0.15
        if profile.goals and profile.goals.lower() in text:
            score += 0.2
        
        # 3. Соответствие категории
        if profile.interests:
            cat = course.get('category', '').lower()
            if any(interest.lower() == cat for interest in profile.interests):
                score += 0.1
        
        return min(score, 1.0)
    
    def _fuzzy_score(self, profile: UserProfile, course: Dict) -> float:
        """Нечёткий скор на основе правил"""
        fuzzy_input = profile.to_fuzzy_input()
        if not fuzzy_input:
            return 0.5
        
        conclusions = self.fuzzy_engine.infer(fuzzy_input)
        course_type = course.get('type', '')
        
        # Сопоставление типа курса с выводами движка
        score = 0.0
        for concl_type, weight in conclusions.items():
            if concl_type == course_type:
                score += weight * 0.9
            # Частичное совпадение по ключевым словам типа
            elif any(kw in course.get('description', '').lower() 
                    for kw in self._get_type_keywords(concl_type)):
                score += weight * 0.4
        
        return min(score, 1.0)
    
    def _get_type_keywords(self, course_type: str) -> List[str]:
        """Ключевые слова для каждого типа курса"""
        mapping = {
            'practical_intensive': ['интенсив', 'практика', 'проект', 'быстро', 'навык'],
            'theoretical_comprehensive': ['теория', 'фундамент', 'егэ', 'академ', 'математик'],
            'budget_friendly': ['бесплат', 'бюджет', 'доступ', 'эконом'],
            'premium_certified': ['сертификат', 'диплом', 'карьера', 'трудоустройств'],
            'flexible_selfpaced': ['гибк', 'самостоя', 'онлайн', 'хобби', 'личн']
        }
        return mapping.get(course_type, [])
    
    def recommend(self, profile: UserProfile, top_k: int = 5) -> List[Dict]:
        """Основной метод рекомендаций"""
        scored = []
        
        for course in self.courses:
            # Фильтры по обязательным критериям
            if profile.budget and course['price'] > profile.budget * 1.5:
                continue  # курс значительно дороже бюджета
            
            sem_score = self._semantic_score(profile, course)
            fuzzy_score = self._fuzzy_score(profile, course)
            
            # Гибридная формула
            final_score = 0.6 * sem_score + 0.4 * fuzzy_score
            
            scored.append({**course, "score": round(final_score, 3)})
        
        # Сортировка и возврат топ-k
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
    
    def explain(self, profile: UserProfile, course: Dict) -> Dict:
        """Генерация объяснения рекомендации"""
        trace = self.fuzzy_engine.get_trace(profile.to_fuzzy_input())
        explanations = []
        
        # 1. Соответствие целям
        if profile.goals:
            desc = course.get('description', '').lower()
            if profile.goals.lower() in desc:
                explanations.append(f"✓ Курс соответствует вашей цели: «{profile.goals}»")
        
        # 2. Активные нечёткие правила
        for t in trace:
            if t["activation"] > 0.25:
                explanations.append(f"✓ {t['rule_name']} (уверенность: {t['activation']:.0%})")
        
        # 3. Навыки и интересы
        matched_skills = [s for s in course.get('skills', []) 
                         if any(s.lower() in i.lower() for i in profile.interests)]
        if matched_skills:
            explanations.append(f"✓ Развивает навыки: {', '.join(matched_skills[:3])}")
        
        # 4. Практические параметры
        if profile.budget:
            price = course.get('price', 0)
            if price == 0:
                explanations.append("✓ Бесплатный курс")
            elif price <= profile.budget:
                explanations.append(f"✓ Укладывается в бюджет ({int(price)} ₽)")
        
        if course.get('language') == profile.__dict__.get('preferred_language', 'ru'):
            explanations.append(f"✓ На русском языке")
        
        # 5. Категория
        cat = course.get('category')
        if cat and cat != 'Other':
            explanations.append(f"✓ Категория: {cat}")
        
        return {
            "course_id": course.get("id"),
            "explanations": explanations if explanations else ["✓ Рекомендовано на основе общего соответствия"],
            "confidence": round(sum(t["activation"] for t in trace) / max(len(trace), 1), 2),
            "fuzzy_trace": trace
        }
    
    def get_stats(self) -> Dict:
        """Статистика по базе курсов для демо"""
        df = self.loader.load()
        return {
            "total_courses": len(df),
            "categories": df['category'].value_counts().head(5).to_dict(),
            "price_range": {
                "min": float(df['price'].min()),
                "max": float(df['price'].max()),
                "median": float(df['price'].median())
            },
            "languages": df['language'].value_counts().to_dict()
        }