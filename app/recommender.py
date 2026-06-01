# app/recommender.py — ULTRA-SAFE версия для MVP
from typing import List, Dict, Optional, Union, Any
import pandas as pd
import numpy as np
from app.fuzzy_engine import FuzzyInferenceEngine
from app.user_profile import UserProfile
from app.course_loader import CourseLoader
import re

def to_scalar(val: Any, default: Any = None) -> Any:
    """Гарантированно возвращает скалярное значение из любого pandas-типа"""
    if val is None:
        return default
    if isinstance(val, (pd.Series, np.ndarray)):
        if len(val) == 0:
            return default
        val = val.iloc[0] if hasattr(val, 'iloc') else val.item()
    if isinstance(val, (np.generic, np.bool_)):
        return val.item()
    if pd.isna(val):
        return default
    return val

def to_str(val: Any, default: str = "") -> str:
    """Безопасное преобразование в строку"""
    scalar = to_scalar(val, default)
    return str(scalar).strip() if scalar is not None else default

def to_float(val: Any, default: float = 0.0) -> float:
    """Безопасное преобразование в float"""
    try:
        scalar = to_scalar(val, default)
        return float(scalar) if scalar is not None else default
    except (ValueError, TypeError):
        return default

class HybridRecommender:
    """Рекомендатель с максимальной защитой от pandas-ошибок"""
    
    def __init__(self, csv_path: str = "data/courses_processed.csv", 
                 use_embeddings: bool = False):
        self.fuzzy_engine = FuzzyInferenceEngine()
        self.loader = CourseLoader(csv_path)
        
        # Загружаем данные
        df = self.loader.load(limit=500)
        self.courses = self._convert_to_safe_list(df)
        
        # Эмбеддинги отключены для стабильности
        self.use_embeddings = False
        print(f"✅ Recommender initialized with {len(self.courses)} courses")
    
    def _convert_to_safe_list(self, df: pd.DataFrame) -> List[Dict]:
        """Конвертирует DataFrame в список словарей с ТОЛЬКО скалярными значениями"""
        courses = []
        for idx, row in df.iterrows():
            try:
                course = {
                    "id": to_str(row.get('course_id'), f"unknown_{idx}"),
                    "title": to_str(row.get('title'), 'Без названия'),
                    "description": to_str(row.get('description'), ''),
                    "price": to_float(row.get('price'), 0),
                    "language": to_str(row.get('language'), 'ru'),
                    "url": to_str(row.get('url'), ''),
                    "category": to_str(row.get('category'), 'Other'),
                    "skills": self._extract_skills(
                        to_str(row.get('description')), 
                        to_str(row.get('category'))
                    ),
                    "format": self._infer_format(to_str(row.get('description'))),
                    "duration_estimate": self._infer_duration(to_str(row.get('description'))),
                    "type": self._infer_course_type(
                        to_str(row.get('category')), 
                        to_str(row.get('description'))
                    ),
                }
                # Финальная проверка: все значения должны быть скалярами
                for k, v in course.items():
                    if isinstance(v, (pd.Series, np.ndarray, list)) and k != 'skills':
                        course[k] = str(v.iloc[0]) if hasattr(v, 'iloc') else str(v)
                courses.append(course)
            except Exception as e:
                print(f"⚠️  Пропущена строка {idx}: {e}")
                continue
        return courses
    
    def _extract_skills(self, description: str, category: str) -> List[str]:
        """Извлечение навыков — только со строками"""
        if not isinstance(description, str):
            description = ""
        if not isinstance(category, str):
            category = ""
            
        skills = []
        if category and category != 'Other':
            skills.append(category.lower())
        
        skill_keywords = {
            'python': ['python', 'django', 'flask', 'fastapi'],
            'sql': ['sql', 'postgresql', 'mysql', 'база данных'],
            'анализ данных': ['анализ данных', 'pandas', 'numpy', 'визуализация'],
            'машинное обучение': ['машинное обучение', 'ml', 'нейросеть', 'scikit'],
            'веб-разработка': ['веб', 'html', 'css', 'javascript', 'frontend', 'backend'],
            'английский': ['английский', 'english', 'language', 'язык'],
            'математика': ['математика', 'алгебра', 'геометрия', 'статистика'],
            'тестирование': ['тестирование', 'qa', 'автотест', 'selenium'],
        }
        
        desc_lower = description.lower()
        for skill, keywords in skill_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                skills.append(skill)
        return list(set(skills))
    
    def _infer_format(self, description: str) -> str:
        desc = description.lower() if isinstance(description, str) else ""
        if any(w in desc for w in ['онлайн', 'самостоя', 'гибк', 'дистанц']):
            return 'online'
        elif any(w in desc for w in ['очно', 'аудитория', 'лекция', 'группа']):
            return 'offline'
        return 'online'
    
    def _infer_duration(self, description: str) -> Optional[int]:
        desc = description.lower() if isinstance(description, str) else ""
        weeks = re.findall(r'(\d+)\s*(нед|недель|неделя|неделю)', desc)
        if weeks:
            try:
                return int(weeks[0][0])
            except:
                pass
        hours = re.findall(r'(\d+)\s*(час|часов|часа|ч\.)', desc)
        if hours:
            try:
                return max(1, int(hours[0][0]) // 4)
            except:
                pass
        return None
    
    def _infer_course_type(self, category: str, description: str) -> str:
        desc = description.lower() if isinstance(description, str) else ""
        cat = category.lower() if isinstance(category, str) else ""
        
        if any(w in desc for w in ['интенсив', 'практик', 'проект', 'быстро', 'старт']):
            return 'practical_intensive'
        if any(w in desc for w in ['теория', 'фундамент', 'академ', 'егэ', 'олимпиад']):
            return 'theoretical_comprehensive'
        if any(w in desc for w in ['бесплат', 'бюджет', 'доступ']):
            return 'budget_friendly'
        if any(w in desc for w in ['сертификат', 'диплом', 'премиум', 'карьера']):
            return 'premium_certified'
        if any(w in desc for w in ['самостоя', 'гибк', 'хобби', 'личн', 'для себя']):
            return 'flexible_selfpaced'
        if cat in ['python', 'programming', 'it']:
            return 'practical_intensive'
        if cat in ['mathematics', 'school', 'ege']:
            return 'theoretical_comprehensive'
        return 'flexible_selfpaced'
    
    def _semantic_score(self, profile: UserProfile, course: Dict) -> float:
        """Семантический скор — только со скалярными строками"""
        score = 0.0
        
        # Гарантируем строки
        desc = to_str(course.get('description'), '')
        title = to_str(course.get('title'), '')
        skills = course.get('skills', []) or []
        
        text = f"{desc} {title} {' '.join(skills)}".lower()
        
        # Интересы
        interests = profile.interests or []
        for interest in interests:
            if interest and isinstance(interest, str) and interest.lower() in text:
                score += 0.15
        
        # Цель
        goals = profile.goals or ""
        if isinstance(goals, str) and goals.lower() in text:
            score += 0.2
        
        # Категория
        cat = to_str(course.get('category'), '').lower()
        for interest in interests:
            if interest and isinstance(interest, str) and interest.lower() == cat:
                score += 0.1
                break
        
        return min(score, 1.0)
    
    def _fuzzy_score(self, profile: UserProfile, course: Dict) -> float:
        """Нечёткий скор"""
        fuzzy_input = profile.to_fuzzy_input()
        if not fuzzy_input:
            return 0.5
        
        conclusions = self.fuzzy_engine.infer(fuzzy_input)
        course_type = to_str(course.get('type'), '')
        
        score = 0.0
        for concl_type, weight in conclusions.items():
            if concl_type == course_type:
                score += weight * 0.9
            else:
                desc = to_str(course.get('description'), '')
                keywords = self._get_type_keywords(concl_type)
                if any(kw in desc.lower() for kw in keywords):
                    score += weight * 0.4
        
        return min(score, 1.0)
    
    def _get_type_keywords(self, course_type: str) -> List[str]:
        mapping = {
            'practical_intensive': ['интенсив', 'практика', 'проект', 'быстро', 'навык'],
            'theoretical_comprehensive': ['теория', 'фундамент', 'егэ', 'академ', 'математик'],
            'budget_friendly': ['бесплат', 'бюджет', 'доступ', 'эконом'],
            'premium_certified': ['сертификат', 'диплом', 'карьера', 'трудоустройств'],
            'flexible_selfpaced': ['гибк', 'самостоя', 'онлайн', 'хобби', 'личн']
        }
        return mapping.get(course_type, [])
    
    def recommend(self, profile: UserProfile, top_k: int = 5) -> List[Dict]:
        """Основной метод — с безопасными сравнениями"""
        scored = []
        
        for course in self.courses:
            # Безопасная фильтрация по бюджету
            course_price = to_float(course.get('price'), 0)
            profile_budget = profile.budget
            
            # Явное сравнение скаляров
            if profile_budget is not None and isinstance(profile_budget, (int, float)):
                if course_price > profile_budget * 1.5:
                    continue
            
            sem_score = self._semantic_score(profile, course)
            fuzzy_score = self._fuzzy_score(profile, course)
            
            final_score = 0.6 * sem_score + 0.4 * fuzzy_score
            
            scored.append({**course, "score": round(final_score, 3)})
        
        # Сортировка
        try:
            return sorted(scored, key=lambda x: float(x.get("score", 0)), reverse=True)[:top_k]
        except:
            return scored[:top_k]
    
    def explain(self, profile: UserProfile, course: Dict) -> Dict:
        """Генерация объяснений"""
        trace = self.fuzzy_engine.get_trace(profile.to_fuzzy_input())
        explanations = []
        
        # Цели
        goals = profile.goals or ""
        desc = to_str(course.get('description'), '')
        if isinstance(goals, str) and goals and goals.lower() in desc.lower():
            explanations.append(f"✓ Курс соответствует вашей цели: «{goals}»")
        
        # Правила
        for t in trace:
            if t.get("activation", 0) > 0.25:
                explanations.append(f"✓ {t.get('rule_name', '')} (уверенность: {t['activation']:.0%})")
        
        # Навыки
        matched_skills = [
            s for s in (course.get('skills') or []) 
            if any(str(s).lower() in str(i).lower() for i in (profile.interests or []))
        ]
        if matched_skills:
            explanations.append(f"✓ Развивает навыки: {', '.join(matched_skills[:3])}")
        
        # Бюджет
        if profile.budget is not None:
            price = to_float(course.get('price'), 0)
            if price == 0:
                explanations.append("✓ Бесплатный курс")
            elif price <= profile.budget:
                explanations.append(f"✓ Укладывается в бюджет ({int(price)} ₽)")
        
        # Язык и категория
        if to_str(course.get('language'), 'ru') == 'ru':
            explanations.append("✓ На русском языке")
        
        cat = course.get('category')
        if cat and cat != 'Other':
            explanations.append(f"✓ Категория: {cat}")
        
        return {
            "course_id": to_str(course.get("id"), ""),
            "explanations": explanations if explanations else ["✓ Рекомендовано на основе общего соответствия"],
            "confidence": round(sum(t.get("activation", 0) for t in trace) / max(len(trace), 1), 2),
            "fuzzy_trace": trace
        }
    
    def get_stats(self) -> Dict:
        """Статистика"""
        df = self.loader.load()
        return {
            "total_courses": int(len(df)),
            "categories": df['category'].value_counts().head(5).to_dict(),
            "price_range": {
                "min": float(df['price'].min()),
                "max": float(df['price'].max()),
                "median": float(df['price'].median())
            },
            "languages": df['language'].value_counts().to_dict()
        }