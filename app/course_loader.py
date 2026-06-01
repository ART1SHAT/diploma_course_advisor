# app/course_loader.py
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path

class CourseLoader:
    """Загрузчик и нормализатор курсов из CSV"""
    
    REQUIRED_FIELDS = ['course_id', 'title', 'description']
    
    def __init__(self, csv_path: str = "data/courses_processed.csv"):
        self.csv_path = Path(csv_path)
        self._cache: Optional[pd.DataFrame] = None
    
    def load(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Загружает и кэширует данные"""
        if self._cache is None:
            df = pd.read_csv(
                self.csv_path,
                encoding='utf-8',
                on_bad_lines='skip',
                low_memory=False
            )
            # Нормализация колонок
            df = df.rename(columns={
                df.columns[0]: 'course_id',
                df.columns[1]: 'title', 
                df.columns[2]: 'price',
                df.columns[3]: 'language',
                df.columns[4]: 'url',
                df.columns[5]: 'description',
                df.columns[6]: 'category'
            })
            # Приведение типов
            df['course_id'] = pd.to_numeric(df['course_id'], errors='coerce')
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
            df['language'] = df['language'].fillna('ru')
            df['category'] = df['category'].fillna('Other')
            df['description'] = df['description'].fillna('')
            df['title'] = df['title'].fillna('Без названия')
            
            # Удаление дубликатов и пустых записей
            df = df.dropna(subset=['course_id', 'title']).drop_duplicates('course_id')
            self._cache = df
        
        result = self._cache
        if limit:
            result = result.head(limit)
        return result
    
    def to_dict_list(self, df: pd.DataFrame) -> List[Dict]:
        """Конвертирует DataFrame в список словарей для рекомендателя"""
        courses = []
        for _, row in df.iterrows():
            course = {
                "id": str(int(row['course_id'])),
                "title": row['title'],
                "description": row['description'],
                "price": float(row['price']) if pd.notna(row['price']) else 0,
                "language": row['language'],
                "url": row['url'],
                "category": row['category'],
                # Дополнительные вычисляемые поля
                "skills": self._extract_skills(row['description'], row['category']),
                "format": self._infer_format(row['description']),
                "duration_estimate": self._infer_duration(row['description']),
                "type": self._infer_course_type(row['category'], row['description']),
            }
            courses.append(course)
        return courses
    
    def _extract_skills(self, description: str, category: str) -> List[str]:
        """Извлекает навыки из описания и категории"""
        skills = []
        # Добавляем категорию как навык
        if category and category != 'Other':
            skills.append(category.lower())
        
        # Ключевые слова для извлечения
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
        """Определяет формат курса по описанию"""
        desc = description.lower()
        if any(w in desc for w in ['онлайн', 'самостоя', 'гибк', 'дистанц']):
            return 'online'
        elif any(w in desc for w in ['очно', 'аудитория', 'лекция', 'группа']):
            return 'offline'
        elif any(w in desc for w in ['смешан', 'гибрид', 'онлайн и очно']):
            return 'hybrid'
        return 'online'  # default
    
    def _infer_duration(self, description: str) -> Optional[int]:
        """Оценивает длительность в неделях"""
        import re
        desc = description.lower()
        # Ищем упоминания времени
        weeks = re.findall(r'(\d+)\s*(нед|недель|неделя|неделю)', desc)
        if weeks:
            return int(weeks[0][0])
        hours = re.findall(r'(\d+)\s*(час|часов|часа|ч\.)', desc)
        if hours:
            return max(1, int(hours[0][0]) // 4)  # грубая оценка: 4 часа = 1 неделя
        return None
    
    def _infer_course_type(self, category: str, description: str) -> str:
        """Определяет тип курса для fuzzy-правил"""
        desc = description.lower()
        cat = (category or '').lower()
        
        # Практические интенсивы
        if any(w in desc for w in ['интенсив', 'практик', 'проект', 'быстро', 'старт']):
            return 'practical_intensive'
        # Теоретические/академические
        if any(w in desc for w in ['теория', 'фундамент', 'академ', 'егэ', 'олимпиад']):
            return 'theoretical_comprehensive'
        # Бюджетные/бесплатные
        if 'price' in desc or any(w in desc for w in ['бесплат', 'бюджет', 'доступ']):
            return 'budget_friendly'
        # Премиум с сертификатом
        if any(w in desc for w in ['сертификат', 'диплом', 'премиум', 'карьера']):
            return 'premium_certified'
        # Гибкие для саморазвития
        if any(w in desc for w in ['самостоя', 'гибк', 'хобби', 'личн', 'для себя']):
            return 'flexible_selfpaced'
        
        # По категории
        if cat in ['python', 'programming', 'it']:
            return 'practical_intensive'
        if cat in ['mathematics', 'school', 'ege']:
            return 'theoretical_comprehensive'
        
        return 'flexible_selfpaced'  # default