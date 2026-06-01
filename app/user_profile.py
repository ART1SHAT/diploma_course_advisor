# app/user_profile.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class UserProfile:
    """Модель профиля пользователя с лингвистическими переменными"""
    
    # Структурированные атрибуты
    age: Optional[int] = None
    education: Optional[str] = None
    location: Optional[str] = None
    
    # Числовые предпочтения (для fuzzy-переменных)
    budget: Optional[float] = None  # в рублях
    knowledge_level: Optional[float] = None  # 0-10 шкала
    time_availability: Optional[float] = None  # часов в неделю
    career_focus: Optional[float] = None  # 0=employment, 0.5=academic, 1=personal
    
    # Текстовые предпочтения
    interests: List[str] = field(default_factory=list)
    goals: str = ""
    
    # Контекст диалога
    clarified_slots: List[str] = field(default_factory=list)
    
    def to_fuzzy_input(self) -> Dict[str, float]:
        """Преобразует профиль в формат для нечёткого движка"""
        result = {}
        if self.budget is not None:
            result["budget"] = self.budget
        if self.knowledge_level is not None:
            result["knowledge_level"] = self.knowledge_level
        if self.time_availability is not None:
            result["time_availability"] = self.time_availability
        if self.career_focus is not None:
            result["career_focus"] = self.career_focus
        return result
    
    def is_complete(self, required_slots: List[str] = None) -> bool:
        """Проверяет, достаточно ли данных для рекомендации"""
        required = required_slots or ["budget", "career_focus"]
        fuzzy_input = self.to_fuzzy_input()
        return all(slot in fuzzy_input for slot in required)