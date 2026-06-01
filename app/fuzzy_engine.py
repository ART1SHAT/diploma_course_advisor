# app/fuzzy_engine.py
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import math
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class FuzzyTerm:
    """Лингвистический терм с функцией принадлежности"""
    name: str
    params: List[float]  # [a, b, c] для треугольной или [a, b, c, d] для трапеции
    shape: str = "triangle"  # "triangle" или "trapezoid"
    
    def membership(self, x: float) -> float:
        """Вычисляет степень принадлежности μ(x) ∈ [0,1]"""
        if self.shape == "triangle":
            a, b, c = self.params
            if a <= x <= b:
                return (x - a) / (b - a) if b != a else 0
            elif b < x <= c:
                return (c - x) / (c - b) if c != b else 0
            return 0
        elif self.shape == "trapezoid":
            a, b, c, d = self.params
            if a <= x <= b:
                return (x - a) / (b - a) if b != a else 0
            elif b < x < c:
                return 1.0
            elif c <= x <= d:
                return (d - x) / (d - c) if d != c else 0
            return 0
        return 0

class FuzzyInferenceEngine:
    """Минимальный движок нечёткого вывода (Mamdani)"""
    
    def __init__(self):
        # Лингвистические переменные
        self.variables = {
            "budget": {
                "low": FuzzyTerm("low", [0, 0, 30000, 60000], "trapezoid"),
                "medium": FuzzyTerm("medium", [40000, 70000, 100000], "triangle"),
                "high": FuzzyTerm("high", [80000, 150000, 300000, 300000], "trapezoid"),
            },
            "knowledge_level": {
                "beginner": FuzzyTerm("beginner", [0, 0, 3, 5], "trapezoid"),
                "intermediate": FuzzyTerm("intermediate", [3, 5, 7], "triangle"),
                "advanced": FuzzyTerm("advanced", [5, 8, 10, 10], "trapezoid"),
            },
            "time_availability": {
                "short": FuzzyTerm("short", [0, 0, 2, 4], "trapezoid"),
                "medium": FuzzyTerm("medium", [2, 4, 6], "triangle"),
                "long": FuzzyTerm("long", [4, 7, 12, 12], "trapezoid"),
            },
            "career_focus": {
                "employment": FuzzyTerm("employment", [0, 0, 1], "triangle"),  # 0/1
                "academic": FuzzyTerm("academic", [0, 0.5, 1], "triangle"),
                "personal": FuzzyTerm("personal", [0, 1, 1], "triangle"),
            }
        }
        
        # Правила нечёткого вывода (упрощённые)
        self.rules = [
            {
                "id": "R001",
                "name": "Быстрый старт в профессии",
                "conditions": [
                    {"var": "career_focus", "term": "employment", "weight": 1.0},
                    {"var": "knowledge_level", "term": "beginner", "weight": 0.9},
                    {"var": "time_availability", "term": "short", "weight": 0.8},
                ],
                "conclusion": {"type": "practical_intensive", "boost": 0.3},
                "priority": 1
            },
            {
                "id": "R002",
                "name": "Академическая глубина",
                "conditions": [
                    {"var": "career_focus", "term": "academic", "weight": 1.0},
                    {"var": "knowledge_level", "term": "intermediate", "weight": 0.8},
                ],
                "conclusion": {"type": "theoretical_comprehensive", "boost": 0.25},
                "priority": 2
            },
            {
                "id": "R003",
                "name": "Бюджетный выбор",
                "conditions": [
                    {"var": "budget", "term": "low", "weight": 1.0},
                ],
                "conclusion": {"type": "budget_friendly", "boost": 0.2},
                "priority": 3
            },
            {
                "id": "R004",
                "name": "Премиум-образование",
                "conditions": [
                    {"var": "budget", "term": "high", "weight": 1.0},
                    {"var": "career_focus", "term": "employment", "weight": 0.7},
                ],
                "conclusion": {"type": "premium_certified", "boost": 0.35},
                "priority": 2
            },
            {
                "id": "R005",
                "name": "Саморазвитие",
                "conditions": [
                    {"var": "career_focus", "term": "personal", "weight": 1.0},
                    {"var": "time_availability", "term": "long", "weight": 0.6},
                ],
                "conclusion": {"type": "flexible_selfpaced", "boost": 0.15},
                "priority": 4
            }
        ]
    
    def evaluate_variable(self, var_name: str, value: float) -> Dict[str, float]:
        """Возвращает степени принадлежности для всех термов переменной"""
        if var_name not in self.variables:
            return {}
        return {
            term_name: term.membership(value)
            for term_name, term in self.variables[var_name].items()
        }
    
    def infer(self, user_profile: Dict[str, float]) -> Dict[str, float]:
        """
        Основной метод вывода.
        user_profile: {"budget": 50000, "knowledge_level": 2, "time_availability": 3, "career_focus": 1}
        Возвращает: {"practical_intensive": 0.45, "theoretical_comprehensive": 0.12, ...}
        """
        # 1. Fuzzification: вычисляем степени принадлежности для всех входных переменных
        fuzzified = {}
        for var_name in self.variables:
            if var_name in user_profile:
                fuzzified[var_name] = self.evaluate_variable(var_name, user_profile[var_name])
        
        # 2. Rule evaluation: оцениваем каждое правило
        conclusions = {}
        for rule in self.rules:
            # Вычисляем степень активации правила (min-operator для AND)
            activation = 1.0
            for cond in rule["conditions"]:
                var, term = cond["var"], cond["term"]
                weight = cond.get("weight", 1.0)
                if var in fuzzified and term in fuzzified[var]:
                    activation = min(activation, fuzzified[var][term] * weight)
                else:
                    activation = 0
                    break
            
            if activation > 0.1:  # порог срабатывания
                concl_type = rule["conclusion"]["type"]
                boost = rule["conclusion"]["boost"]
                # Агрегация: максимум по одинаковым выводам
                conclusions[concl_type] = max(
                    conclusions.get(concl_type, 0),
                    activation * boost * (rule["priority"] / 4)  # нормировка по приоритету
                )
        
        # 3. Нормализация выводов
        total = sum(conclusions.values())
        if total > 0:
            conclusions = {k: v/total for k, v in conclusions.items()}
        
        return conclusions
    
    def get_trace(self, user_profile: Optional[Dict[str, float]]) -> List[Dict]:
        """Возвращает трассировку для объяснений с нормализованными активациями."""
        if not user_profile:
            return []

        trace: List[Dict] = []
        fuzzified = {
            var: self.evaluate_variable(var, val)
            for var, val in user_profile.items()
            if var in self.variables
        }

        if not fuzzified:
            return []

        for rule in self.rules:
            conditions = rule.get("conditions") or []
            if not conditions:
                logger.warning("Правило %s без условий — пропуск", rule.get("id", "?"))
                continue

            activation = 1.0
            details: List[str] = []
            for cond in conditions:
                var, term = cond["var"], cond["term"]
                if var not in fuzzified or term not in fuzzified[var]:
                    activation = 0.0
                    break
                deg = fuzzified[var][term]
                details.append(f"{var}={term}: {deg:.2f}")
                activation = min(activation, deg * cond.get("weight", 1.0))

            if not details:
                continue

            if activation > 0.1:
                trace.append({
                    "rule_id": rule["id"],
                    "rule_name": rule["name"],
                    "activation": activation,
                    "activation_raw": activation,
                    "details": details,
                    "conclusion": rule["conclusion"]["type"],
                })

        if not trace:
            return []

        max_act = max(t["activation"] for t in trace)
        if max_act > 0:
            for t in trace:
                t["activation_normalized"] = round(t["activation"] / max_act, 4)
                t["activation"] = round(t["activation_normalized"], 4)
        else:
            for t in trace:
                t["activation_normalized"] = 0.0

        return trace
    
    def calibrate_from_data(self, df: "pd.DataFrame" = None, budget_col: str = "price", duration_col: str = "duration_weeks"):
        """Калибрует термы по эмпирическим квантилям датасета (§2.2)"""
        
        if df is None:
            return  # используем дефолтные значения
            
        # Бюджет
        if budget_col in df.columns:
            q25, q50, q75 = df[budget_col].quantile([0.25, 0.50, 0.75])
            max_b = df[budget_col].max()
            self.variables["budget"]["low"].params = [0, 0, q25, q50]
            self.variables["budget"]["medium"].params = [q25, q50, q75]
            self.variables["budget"]["high"].params = [q50, q75, max_b, max_b]
            
        # Длительность (если доступна)
        if duration_col in df.columns and df[duration_col].notna().sum() > 0:
            d_q = df[duration_col].dropna().quantile([0.33, 0.66])
            self.variables["time_availability"]["short"].params = [0, 0, 2, d_q.iloc[0]]
            self.variables["time_availability"]["medium"].params = [2, d_q.iloc[0], d_q.iloc[1]]
            self.variables["time_availability"]["long"].params = [d_q.iloc[0], d_q.iloc[1], 24, 24]
            
        logger.info("Термы нечётких переменных откалиброваны по квантилям датасета")