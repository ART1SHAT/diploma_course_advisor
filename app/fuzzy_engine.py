# app/fuzzy_engine.py
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class FuzzyTerm:
    """Лингвистический терм с функцией принадлежности"""
    name: str
    params: List[float]
    shape: str = "triangle"
    
    def membership(self, x: float) -> float:
        """Вычисляет степень принадлежности μ(x) ∈ [0,1]"""
        if self.shape == "triangle":
            a, b, c = self.params
            if a <= x <= b:
                return (x - a) / (b - a) if b != a else 0.0
            elif b < x <= c:
                return (c - x) / (c - b) if c != b else 0.0
            return 0.0
        elif self.shape == "trapezoid":
            a, b, c, d = self.params
            if a <= x <= b:
                return (x - a) / (b - a) if b != a else 0.0
            elif b < x < c:
                return 1.0
            elif c <= x <= d:
                return (d - x) / (d - c) if d != c else 0.0
            return 0.0
        return 0.0

class FuzzyInferenceEngine:
    """Движок нечёткого вывода (Mamdani)"""
    
    def __init__(self):
        self.variables = {
            "budget": {
                "low": FuzzyTerm("low", [0, 0, 30000, 60000], "trapezoid"),
                "medium": FuzzyTerm("medium", [20000, 50000, 100000], "triangle"),
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
                "long": FuzzyTerm("long", [4, 7, 15, 20], "trapezoid"),
            },
            "career_focus": {
                "employment": FuzzyTerm("employment", [0.5, 1.0, 1.0], "triangle"),
                "academic": FuzzyTerm("academic", [0.0, 0.5, 1.0], "triangle"),
                "personal": FuzzyTerm("personal", [0.0, 0.0, 0.5], "triangle"),
            }
        }
        
        self.rules = [
            {
                "id": "R001",
                "name": "Быстрый старт в профессии",
                "antecedents": [
                    {"var": "career_focus", "term": "employment", "weight": 1.0},
                    {"var": "knowledge_level", "term": "beginner", "weight": 0.9},
                    {"var": "time_availability", "term": "short", "weight": 0.8},
                ],
                "consequent": {"type": "practical_intensive", "boost": 0.6},
                "priority": 1
            },
            {
                "id": "R002",
                "name": "Академическая глубина",
                "antecedents": [
                    {"var": "career_focus", "term": "academic", "weight": 1.0},
                    {"var": "knowledge_level", "term": "intermediate", "weight": 0.8},
                ],
                "consequent": {"type": "theoretical_comprehensive", "boost": 0.5},
                "priority": 2
            },
            {
                "id": "R003",
                "name": "Бюджетный выбор",
                "antecedents": [
                    {"var": "budget", "term": "low", "weight": 1.0},
                ],
                "consequent": {"type": "budget_friendly", "boost": 0.7},
                "priority": 1
            },
            {
                "id": "R004",
                "name": "Премиум-образование",
                "antecedents": [
                    {"var": "budget", "term": "high", "weight": 1.0},
                    {"var": "career_focus", "term": "employment", "weight": 0.7},
                ],
                "consequent": {"type": "premium_certified", "boost": 0.5},
                "priority": 2
            },
            {
                "id": "R005",
                "name": "Саморазвитие",
                "antecedents": [
                    {"var": "career_focus", "term": "personal", "weight": 1.0},
                    {"var": "time_availability", "term": "long", "weight": 0.6},
                ],
                "consequent": {"type": "flexible_selfpaced", "boost": 0.4},
                "priority": 3
            },
            {
                "id": "R006",
                "name": "Продвинутый уровень",
                "antecedents": [
                    {"var": "knowledge_level", "term": "advanced", "weight": 1.0},
                ],
                "consequent": {"type": "theoretical_comprehensive", "boost": 0.6},
                "priority": 2
            },
            {
                "id": "R007",
                "name": "Много времени на обучение",
                "antecedents": [
                    {"var": "time_availability", "term": "long", "weight": 1.0},
                ],
                "consequent": {"type": "theoretical_comprehensive", "boost": 0.5},
                "priority": 2
            },
            {
                "id": "R008",
                "name": "Средний бюджет",
                "antecedents": [
                    {"var": "budget", "term": "medium", "weight": 1.0},
                ],
                "consequent": {"type": "practical_intensive", "boost": 0.5},
                "priority": 2
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
        """Основной метод вывода"""
        fuzzified = {}
        for var_name, value in user_profile.items():
            if var_name in self.variables:
                fuzzified[var_name] = self.evaluate_variable(var_name, float(value))
        
        conclusions = {}
        for rule in self.rules:
            activation = 1.0
            for cond in rule["antecedents"]:
                var, term = cond["var"], cond["term"]
                weight = cond.get("weight", 1.0)
                if var in fuzzified and term in fuzzified[var]:
                    activation = min(activation, fuzzified[var][term] * weight)
                else:
                    activation = 0.0
                    break
            
            if activation > 0.1:
                concl_type = rule["consequent"]["type"]
                boost = rule["consequent"]["boost"]
                priority_factor = rule["priority"] / 4.0
                conclusions[concl_type] = max(
                    conclusions.get(concl_type, 0.0),
                    activation * boost * priority_factor
                )
        
        total = sum(conclusions.values())
        if total > 0:
            conclusions = {k: v / total for k, v in conclusions.items()}
        
        return conclusions
    
    def get_trace(self, user_profile: Optional[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Возвращает трассировку для объяснений"""
        if not user_profile:
            return []

        trace: List[Dict[str, Any]] = []
        fuzzified = {
            var: self.evaluate_variable(var, float(val))
            for var, val in user_profile.items()
            if var in self.variables
        }

        if not fuzzified:
            return []

        for rule in self.rules:
            conditions = rule.get("antecedents") or []
            if not conditions:
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
                    "activation": float(activation),
                    "details": details,
                    "conclusion": rule["consequent"]["type"],
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
        """Калибрует термы по эмпирическим квантилям датасета"""
        if df is None:
            return
            
        if budget_col in df.columns:
            q25, q50, q75 = df[budget_col].quantile([0.25, 0.50, 0.75])
            max_b = df[budget_col].max()
            self.variables["budget"]["low"].params = [0, 0, float(q25), float(q50)]
            self.variables["budget"]["medium"].params = [float(q25), float(q50), float(q75)]
            self.variables["budget"]["high"].params = [float(q50), float(q75), float(max_b), float(max_b)]
            
        if duration_col in df.columns and df[duration_col].notna().sum() > 0:
            d_q = df[duration_col].dropna().quantile([0.33, 0.66])
            self.variables["time_availability"]["short"].params = [0, 0, 2, float(d_q.iloc[0])]
            self.variables["time_availability"]["medium"].params = [2, float(d_q.iloc[0]), float(d_q.iloc[1])]
            self.variables["time_availability"]["long"].params = [float(d_q.iloc[0]), float(d_q.iloc[1]), 24, 24]
            
        logger.info("Термы нечётких переменных откалиброваны по квантилям датасета")