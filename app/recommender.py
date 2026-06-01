# app/recommender.py — Гибридный скоринг v2 (семантика + калиброванный fuzzy)
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.audit_log import log_explanation_trace
from app.course_loader import CourseLoader
from app.fuzzy_engine import FuzzyInferenceEngine
from app.semantic_search import SemanticIndex
from app.services.explainer import ExplanationGenerator, profile_to_dict
from app.user_profile import UserProfile

logger = logging.getLogger(__name__)

NEUTRAL_SCORE = 0.5


class HybridRecommender:
    def __init__(self, json_path: str = "data/unified_courses.json"):
        self.fuzzy_engine = FuzzyInferenceEngine()
        self.loader = CourseLoader(json_path)
        self.courses = self.loader.load()

        self.sem_index = SemanticIndex()
        try:
            self.sem_index.build(self.courses)
        except Exception as e:
            logger.warning("Семантический индекс не построен: %s", e)

        try:
            df = pd.DataFrame(self.courses)
            self.fuzzy_engine.calibrate_from_data(df)
        except Exception as e:
            logger.warning("Калибровка нечётких термов пропущена: %s", e)

    def _profile_query(self, profile: UserProfile) -> str:
        goals = (profile.goals or "").strip()
        interests = " ".join(profile.interests or [])
        return f"{goals} {interests}".strip()

    def _semantic_score(self, profile: UserProfile, course: Dict) -> float:
        try:
            query = self._profile_query(profile)
            if not query or not getattr(self, "sem_index", None):
                return NEUTRAL_SCORE

            for c, score in self.sem_index.search(query, top_k=100):
                if str(c.get("id")) == str(course.get("id")):
                    return max(0.0, min(1.0, float(score)))
            return NEUTRAL_SCORE
        except Exception as e:
            logger.warning("_semantic_score fallback: %s", e)
            return NEUTRAL_SCORE

    def _fuzzy_score(self, profile: UserProfile, course: Dict) -> float:
        fuzzy_input = profile.to_fuzzy_input()
        if not fuzzy_input:
            return NEUTRAL_SCORE

        conclusions = self.fuzzy_engine.infer(fuzzy_input)
        if not conclusions:
            return NEUTRAL_SCORE

        course_type = str(course.get("provider_type", ""))
        score = 0.0
        mapping = {
            "practical_intensive": ["bootcamp", "corporate"],
            "theoretical_comprehensive": ["university"],
            "budget_friendly": ["government_platform"],
            "premium_certified": ["bootcamp", "university"],
        }
        for concl_type, weight in conclusions.items():
            if course_type in mapping.get(concl_type, []):
                score += weight * 0.9

        return min(score, 1.0) if score > 0 else NEUTRAL_SCORE

    def _score_course(self, profile: UserProfile, course: Dict) -> float:
        price = float(course.get("price", 0) or 0)
        sem = self._semantic_score(profile, course)
        fuzzy = self._fuzzy_score(profile, course)
        return round(0.55 * sem + 0.35 * fuzzy + 0.10 * (1.0 if price == 0 else 0.5), 3)

    def score_all(self, profile: UserProfile) -> List[Dict]:
        """Полный ранжированный список курсов (для контрфактуалов)."""
        scored = []
        for course in self.courses:
            price = float(course.get("price", 0) or 0)
            if profile.budget is not None and profile.budget > 0:
                if price > profile.budget * 1.5:
                    continue
            scored.append({**course, "score": self._score_course(profile, course)})
        return sorted(scored, key=lambda x: x["score"], reverse=True)

    def recommend(self, profile: UserProfile, top_k: int = 5) -> List[Dict]:
        return self.score_all(profile)[:top_k]

    def recommend_from_belief(self, belief: Any, top_k: int = 5) -> List[Dict]:
        """Рекомендации по состоянию диалога (§2.4)."""
        from app.services.belief_state import BeliefState as _BeliefState

        if not isinstance(belief, _BeliefState):
            raise TypeError("Ожидается BeliefState")
        profile = belief.to_user_profile()
        return self.recommend(profile, top_k=top_k)

    def rank_course(self, profile: UserProfile, course_id: str) -> Dict[str, Any]:
        """Позиция курса в общем рейтинге (1-based)."""
        ranked = self.score_all(profile)
        for i, c in enumerate(ranked, start=1):
            if str(c.get("id")) == str(course_id):
                return {
                    "rank": i,
                    "total": len(ranked),
                    "score": c.get("score"),
                    "in_list": True,
                }
        return {"rank": 0, "total": len(ranked), "score": None, "in_list": False}

    @staticmethod
    def _fuzzy_rules_legacy(trace: List[Dict]) -> List[Dict[str, Any]]:
        rules: List[Dict[str, Any]] = []
        for t in trace:
            act = float(t.get("activation", 0))
            if act <= 0.1:
                continue
            rules.append({
                "rule_id": t["rule_id"],
                "rule_name": t["rule_name"],
                "activation": act,
                "activation_raw": t.get("activation_raw", act),
                "activation_normalized": t.get("activation_normalized", act),
                "details": t.get("details") or [],
                "conclusion": t.get("conclusion", ""),
                "summary": (
                    f"Активировано правило {t['rule_id']} "
                    f"({act:.0%}) — {t['rule_name']}"
                ),
            })
        return rules

    @staticmethod
    def _legacy_blocks_from_reasons(
        reasons: List[Dict[str, Any]],
    ) -> tuple[List[Dict], List[Dict], Optional[Dict]]:
        semantic: List[Dict] = []
        fuzzy_rules: List[Dict] = []
        budget_block: Optional[Dict] = None

        for r in reasons:
            if r["type"] == "semantic_match":
                semantic.append({
                    "match_type": "semantic",
                    "summary": r["text"],
                    "score": r["weight"],
                })
            elif r["type"] == "fuzzy_rule":
                fuzzy_rules.append({
                    "rule_id": r.get("rule_id"),
                    "rule_name": r["text"],
                    "activation": r["weight"],
                    "summary": r["text"],
                })
            elif r["type"] == "budget_constraint":
                within = r["weight"] >= 0.5
                budget_block = {
                    "checked": True,
                    "within_budget": within,
                    "summary": r["text"],
                }
        return semantic, fuzzy_rules, budget_block

    def explain(self, profile: UserProfile, course: Dict) -> Dict:
        """Структурированное объяснение через ExplanationGenerator (§2.3)."""
        fuzzy_input = profile.to_fuzzy_input()
        trace = self.fuzzy_engine.get_trace(fuzzy_input)
        sem_score = self._semantic_score(profile, course)
        fuzzy_score = self._fuzzy_score(profile, course)

        generated = ExplanationGenerator(
            profile, course, trace, sem_score
        ).generate()

        log_explanation_trace(
            course_id=str(course.get("id")),
            profile_data=profile_to_dict(profile),
            fuzzy_trace=trace,
            confidence=generated["confidence"],
            reason_types=[r["type"] for r in generated["reasons"]],
        )

        semantic, fuzzy_from_reasons, budget_block = self._legacy_blocks_from_reasons(
            generated["reasons"]
        )
        fuzzy_rules = self._fuzzy_rules_legacy(trace) or fuzzy_from_reasons

        explanations = []
        for r in generated["reasons"]:
            prefix = "✓ "
            if r["type"] == "budget_constraint" and r["weight"] < 0.5:
                prefix = "○ "
            explanations.append(prefix + r["text"])

        return {
            "course_id": str(course.get("id")),
            "confidence": generated["confidence"],
            "evidence": generated.get("evidence"),
            "reasons": generated["reasons"],
            "semantic": semantic,
            "fuzzy_rules": fuzzy_rules,
            "budget": budget_block,
            "explanations": explanations,
            "fuzzy_trace": trace,
            "scores": {
                "semantic": round(sem_score, 3),
                "fuzzy": round(fuzzy_score, 3),
            },
        }

    def get_stats(self) -> Dict:
        return self.loader.get_stats()
