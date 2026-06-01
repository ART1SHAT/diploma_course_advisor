# app/recommender.py — Гибридный скоринг v2 (семантика + калиброванный fuzzy)
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.course_loader import CourseLoader
from app.fuzzy_engine import FuzzyInferenceEngine
from app.semantic_search import SemanticIndex
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
        """Текстовый запрос для семантики; пустой профиль → пустая строка."""
        goals = (profile.goals or "").strip()
        interests = " ".join(profile.interests or [])
        return f"{goals} {interests}".strip()

    def _semantic_score(self, profile: UserProfile, course: Dict) -> float:
        """Безопасный семантический скор; без запроса — нейтральный 0.5."""
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
        """Нечёткий скор на основе правил; без входов — нейтральный 0.5."""
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

    def recommend(self, profile: UserProfile, top_k: int = 5) -> List[Dict]:
        scored = []
        for course in self.courses:
            price = float(course.get("price", 0) or 0)
            if profile.budget is not None and profile.budget > 0:
                if price > profile.budget * 1.5:
                    continue

            sem = self._semantic_score(profile, course)
            fuzzy = self._fuzzy_score(profile, course)
            final = 0.55 * sem + 0.35 * fuzzy + 0.10 * (1.0 if price == 0 else 0.5)
            scored.append({**course, "score": round(final, 3)})

        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]

    def _extract_evidence(
        self, query: str, description: str, max_len: int = 140
    ) -> Optional[str]:
        """Цитата из описания курса при совпадении с запросом."""
        if not query or not description:
            return None

        desc = description.strip()
        desc_lower = desc.lower()
        query_lower = query.lower().strip()

        idx = desc_lower.find(query_lower)
        needle = query_lower
        if idx < 0:
            tokens = [w for w in re.split(r"\W+", query_lower) if len(w) > 4]
            for token in tokens:
                idx = desc_lower.find(token)
                if idx >= 0:
                    needle = token
                    break
            if idx < 0:
                return None

        start = max(0, idx - 50)
        end = min(len(desc), idx + len(needle) + 70)
        snippet = desc[start:end].strip()
        if start > 0:
            snippet = "…" + snippet
        if end < len(desc):
            snippet = snippet + "…"
        return snippet[:max_len] if len(snippet) > max_len else snippet

    def _build_semantic_block(
        self, profile: UserProfile, course: Dict
    ) -> List[Dict[str, Any]]:
        """Семантические совпадения (цель, интересы)."""
        items: List[Dict[str, Any]] = []
        query = self._profile_query(profile)
        if not query:
            return items

        score = self._semantic_score(profile, course)
        description = str(course.get("description", "") or "")
        goals = (profile.goals or "").strip()

        if goals and goals.lower() in description.lower():
            items.append({
                "match_type": "goal",
                "query": goals,
                "score": round(score, 3),
                "summary": f"Соответствует цели: «{goals}»",
            })
        elif profile.interests:
            matched = [
                i for i in profile.interests
                if i.lower() in description.lower()
            ]
            if matched:
                items.append({
                    "match_type": "interests",
                    "query": ", ".join(matched),
                    "score": round(score, 3),
                    "summary": f"Совпадение по интересам: {', '.join(matched[:3])}",
                })
        elif score > NEUTRAL_SCORE:
            items.append({
                "match_type": "semantic",
                "query": query,
                "score": round(score, 3),
                "summary": "Семантическая близость к запросу",
            })

        return items

    def _build_budget_block(
        self, profile: UserProfile, course: Dict
    ) -> Optional[Dict[str, Any]]:
        """Проверка бюджета (включая budget=0 как «только бесплатные»)."""
        if profile.budget is None:
            return None

        price = float(course.get("price", 0) or 0)
        user_budget = float(profile.budget)

        if user_budget <= 0:
            within = price == 0
            summary = (
                "Бесплатный курс — соответствует нулевому бюджету"
                if within
                else f"Платный курс ({int(price)} ₽) при нулевом бюджете"
            )
        else:
            within = price <= user_budget
            if price == 0:
                summary = "Бесплатный курс"
            elif within:
                summary = f"Укладывается в бюджет ({int(price)} ₽ ≤ {int(user_budget)} ₽)"
            else:
                summary = f"Выше бюджета ({int(price)} ₽ > {int(user_budget)} ₽)"

        return {
            "checked": True,
            "within_budget": within,
            "price": price,
            "user_budget": user_budget,
            "summary": summary,
        }

    def _build_fuzzy_rules_block(self, trace: List[Dict]) -> List[Dict[str, Any]]:
        """Активированные нечёткие правила из трассировки."""
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

    def _compute_confidence(
        self,
        semantic_score: float,
        fuzzy_score: float,
        trace: List[Dict],
        budget_block: Optional[Dict[str, Any]],
    ) -> float:
        """Сводная уверенность 0.0–1.0."""
        if trace:
            fuzzy_part = float(np.mean([t["activation"] for t in trace]))
        else:
            fuzzy_part = NEUTRAL_SCORE

        budget_part = NEUTRAL_SCORE
        if budget_block and budget_block.get("checked"):
            budget_part = 1.0 if budget_block.get("within_budget") else 0.25

        raw = 0.45 * semantic_score + 0.35 * fuzzy_part + 0.20 * budget_part
        return round(max(0.0, min(1.0, raw)), 3)

    def _flatten_explanations(
        self,
        semantic: List[Dict],
        fuzzy_rules: List[Dict],
        budget_block: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Плоский список для обратной совместимости с UI."""
        lines: List[str] = []
        for s in semantic:
            lines.append("✓ " + s["summary"])
        for r in fuzzy_rules:
            lines.append("✓ " + r["summary"])
        if budget_block and budget_block.get("checked"):
            prefix = "✓ " if budget_block.get("within_budget") else "○ "
            lines.append(prefix + budget_block["summary"])
        return lines or ["✓ Общее соответствие запросу (нейтральный скоринг)"]

    def explain(self, profile: UserProfile, course: Dict) -> Dict:
        """Структурированное объяснение рекомендации."""
        fuzzy_input = profile.to_fuzzy_input()
        trace = self.fuzzy_engine.get_trace(fuzzy_input)

        semantic = self._build_semantic_block(profile, course)
        fuzzy_rules = self._build_fuzzy_rules_block(trace)
        budget_block = self._build_budget_block(profile, course)

        query = self._profile_query(profile)
        evidence = self._extract_evidence(
            query or (profile.goals or ""),
            str(course.get("description", "") or ""),
        )

        sem_score = self._semantic_score(profile, course)
        fuzzy_score = self._fuzzy_score(profile, course)
        confidence = self._compute_confidence(
            sem_score, fuzzy_score, trace, budget_block
        )

        return {
            "course_id": str(course.get("id")),
            "confidence": confidence,
            "evidence": evidence,
            "semantic": semantic,
            "fuzzy_rules": fuzzy_rules,
            "budget": budget_block,
            "explanations": self._flatten_explanations(
                semantic, fuzzy_rules, budget_block
            ),
            "fuzzy_trace": trace,
            "scores": {
                "semantic": round(sem_score, 3),
                "fuzzy": round(fuzzy_score, 3),
            },
        }

    def get_stats(self) -> Dict:
        return self.loader.get_stats()
