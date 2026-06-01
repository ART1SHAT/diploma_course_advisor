"""Тесты ExplanationGenerator, audit log и контрфактуалов (§2.3, §3.3)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.core.audit_log import log_explanation_trace, profile_hash
from app.services.explainer import (
    ExplanationGenerator,
    apply_profile_changes,
    build_what_if_explanation,
    find_course_rank,
)
from app.user_profile import UserProfile


class TestExplanationGenerator(unittest.TestCase):
    def setUp(self):
        self.profile = UserProfile(
            budget=50000,
            knowledge_level=5,
            time_availability=8,
            career_focus=1.0,
            interests=["python", "данные"],
            goals="работа в IT",
        )
        self.course = {
            "id": "c1",
            "title": "Python для анализа данных",
            "description": "Курс по python и анализу данных для работы в IT",
            "price": 40000,
            "skills": ["python", "pandas"],
        }
        self.trace = [
            {
                "rule_id": "R001",
                "rule_name": "Быстрый старт",
                "activation": 0.78,
                "details": ["career_focus=employment: 0.90"],
                "conclusion": "practical_intensive",
            }
        ]

    def test_generate_returns_reason_types(self):
        gen = ExplanationGenerator(
            self.profile, self.course, self.trace, semantic_sim=0.82
        )
        result = gen.generate()

        self.assertIn("reasons", result)
        self.assertIn("confidence", result)
        types = {r["type"] for r in result["reasons"]}
        self.assertIn("fuzzy_rule", types)
        self.assertIn("semantic_match", types)
        self.assertIn("budget_constraint", types)
        self.assertIn("competency_align", types)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_reasons_have_weight_and_text(self):
        gen = ExplanationGenerator(
            self.profile, self.course, self.trace, semantic_sim=0.7
        )
        for r in gen.generate()["reasons"]:
            self.assertIn(r["type"], (
                "fuzzy_rule",
                "semantic_match",
                "budget_constraint",
                "competency_align",
            ))
            self.assertTrue(r["text"])
            self.assertGreater(r["weight"], 0)


class TestAuditLog(unittest.TestCase):
    def test_log_writes_json_with_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            entry = log_explanation_trace(
                course_id="99",
                profile_data={"budget": 50000, "goals": "IT"},
                fuzzy_trace=[
                    {"rule_id": "R001", "activation": 0.8},
                    {"rule_id": "R002", "activation": 0.05},
                ],
                confidence=0.71,
                reason_types=["fuzzy_rule", "semantic_match"],
                log_path=path,
            )
            self.assertIn("timestamp", entry)
            self.assertEqual(len(entry["profile_hash"]), 16)
            self.assertEqual(entry["activated_rules"], ["R001"])

            line = path.read_text(encoding="utf-8").strip()
            parsed = json.loads(line)
            self.assertEqual(parsed["course_id"], "99")


class TestCounterfactual(unittest.TestCase):
    def test_apply_profile_changes(self):
        p = UserProfile(budget=50000, goals="IT")
        p2 = apply_profile_changes(p, {"budget": 60000})
        self.assertEqual(p2.budget, 60000)
        self.assertEqual(p2.goals, "IT")

    def test_build_what_if_explanation_budget_up(self):
        text = build_what_if_explanation("budget", 50000, 60000, delta_rank=2)
        self.assertIn("бюджет", text.lower())
        self.assertIn("2", text)
        self.assertIn("поднимется", text.lower())

    def test_find_course_rank(self):
        ranked = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        self.assertEqual(find_course_rank(ranked, "b"), (2, 3))
        self.assertEqual(find_course_rank(ranked, "x"), (0, 3))

    @patch("app.api.routes.get_recommender")
    def test_what_if_rank_improves_with_budget(self, mock_get):
        courses = [
            {"id": "1", "price": 55000, "provider_type": "bootcamp", "description": "IT"},
            {"id": "2", "price": 30000, "provider_type": "bootcamp", "description": "IT"},
            {"id": "3", "price": 0, "provider_type": "government_platform", "description": "IT"},
        ]

        mock_rec = MagicMock()

        def score_side_effect(profile):
            budget = profile.budget or 0
            scored = []
            for c in courses:
                price = c["price"]
                if budget > 0 and price > budget * 1.5:
                    continue
                if str(c["id"]) == "1":
                    score = 0.95 if budget >= 90000 else 0.55
                elif str(c["id"]) == "2":
                    score = 0.85
                else:
                    score = 0.5
                scored.append({**c, "score": score})
            return sorted(scored, key=lambda x: x["score"], reverse=True)

        mock_rec.score_all.side_effect = score_side_effect
        mock_get.return_value = mock_rec

        from fastapi.testclient import TestClient
        from app.main import create_app

        client = TestClient(create_app())
        resp = client.post(
            "/api/explain/what_if",
            json={
                "course_id": "1",
                "base_profile": {
                    "budget": 50000,
                    "career_focus": 1.0,
                    "goals": "IT",
                },
                "changed_profile": {"budget": 100000},
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertGreater(data["delta_rank"], 0)
        self.assertIn("поднимется", data["explanation"].lower())


class TestProfileHash(unittest.TestCase):
    def test_stable_hash(self):
        h1 = profile_hash({"budget": 1, "goals": "a"})
        h2 = profile_hash({"goals": "a", "budget": 1})
        self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
