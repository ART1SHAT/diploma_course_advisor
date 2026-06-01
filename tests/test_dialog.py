"""Тесты диалогового менеджера, NLU и политики (§2.4)."""
import unittest

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.belief_state import BeliefState
from app.services.dialog_manager import apply_nlu_to_belief, process_dialog_turn
from app.services.dialog_policy import select_next_action
from app.services.nlu_parser import parse_user_message


class TestNLUParsing(unittest.TestCase):
    def test_parse_budget_and_career(self):
        result = parse_user_message("хочу курс до 50 тысяч для работы")
        self.assertIn("budget", result["entities"])
        self.assertIn("career_focus", result["entities"])
        self.assertEqual(result["entities"]["budget"]["term"], "medium")
        self.assertEqual(result["entities"]["career_focus"]["term"], "employment")

    def test_parse_budget_fuzzy_cheap(self):
        result = parse_user_message("недорогой курс по python")
        self.assertEqual(result["entities"]["budget"]["term"], "low")


class TestBeliefAccumulation(unittest.TestCase):
    def test_confidence_increases_with_messages(self):
        belief = BeliefState()
        nlu1 = parse_user_message("бюджет 50000 рублей")
        apply_nlu_to_belief(belief, nlu1)
        c1 = belief.get_slot_confidence("budget")

        nlu2 = parse_user_message("бюджет 55000")
        apply_nlu_to_belief(belief, nlu2)
        c2 = belief.get_slot_confidence("budget")

        self.assertGreaterEqual(c2, c1)
        self.assertGreater(c2, 0.5)

    def test_merge_formula(self):
        b = BeliefState()
        b.update("budget", "medium", 50_000, 0.5)
        c1 = b.get_slot_confidence("budget")
        b.update("budget", "medium", 55_000, 0.5)
        c2 = b.get_slot_confidence("budget")
        self.assertAlmostEqual(c2, c1 + 0.5 * (1 - c1), places=5)


class TestDialogPolicy(unittest.TestCase):
    def test_asks_when_low_confidence(self):
        belief = BeliefState()
        action = select_next_action(belief, {"intent": "unknown", "entities": {}})
        self.assertEqual(action["action"], "ask_question")
        self.assertIsNotNone(action["question"])

    def test_shows_recommendations_when_confident(self):
        belief = BeliefState()
        apply_nlu_to_belief(
            belief,
            parse_user_message("бюджет 60000, трудоустройство в IT, уровень 5, 8 часов в неделю"),
        )
        belief.update("goals", "работа в IT", 0.0, 0.85)
        action = select_next_action(belief, {"intent": "unknown", "entities": {}})
        self.assertEqual(action["action"], "show_recommendations")

    def test_ask_recommend_intent(self):
        belief = BeliefState()
        apply_nlu_to_belief(belief, parse_user_message("бюджет 70000, для работы"))
        action = select_next_action(
            belief, {"intent": "ask_recommend", "entities": {}}
        )
        self.assertIn(
            action["action"],
            ("show_recommendations", "ask_question"),
        )


class TestDialogApi(unittest.TestCase):
    def test_start_and_step_flow(self):
        client = TestClient(create_app())
        start = client.post("/api/dialog/start")
        self.assertEqual(start.status_code, 200)
        sid = start.json()["session_id"]
        self.assertIsNotNone(sid)

        step = client.post(
            "/api/dialog/step",
            json={
                "session_id": sid,
                "user_message": "бюджет 60000, хочу работу в IT, 6 часов в неделю",
            },
        )
        self.assertEqual(step.status_code, 200, step.text)
        data = step.json()
        self.assertIn("next_action", data)
        self.assertIn("belief_summary", data)


class TestProcessDialogTurn(unittest.TestCase):
    def test_turn_ready_with_full_profile(self):
        belief = BeliefState()
        msg = "бюджет 60000 руб, трудоустройство в IT, уровень 4, 8 часов в неделю"
        nlu = parse_user_message(msg)
        updated, question, ready, _meta = process_dialog_turn(belief, msg, nlu)
        self.assertTrue(ready)
        self.assertIsNone(question)
        profile = updated.to_user_profile()
        self.assertIsNotNone(profile.budget)
        self.assertIsNotNone(profile.career_focus)


if __name__ == "__main__":
    unittest.main()
