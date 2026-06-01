"""Юнит-тесты диалогового менеджера и NLU (§2.4)."""
import unittest

from app.services.dialog_manager import (
    BeliefState,
    apply_nlu_to_belief,
    process_dialog_turn,
)
from app.services.nlu_parser import NLUParser


class TestBudgetClarification(unittest.TestCase):
    """Кейс 1: неполный профиль → вопрос про бюджет."""

    def test_next_question_asks_budget_when_empty(self):
        belief = BeliefState()
        nlu = NLUParser().parse("хочу учиться в IT")
        updated, question, ready, _meta = process_dialog_turn(
            belief, "хочу учиться в IT", nlu
        )
        self.assertFalse(ready)
        self.assertIsNotNone(question)
        self.assertIn("бюджет", question.lower())

    def test_nlu_extracts_budget_from_text(self):
        nlu = NLUParser().parse("мой бюджет 50000 рублей, хочу в IT")
        self.assertIn("budget", nlu["entities"])
        self.assertEqual(nlu["entities"]["budget"]["value"], 50000.0)


class TestSlotConflict(unittest.TestCase):
    """Кейс 2: конфликтующие обновления слота budget."""

    def test_conflict_detected_and_clarification_question(self):
        belief = BeliefState()
        belief.update("budget", "low", 0.9)
        belief.update("budget", "high", 0.88)

        self.assertTrue(belief.has_conflict("budget"))
        self.assertIn("budget", belief.get_conflicts())

        question = belief.next_question()
        self.assertIsNotNone(question)
        self.assertIn("противоречие", question.lower())

    def test_conflict_blocks_readiness(self):
        belief = BeliefState()
        belief.update("budget", "medium", 0.85)
        belief.update("budget", "high", 0.85)
        belief.update("career_focus", "employment", 0.9)

        self.assertFalse(belief.is_ready_for_recommend())


class TestReadyForRecommend(unittest.TestCase):
    """Кейс 3: достаточное заполнение → готовность к рекомендациям."""

    def test_ready_when_required_slots_filled(self):
        belief = BeliefState()
        apply_nlu_to_belief(
            belief,
            {
                "budget": {"value": 50000, "confidence": 0.9},
                "career_focus": {"value": "employment", "confidence": 0.85},
                "knowledge_level": {"value": 5, "confidence": 0.8},
            },
        )
        belief.update("goals", "получить навыки Python для работы", 0.7)

        self.assertTrue(belief.is_ready_for_recommend())
        self.assertIsNone(belief.next_question())

    def test_dialog_turn_ready_via_api_flow(self):
        belief = BeliefState()
        parser = NLUParser()
        msg = "бюджет 60000 руб, цель — трудоустройство в IT, уровень 4"
        nlu = parser.parse(msg)
        updated, question, ready, meta = process_dialog_turn(belief, msg, nlu)

        self.assertTrue(ready)
        self.assertIsNone(question)
        self.assertIn("budget", updated.to_profile_dict())
        self.assertIn("career_focus", updated.to_profile_dict())
        self.assertLess(meta["entropy"], 0.6)


class TestBeliefStateMath(unittest.TestCase):
    def test_confidence_merge_formula(self):
        b = BeliefState()
        b.update("budget", 50000, 0.5)
        conf1 = b.slots["budget"]["source_conf"]
        b.update("budget", 60000, 0.5)
        conf2 = b.slots["budget"]["source_conf"]
        expected = conf1 + 0.5 * (1 - conf1)
        self.assertAlmostEqual(conf2, expected, places=5)

    def test_entropy_decreases_after_update(self):
        b = BeliefState()
        h0 = b.slot_entropy("budget")
        b.update("budget", "medium", 0.85)
        h1 = b.slot_entropy("budget")
        self.assertLess(h1, h0)


if __name__ == "__main__":
    unittest.main()
