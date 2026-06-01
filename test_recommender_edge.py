"""Крайние случаи recommender / fuzzy_engine."""
from app.recommender import HybridRecommender, NEUTRAL_SCORE
from app.user_profile import UserProfile
from app.fuzzy_engine import FuzzyInferenceEngine


def test_empty_profile_scores():
    rec = HybridRecommender(json_path="data/unified_courses.json")
    profile = UserProfile(
        budget=None,
        knowledge_level=None,
        time_availability=None,
        career_focus=None,
        interests=[],
        goals="",
    )
    course = rec.courses[0]
    sem = rec._semantic_score(profile, course)
    fuzzy = rec._fuzzy_score(profile, course)
    assert sem == NEUTRAL_SCORE, f"semantic expected 0.5, got {sem}"
    assert fuzzy == NEUTRAL_SCORE, f"fuzzy expected 0.5, got {fuzzy}"
    results = rec.recommend(profile, top_k=3)
    assert len(results) >= 1
    exp = rec.explain(profile, course)
    assert 0.0 <= exp["confidence"] <= 1.0
    assert exp["semantic"] == []
    assert exp["budget"] is None
    print("OK empty profile:", sem, fuzzy, exp["confidence"])


def test_budget_zero():
    rec = HybridRecommender(json_path="data/unified_courses.json")
    profile = UserProfile(budget=0, interests=[], goals="")
    results = rec.recommend(profile, top_k=5)
    assert results, "expected recommendations with budget=0"
    exp = rec.explain(profile, results[0])
    assert exp["budget"] is not None
    assert exp["budget"]["checked"] is True
    print("OK budget=0:", exp["budget"]["summary"][:60])


def test_empty_goals_interests():
    rec = HybridRecommender(json_path="data/unified_courses.json")
    profile = UserProfile(
        budget=50000,
        knowledge_level=5,
        time_availability=8,
        career_focus=1.0,
        interests=[],
        goals="",
    )
    results = rec.recommend(profile, top_k=3)
    assert results
    exp = rec.explain(profile, results[0])
    assert "fuzzy_rules" in exp and "semantic" in exp
    print("OK goals='' interests=[]:", len(exp["fuzzy_rules"]), "rules")


def test_get_trace_empty():
    engine = FuzzyInferenceEngine()
    assert engine.get_trace({}) == []
    assert engine.get_trace(None) == []  # type: ignore[arg-type]
    trace = engine.get_trace({"budget": 50000, "knowledge_level": 3})
    if trace:
        assert all(0 <= t["activation"] <= 1 for t in trace)
        assert "activation_normalized" in trace[0]
    print("OK get_trace empty + normalization")


if __name__ == "__main__":
    test_get_trace_empty()
    test_empty_profile_scores()
    test_budget_zero()
    test_empty_goals_interests()
    print("\nВсе крайние случаи пройдены.")
