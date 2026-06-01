"""
Состояние убеждений диалога (§2.4): распределения по лингвистическим термам.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List, Optional

from app.user_profile import UserProfile

# Словарь термов по слотам (согласован с fuzzy_engine)
SLOT_TERM_VOCAB: Dict[str, List[str]] = {
    "budget": ["low", "medium", "high"],
    "knowledge_level": ["beginner", "intermediate", "advanced"],
    "time_availability": ["short", "medium", "long"],
    "career_focus": ["employment", "academic", "personal"],
}

NUMERIC_TERM_ANCHORS: Dict[str, Dict[str, float]] = {
    "budget": {"low": 15_000, "medium": 50_000, "high": 120_000},
    "knowledge_level": {"beginner": 2, "intermediate": 5, "advanced": 8},
    "time_availability": {"short": 3, "medium": 8, "long": 15},
    "career_focus": {"personal": 0.0, "academic": 0.5, "employment": 1.0},
}

REQUIRED_SLOTS = ["budget", "career_focus"]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _normalize_terms(terms: Dict[str, float], vocab: List[str]) -> Dict[str, float]:
    total = sum(max(0.0, terms.get(t, 0.0)) for t in vocab)
    if total <= 0:
        n = len(vocab)
        return {t: 1.0 / n for t in vocab} if n else {}
    return {t: max(0.0, terms.get(t, 0.0)) / total for t in vocab}


def numeric_to_term(slot: str, value: float) -> str:
    anchors = NUMERIC_TERM_ANCHORS.get(slot, {})
    if not anchors:
        return str(value)
    return min(anchors, key=lambda t: abs(anchors[t] - value))


def term_to_numeric(slot: str, term: str) -> Optional[float]:
    return NUMERIC_TERM_ANCHORS.get(slot, {}).get(term)


class BeliefState:
    """
    Слоты: {slot_name: {term: belief, ..., "source_conf": float}}.
    Слияние: new_conf = old_conf + source_conf * (1 - old_conf).
    """

    def __init__(self) -> None:
        self._slots: Dict[str, Dict[str, float]] = {}
        self.goals_text: str = ""
        self.goals_conf: float = 0.0
        self.interests: List[str] = []
        self.turn_count: int = 0

    def _init_slot(self, slot: str) -> Dict[str, float]:
        vocab = SLOT_TERM_VOCAB.get(slot, [])
        entry: Dict[str, float] = {t: 0.0 for t in vocab}
        entry["source_conf"] = 0.0
        return entry

    def get_slot_confidence(self, slot: str) -> float:
        if slot == "goals":
            return self.goals_conf
        entry = self._slots.get(slot)
        if not entry:
            return 0.0
        return float(entry.get("source_conf", 0.0))

    def update(
        self,
        slot: str,
        term: str,
        value: float,
        source_conf: float,
    ) -> None:
        """Нечёткое слияние уверенности и усиление выбранного терма."""
        conf = _clamp01(float(source_conf))
        slot = slot.strip()

        if slot == "goals":
            if term or value:
                text = str(term) if isinstance(term, str) and term else str(value)
                if text.strip():
                    self.goals_text = text.strip()
            old = self.goals_conf
            self.goals_conf = _clamp01(old + conf * (1.0 - old))
            self.turn_count += 1
            return

        if slot == "interests":
            if isinstance(term, list):
                self.interests = list(term)
            elif term:
                self.interests = [str(term)]
            self.turn_count += 1
            return

        if slot not in SLOT_TERM_VOCAB:
            return

        vocab = SLOT_TERM_VOCAB[slot]
        entry = self._slots.setdefault(slot, self._init_slot(slot))

        # Определяем терм из value или явного term
        if term in vocab:
            chosen_term = term
        elif isinstance(value, (int, float)):
            chosen_term = numeric_to_term(slot, float(value))
        else:
            chosen_term = str(term) if term in vocab else numeric_to_term(slot, float(value or 0))

        old_conf = float(entry.get("source_conf", 0.0))
        entry["source_conf"] = _clamp01(old_conf + conf * (1.0 - old_conf))

        for t in vocab:
            if t == "source_conf":
                continue
            old_belief = float(entry.get(t, 0.0))
            if t == chosen_term:
                entry[t] = _clamp01(old_belief + conf * (1.0 - old_belief))
            else:
                entry[t] = _clamp01(old_belief * (1.0 - 0.5 * conf))

        normalized = _normalize_terms(
            {t: entry[t] for t in vocab},
            vocab,
        )
        for t in vocab:
            entry[t] = normalized[t]
        if isinstance(value, (int, float)):
            entry["_numeric"] = float(value)
        else:
            num = term_to_numeric(slot, chosen_term)
            if num is not None:
                entry["_numeric"] = num
        entry["_most_likely"] = chosen_term
        self.turn_count += 1

    def get_entropy(self, slot: str) -> float:
        """Нормированная энтропия распределения по термам слота (0 — уверенность, 1 — неопределённость)."""
        if slot == "goals":
            return 0.1 if self.goals_text else 1.0
        entry = self._slots.get(slot)
        if not entry:
            return 1.0
        vocab = SLOT_TERM_VOCAB.get(slot, [])
        probs = _normalize_terms({t: entry.get(t, 0.0) for t in vocab}, vocab)
        h = 0.0
        for p in probs.values():
            if p > 1e-9:
                h -= p * math.log2(p)
        max_h = math.log2(len(probs)) if len(probs) > 1 else 1.0
        return h / max_h if max_h > 0 else 0.0

    def get_uncertain_slots(self, threshold: float = 0.7) -> List[str]:
        """Слоты с уверенностью ниже порога."""
        uncertain: List[str] = []
        for slot in SLOT_TERM_VOCAB:
            if self.get_slot_confidence(slot) < threshold:
                uncertain.append(slot)
        if self.goals_conf < threshold and not self.goals_text:
            uncertain.append("goals")
        return uncertain

    def belief_summary(self) -> Dict[str, Dict[str, Any]]:
        """Краткое представление для API."""
        summary: Dict[str, Dict[str, Any]] = {}
        for slot in SLOT_TERM_VOCAB:
            entry = self._slots.get(slot)
            if not entry:
                continue
            vocab = SLOT_TERM_VOCAB[slot]
            terms = {t: entry.get(t, 0.0) for t in vocab}
            most = max(terms, key=terms.get) if terms else None
            summary[slot] = {
                "most_likely_term": most,
                "confidence": round(self.get_slot_confidence(slot), 3),
            }
        if self.goals_text:
            summary["goals"] = {
                "most_likely_term": self.goals_text[:80],
                "confidence": round(self.goals_conf, 3),
            }
        return summary

    def is_ready_for_recommend(self, threshold: float = 0.7) -> bool:
        for slot in REQUIRED_SLOTS:
            if self.get_slot_confidence(slot) < threshold:
                return False
        return True

    def to_user_profile(self) -> UserProfile:
        """Конвертация belief → UserProfile для recommender."""
        profile = UserProfile(
            interests=list(self.interests),
            goals=self.goals_text,
        )
        for slot in SLOT_TERM_VOCAB:
            entry = self._slots.get(slot)
            if not entry:
                continue
            if "_numeric" in entry:
                setattr(profile, slot, entry["_numeric"])
            else:
                vocab = SLOT_TERM_VOCAB[slot]
                terms = {t: entry.get(t, 0.0) for t in vocab}
                best = max(terms, key=terms.get) if terms else None
                if best:
                    num = term_to_numeric(slot, best)
                    if num is not None:
                        setattr(profile, slot, num)
        return profile

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slots": deepcopy(self._slots),
            "goals_text": self.goals_text,
            "goals_conf": self.goals_conf,
            "interests": list(self.interests),
            "turn_count": self.turn_count,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "BeliefState":
        state = cls()
        if not data:
            return state
        state._slots = deepcopy(data.get("slots", {}))
        state.goals_text = data.get("goals_text", "") or ""
        state.goals_conf = float(data.get("goals_conf", 0.0))
        state.interests = list(data.get("interests", []))
        state.turn_count = int(data.get("turn_count", 0))
        return state

    # Совместимость с прежним API (to_profile_dict, slot_entropy)
    def to_profile_dict(self) -> Dict[str, Any]:
        p = self.to_user_profile()
        return {
            "budget": p.budget,
            "knowledge_level": p.knowledge_level,
            "time_availability": p.time_availability,
            "career_focus": p.career_focus,
            "interests": p.interests,
            "goals": p.goals,
        }

    def slot_entropy(self, slot: str) -> float:
        return self.get_entropy(slot)
