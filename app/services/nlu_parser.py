"""
NLU-парсер: regex + лёгкие эмбеддинги (bag-of-words) для свободного текста.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ——— Regex-паттерны ———
RE_BUDGET = re.compile(
    r"(?:бюджет|до|не\s+более|макс(?:имум)?)\s*"
    r"(\d[\d\s]{2,8})\s*(?:₽|руб(?:\.|лей)?|р\.?)?",
    re.IGNORECASE,
)
RE_BUDGET_PLAIN = re.compile(
    r"(\d[\d\s]{4,9})\s*(?:₽|руб(?:\.|лей)?|р\.?)",
    re.IGNORECASE,
)
RE_TIME = re.compile(
    r"(\d{1,2})\s*(?:ч(?:ас(?:ов)?)?|h)\s*(?:в\s*неделю|/нед|неделю)?",
    re.IGNORECASE,
)
RE_KNOWLEDGE_LEVEL = re.compile(
    r"(?:уровень|оценка|знания?)\s*[:=]?\s*(\d{1,2})(?:\s*/\s*10)?",
    re.IGNORECASE,
)
RE_KNOWLEDGE_WORDS = re.compile(
    r"\b(новичок|начинающ|средн|продвинут|эксперт|advanced|beginner)\b",
    re.IGNORECASE,
)

# ——— Эмбеддинги термов (bag-of-words, офлайн) ———
TERM_VECTORS: Dict[str, Dict[str, float]] = {
    "career_employment": {
        "работ": 1.0,
        "карьер": 1.0,
        "трудоустрой": 1.0,
        "it": 0.8,
        "професс": 0.9,
        "разработ": 0.7,
        "зарплат": 0.6,
    },
    "career_academic": {
        "егэ": 1.0,
        "университет": 1.0,
        "академ": 1.0,
        "экзамен": 0.9,
        "вуз": 0.9,
        "диплом": 0.7,
    },
    "career_personal": {
        "хобби": 1.0,
        "для себя": 1.0,
        "интерес": 0.8,
        "удовольств": 0.7,
        "саморазвит": 0.9,
    },
    "knowledge_beginner": {"новичок": 1.0, "начинающ": 1.0, "zero": 0.8, "с нуля": 1.0},
    "knowledge_intermediate": {"средн": 1.0, "базов": 0.8, "intermediate": 1.0},
    "knowledge_advanced": {"продвинут": 1.0, "эксперт": 1.0, "advanced": 1.0},
    "time_short": {"мало": 0.8, "редко": 0.7, "1-2": 0.6, "занят": 0.5},
    "time_long": {"много": 0.9, "полный": 0.7, "интенсив": 0.6, "каждый день": 0.8},
}

CAREER_TERM_MAP = {
    "career_employment": ("career_focus", "employment", 1.0),
    "career_academic": ("career_focus", "academic", 0.5),
    "career_personal": ("career_focus", "personal", 0.0),
}

KNOWLEDGE_TERM_MAP = {
    "knowledge_beginner": ("knowledge_level", 2.0),
    "knowledge_intermediate": ("knowledge_level", 5.0),
    "knowledge_advanced": ("knowledge_level", 8.0),
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def _bow_vector(tokens: List[str]) -> Dict[str, float]:
    vec: Dict[str, float] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in set(a) | set(b))
    return max(0.0, min(1.0, dot))


def _embed_match(text: str, label: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    doc_vec = _bow_vector(tokens)
    proto = TERM_VECTORS.get(label, {})
    proto_vec = _bow_vector(list(proto.keys()))
    # Веса прототипа
    weighted = {k: proto.get(k, 0.5) for k in proto}
    weighted_vec = _bow_vector(list(weighted.keys()))
    for k, w in weighted.items():
        if k in weighted_vec:
            weighted_vec[k] *= w
    return _cosine(doc_vec, weighted_vec)


def _parse_budget(text: str) -> Optional[Tuple[float, float]]:
    for pattern in (RE_BUDGET, RE_BUDGET_PLAIN):
        m = pattern.search(text)
        if m:
            raw = m.group(1).replace(" ", "")
            try:
                val = float(raw)
                if val > 0:
                    conf = 0.9 if "бюджет" in text.lower() or "₽" in text or "руб" in text.lower() else 0.75
                    return val, conf
            except ValueError:
                continue
    if re.search(r"\bбесплатн", text, re.I):
        return 0.0, 0.85
    return None


def _parse_time(text: str) -> Optional[Tuple[float, float]]:
    m = RE_TIME.search(text)
    if m:
        return float(m.group(1)), 0.85
    score_long = _embed_match(text, "time_long")
    score_short = _embed_match(text, "time_short")
    if score_long > 0.45 and score_long > score_short:
        return 12.0, score_long
    if score_short > 0.45:
        return 3.0, score_short
    return None


def _parse_knowledge(text: str) -> Optional[Tuple[float, float]]:
    m = RE_KNOWLEDGE_LEVEL.search(text)
    if m:
        val = float(m.group(1))
        return min(10.0, max(0.0, val)), 0.9
    m2 = RE_KNOWLEDGE_WORDS.search(text)
    if m2:
        word = m2.group(1).lower()
        if any(x in word for x in ("нович", "начина", "beginner")):
            return 2.0, 0.8
        if any(x in word for x in ("продвин", "эксперт", "advanced")):
            return 8.0, 0.8
        return 5.0, 0.75
    best_val, best_conf = None, 0.0
    for label, (_, num) in KNOWLEDGE_TERM_MAP.items():
        sc = _embed_match(text, label)
        if sc > best_conf:
            best_conf = sc
            best_val = num
    if best_val is not None and best_conf > 0.4:
        return best_val, best_conf
    return None


def _parse_career(text: str) -> Optional[Tuple[str, float, float]]:
    best: Optional[Tuple[str, float, float]] = None
    best_score = 0.0
    for label, (slot, term, numeric) in CAREER_TERM_MAP.items():
        sc = _embed_match(text, label)
        if sc > best_score:
            best_score = sc
            best = (term, numeric, sc)
    if best and best_score > 0.35:
        return best
    if re.search(r"\b(работ|карьер|it|трудоустрой)", text, re.I):
        return "employment", 1.0, 0.7
    if re.search(r"\b(егэ|университет|вуз)", text, re.I):
        return "academic", 0.5, 0.7
    if re.search(r"\b(хобби|для себя)", text, re.I):
        return "personal", 0.0, 0.7
    return None


def detect_intent(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(рекоменд|подбер|найди|покажи курс|что выбрать)\b", t):
        return "request_recommendation"
    if re.search(r"\b(бюджет|руб|₽|уровень|часов|цель|хочу учить)\b", t):
        return "provide_profile"
    return "provide_profile" if len(t.strip()) > 3 else "unknown"


class NLUParser:
    """Парсер свободного текста → intent + entities + confidence."""

    def parse(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {"intent": "unknown", "entities": {}, "confidence": 0.0}

        intent = detect_intent(text)
        entities: Dict[str, Dict[str, Any]] = {}
        confidences: List[float] = []

        budget = _parse_budget(text)
        if budget:
            val, conf = budget
            entities["budget"] = {"value": val, "confidence": conf}
            confidences.append(conf)

        knowledge = _parse_knowledge(text)
        if knowledge:
            val, conf = knowledge
            entities["knowledge_level"] = {"value": val, "confidence": conf}
            confidences.append(conf)

        time_val = _parse_time(text)
        if time_val:
            val, conf = time_val
            entities["time_availability"] = {"value": val, "confidence": conf}
            confidences.append(conf)

        career = _parse_career(text)
        if career:
            term, numeric, conf = career
            entities["career_focus"] = {"value": term, "confidence": conf, "numeric": numeric}
            confidences.append(conf)

        # Цель — остаток длинного текста без числовых сущностей
        if len(text) > 25 and "goals" not in entities:
            entities["goals"] = {"value": text, "confidence": 0.55}
            confidences.append(0.55)

        overall = sum(confidences) / len(confidences) if confidences else 0.35
        return {
            "intent": intent,
            "entities": entities,
            "confidence": round(min(1.0, overall), 3),
        }


# Singleton
nlu_parser = NLUParser()
