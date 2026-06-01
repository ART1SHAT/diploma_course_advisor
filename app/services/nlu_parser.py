"""
NLU-парсер (§2.4): regex-паттерны для извлечения сущностей из свободного текста.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# ——— Паттерны ———
RE_BUDGET_UNTIL = re.compile(
    r"(?:до|не\s+более|макс(?:имум)?)\s*(\d[\d\s]{2,8})\s*(?:₽|руб|тыс|тысяч)?",
    re.IGNORECASE,
)
RE_BUDGET_RANGE = re.compile(
    r"бюджет\s*(\d{1,3})\s*[-–]\s*(\d{1,3})\s*(?:тыс|тысяч)",
    re.IGNORECASE,
)
RE_BUDGET_PLAIN = re.compile(
    r"(\d[\d\s]{4,9})\s*(?:₽|руб(?:\.|лей)?|р\.?|тыс(?:яч)?)?",
    re.IGNORECASE,
)
RE_TIME = re.compile(
    r"(\d{1,2})\s*(?:ч(?:ас(?:ов)?)?|h)\s*(?:в\s*неделю|/нед|неделю)?",
    re.IGNORECASE,
)
RE_KNOWLEDGE_NUM = re.compile(
    r"(?:уровень|оценка|знания?)\s*[:=]?\s*(\d{1,2})",
    re.IGNORECASE,
)

RE_ASK_RECOMMEND = re.compile(
    r"\b(рекоменд|подбер|найди|покажи|что выбрать|дай курс)\b",
    re.IGNORECASE,
)


def _budget_term_and_value(amount_rub: float) -> Tuple[str, float]:
    if amount_rub <= 0:
        return "low", 0.0
    if amount_rub < 30_000:
        return "low", amount_rub
    if amount_rub < 100_000:
        return "medium", amount_rub
    return "high", amount_rub


def _parse_budget(text: str) -> Optional[Dict[str, Any]]:
    t = text.lower()
    if re.search(r"\b(недорог\w*|дешёв\w*|дешев\w*|эконом\w*)\b", t):
        return {"term": "low", "value": 15_000.0, "confidence": 0.75}
    if re.search(r"\b(дорог|премиум|неограничен)\b", t):
        return {"term": "high", "value": 150_000.0, "confidence": 0.75}

    m_range = RE_BUDGET_RANGE.search(text)
    if m_range:
        lo, hi = float(m_range.group(1)) * 1000, float(m_range.group(2)) * 1000
        mid = (lo + hi) / 2
        term, val = _budget_term_and_value(mid)
        return {"term": term, "value": val, "confidence": 0.85}

    for pattern in (RE_BUDGET_UNTIL, RE_BUDGET_PLAIN):
        m = pattern.search(text)
        if m:
            raw = m.group(1).replace(" ", "")
            try:
                val = float(raw)
                if "тыс" in t and val < 1000:
                    val *= 1000
                term, norm = _budget_term_and_value(val)
                conf = 1.0 if "бюджет" in t or "₽" in text else 0.85
                return {"term": term, "value": norm, "confidence": conf}
            except ValueError:
                continue
    return None


def _parse_career(text: str) -> Optional[Dict[str, Any]]:
    t = text.lower()
    if re.search(r"\b(трудоустройств|для работы|работ[ауеы]|карьер|в\s+it)\b", t):
        return {"term": "employment", "value": 1.0, "confidence": 0.9}
    if re.search(r"\b(академ|егэ|университет|вуз|учёб)\b", t):
        return {"term": "academic", "value": 0.5, "confidence": 0.85}
    if re.search(r"\b(для себя|хобби|личн|саморазвит)\b", t):
        return {"term": "personal", "value": 0.0, "confidence": 0.85}
    if re.search(r"\bхочу\b", t) and re.search(r"\bработ", t):
        return {"term": "employment", "value": 1.0, "confidence": 0.8}
    return None


def _parse_time(text: str) -> Optional[Dict[str, Any]]:
    m = RE_TIME.search(text)
    if m:
        hours = float(m.group(1))
        if hours <= 4:
            return {"term": "short", "value": hours, "confidence": 0.9}
        if hours <= 10:
            return {"term": "medium", "value": hours, "confidence": 0.9}
        return {"term": "long", "value": hours, "confidence": 0.9}
    t = text.lower()
    if re.search(r"\b(мало времени|занят|редко)\b", t):
        return {"term": "short", "value": 3.0, "confidence": 0.7}
    if re.search(r"\b(много времени|интенсив|каждый день|готов учиться много)\b", t):
        return {"term": "long", "value": 15.0, "confidence": 0.75}
    return None


def _parse_knowledge(text: str) -> Optional[Dict[str, Any]]:
    m = RE_KNOWLEDGE_NUM.search(text)
    if m:
        val = min(10.0, max(0.0, float(m.group(1))))
        if val <= 3:
            term = "beginner"
        elif val <= 6:
            term = "intermediate"
        else:
            term = "advanced"
        return {"term": term, "value": val, "confidence": 0.95}
    t = text.lower()
    if re.search(r"\b(новичок|начинающ|с нуля|zero)\b", t):
        return {"term": "beginner", "value": 2.0, "confidence": 0.85}
    if re.search(r"\b(продвинут|эксперт|advanced)\b", t):
        return {"term": "advanced", "value": 8.0, "confidence": 0.85}
    if re.search(r"\b(уже знаю|средн|базов)\b", t):
        return {"term": "intermediate", "value": 5.0, "confidence": 0.8}
    return None


def _detect_intent(text: str, entities: Dict[str, Any]) -> str:
    t = text.lower()
    if RE_ASK_RECOMMEND.search(t):
        return "ask_recommend"
    if entities.get("budget") or re.search(r"\b(бюджет|руб|₽|тысяч|недорог)\b", t):
        return "set_budget"
    if entities.get("career_focus") or re.search(r"\b(хочу|цель|работ)\b", t):
        return "set_goal"
    if entities.get("time_availability") or re.search(r"\b(час|неделю|время)\b", t):
        return "set_time"
    if entities.get("knowledge_level") or re.search(r"\b(новичок|уровень|знан)\b", t):
        return "set_knowledge"
    return "unknown" if len(t.strip()) < 4 else "set_goal"


def parse_user_message(text: str) -> Dict[str, Any]:
    """
    Парсинг пользовательского сообщения.
    Возвращает intent, entities {slot: {term, value, confidence}}, raw_text.
    """
    raw = (text or "").strip()
    if not raw:
        return {"intent": "unknown", "entities": {}, "raw_text": ""}

    entities: Dict[str, Dict[str, Any]] = {}

    budget = _parse_budget(raw)
    if budget:
        entities["budget"] = budget

    career = _parse_career(raw)
    if career:
        entities["career_focus"] = career

    time_ent = _parse_time(raw)
    if time_ent:
        entities["time_availability"] = time_ent

    knowledge = _parse_knowledge(raw)
    if knowledge:
        entities["knowledge_level"] = knowledge

    # Цель — только если нет других сущностей или явная формулировка цели
    if (
        len(raw) > 25
        and "goals" not in entities
        and len(entities) <= 1
        and re.search(r"\b(цель|хочу учить|обучени)\b", raw.lower())
    ):
        entities["goals"] = {"term": raw, "value": 0.0, "confidence": 0.6}

    intent = _detect_intent(raw, entities)
    return {"intent": intent, "entities": entities, "raw_text": raw}


class NLUParser:
    """Обёртка для обратной совместимости с прежним API parse()."""

    def parse(self, text: str) -> Dict[str, Any]:
        result = parse_user_message(text)
        legacy_entities: Dict[str, Dict[str, Any]] = {}
        for slot, ent in result["entities"].items():
            if slot == "goals":
                legacy_entities["goals"] = {
                    "value": ent.get("term", result["raw_text"]),
                    "confidence": ent.get("confidence", 0.6),
                }
            else:
                legacy_entities[slot] = {
                    "value": ent.get("value", ent.get("term")),
                    "confidence": ent.get("confidence", 0.7),
                }
        confs = [e.get("confidence", 0.5) for e in legacy_entities.values()]
        overall = sum(confs) / len(confs) if confs else 0.35
        intent_map = {
            "ask_recommend": "request_recommendation",
            "set_budget": "provide_profile",
            "set_goal": "provide_profile",
            "set_time": "provide_profile",
            "set_knowledge": "provide_profile",
            "unknown": "unknown",
        }
        return {
            "intent": intent_map.get(result["intent"], "provide_profile"),
            "entities": legacy_entities,
            "confidence": round(min(1.0, overall), 3),
        }


nlu_parser = NLUParser()
